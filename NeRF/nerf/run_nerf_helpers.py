import torch

# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)#一个浮点数数组转换为 8 位无符号整数数组 clip() 函数将像素值限制在 0 到 1 之间，然后将像素值乘以 255 并转换为无符号整数类型 np.uint8


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:#嵌入函数是否包括原始的3个
            embed_fns.append(
                lambda x: x
            )  # 原封不动的嵌入方式，它将 3D 点坐标作为嵌入向量的值，从而避免了使用正弦和余弦函数的计算。
            out_dim += d  # 每次+3

        max_freq = self.kwargs["max_freq_log2"]#9
        N_freqs = self.kwargs["num_freqs"]#10

        if self.kwargs["log_sampling"]:#频率系数 对数间隔 指数的
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)#([0~9]int) 长度是这么多
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)#线性间隔确实 隔50均分的嘛

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:#γ(p) = ( sin(2^0πp), cos(2^0πp), · · · , sin(2^L−1 πp), cos(2^L−1 πp) )
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat(
            [fn(inputs) for fn in self.embed_fns], -1
        )  # 这个方法是个列表，有了对应的函数对数值进行操作,-1最后一个维度拼接


# 为什么是最后一个维度拼接，分别进行了3个坐标都进行了20D的变换,这样就能变成一个60D的向量了?


def get_embedder(multires, i=0):#10 频率L的数量 L = 10 for γ(x) and L = 4 for γ(d)
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        "include_input": True,
        "input_dims": 3,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(
        x
    )  # 接受x，传递给类的embed方法处理  使用eo类的.embed(x)方法
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        input_ch_views=3,
        output_ch=4,
        skips=[4],
        use_viewdirs=False,
    ):
        """ """
        super(NeRF, self).__init__()
        self.D = D  # D=args.netdepth
        self.W = W  # W=args.netwidth
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.pts_linears = nn.ModuleList(##8个线性层的网络
            [nn.Linear(input_ch, W)]
            + [
                nn.Linear(W, W)
                if i not in self.skips
                else nn.Linear(W + input_ch, W) 
                for i in range(D - 1)
            ]
        )

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
        ##todo funny things
        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:  # 有无相机
            self.feature_linear = nn.Linear(W, W)#W=256
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)#todo 这个是啥意思

    def forward(self, x):  # x(65536,90)
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1
        )
        h = input_pts#(65536,63)
        for i, l in enumerate(self.pts_linears): #i:0 l:Linear(in_features=63, out_features=256, bias=True)
            h = self.pts_linears[i](h)
            h = F.relu(h)#relu过线性层
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)#后面的都是256->256 除了skip后是63+256->256 i=4

        if self.use_viewdirs:#获得最终的rgb
            alpha = self.alpha_linear(h)#(65536,256)->  (65536,1) 
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)#(65536,256+27)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)#283->128
                h = F.relu(h)

            rgb = self.rgb_linear(h)#128->3
            outputs = torch.cat([rgb, alpha], -1)#(65536,4)
        else:
            outputs = self.output_linear(h)

        return outputs#(65536,4)

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears])
            )
            self.pts_linears[i].bias.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears + 1])
            )

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear])
        )
        self.feature_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear + 1])
        )

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears])
        )
        self.views_linears[0].bias.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears + 1])
        )

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear])
        )
        self.rgb_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear + 1])
        )

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear])
        )
        self.alpha_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear + 1])
        )


# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H)
    )  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack(
        [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1
    )#得到图像的HW网格 内参矩阵转化 (400,400,3) 最后一列-1
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs] (400,400,1,3)*(3,4)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape) #变形矩阵最后一列
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):#np实现的
    i, j = np.meshgrid(#像素尺寸 
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )  # (400,400)[0,1,2,3,...,400]~[0,0,0,0,...,0] 400行列
    dirs = np.stack(
        [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1
    )  # 像素坐标K处理 xy缩放和平移   ##(400,400,3) #0的话是(3,400,400) 一个矩阵全是-1可还行 0是按自己的列可能单拎出来 -1是在最后叠
    # Rotate ray directions from camera frame  to the world frame
    rays_d = np.sum(  # (400,400,1,3)*(3,3)  (3是3列xy与-1)（1是单纯多一个包了3变成[[x,y,-1]] )(400就是400个[[x,y,-1]])
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1  # (1,3)与c2w(3,3)逐元素 每行求值 (400,400,3,3)列加后行当列 (1,3)(3,3) 可能这样点乘后+就是c2w@dir吧(3,3)(3,1)最后的都是减去然后sum
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))  # (400,400,3)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = (
        -1.0
        / (W / (2.0 * focal))
        * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    )
    d1 = (
        -1.0
        / (H / (2.0 * focal))
        * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    )
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf   bins(1024,63) weights(1024,62) N_samples=128
    weights = weights + 1e-5  # prevent nans #去头去尾的权重
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (1024,62) / (1024,1) = (1024,62)
    cdf = torch.cumsum(pdf, -1)#(1024,62)累加到1在每一行的方向上 0.16,...1.00 0维度的话就从上往下累加了最后也不是1
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))(1024,64)

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples)#128
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])#->(1024,128) 复制了1024行
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0.0, 1.0, N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF  #u (1024,128) cdf(1024,63) inds b a(1024,128) #todo我理解不了
    u = u.contiguous()#变成连续张量连续内存块 随机生成的可能不是连续的 性能下降不然 
    inds = torch.searchsorted(cdf, u, right=True)#计算了u张量在cdf张量中的索引 cdf张量是颜色 right是代表右侧二分 inds 变量表示每个采样点的颜色值在颜色值累积分布函数（CDF）中的索引 
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)#与0比较每个位置去处inds-1和0的最大值？也没有inds=0呀torch.nonzero(tensor == 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)#62矩阵和inds矩阵取小
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)(1024,128,2) 前面的128行都当作后面矩阵的2列之一

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]] #[1024,128,63] 63维度 inds来找对应概率
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)#gather获取响应的概率值 unsqz(1)(1024,1,63)2维度加个维度 cdf[...,None,:]也同理但类型不一样
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)#同上 bins(1024,63)
    #cdf_g(1024,128,2) bins_g(1024,128,2)
    denom = cdf_g[..., 1] - cdf_g[..., 0]#(1024,128)
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)#小于这个值变成1 避免除以0
    t = (u - cdf_g[..., 0]) / denom#都是小于1的值
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])##todo 这个采样看不懂

    return samples#(1024,128)
