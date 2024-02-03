import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

from torch.utils.tensorboard import SummaryWriter


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    ##返回调用函数fn(embedding)
    def ret(inputs):
        return torch.cat(  # (65536,90)
            [fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0
        )

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):#inputs pts rays_o+d*z_val
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = torch.reshape(
        inputs, [-1, inputs.shape[-1]]
    )  # pts(1024,64,3)->(65536,3) 这里是线+64采样后的
    embedded = embed_fn(inputs_flat)  # (65536,63) 拓展到63维度

    if viewdirs is not None:#(1024,3)
        input_dirs = viewdirs[:, None].expand(inputs.shape)  # (1024,1,3)->(1024,64,3) None在:维度后加
        input_dirs_flat = torch.reshape(
            input_dirs, [-1, input_dirs.shape[-1]]
        )  # (65536,3)
        embedded_dirs = embeddirs_fn(input_dirs_flat)  # (65536,27)
        embedded = torch.cat([embedded, embedded_dirs], -1)  # (65536,63+27)
    #全都flat 1024*64samples fn是网络的函数进了forward 如果没有用chunk直接fn单个 如果用了的话 每个小chunk fn
    outputs_flat = batchify(fn, netchunk)(embedded)  ##todo flat(65536,4)这里return一个函数ret 所以在里面定义了函数 fn找不到函数 在kwargs 
    outputs = torch.reshape( # list[shape] [1024,64,4]reshape的list就行
        outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]#[1024,64,4]
    )#flat的转化为原来的数据
    return outputs#(1024,64,4)再次分成64sample的情况 65536的 这个是rgb+alpha的4


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):#todo render_rays大工程 sample pdf
    """Render rays in smaller minibatches to avoid OOM.OO memory"""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):  # flat 1024
        ret = render_rays(rays_flat[i : i + chunk], **kwargs)  # i+32768
        for k in ret:#return得到了第一次渲染的结果 rgb disp acc z_sample_std等
            if k not in all_ret:
                all_ret[k] = []#第一次为这个变量新建一个列表
            all_ret[k].append(ret[k])#后面为这个列表append一个新值

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}#list[tensor]->tenser
    return all_ret


def render(
    H,
    W,
    K,
    chunk=1024 * 32,
    rays=None,
    c2w=None,
    ndc=True,
    near=0.0,
    far=1.0,
    use_viewdirs=False,
    c2w_staticcam=None,
    **kwargs,
):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None: 
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays  # (2,1024,3)

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(
            viewdirs, dim=-1, keepdim=True
        )  #除以L2范数得到方向 归一
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()#(1024,3)

    sh = rays_d.shape  # [..., 3](1024,3)
    if ndc:  # todo if skip了
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1.0, rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()  # (1024,3)
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(
        rays_d[..., :1]
    ), far * torch.ones_like(  # (1024,1)  rays_d[...,0]的话是(1024,)
        rays_d[..., :1]  # 0列 或者说到0~1开列的话有什么意味吗 写法习惯吧 这样是2维 单写0的1维度取出来
    )
    rays = torch.cat([rays_o, rays_d, near, far], -1)  # 最里的列(1024,3+3+1+1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)#(1024,3+3+1+1+3) L2后的rays_d #todo可能再多一个先验

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:#todo不能理解这样reshape有什么用不一直是这个结果吗 可能结果更规范吧
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])#[1024,+3]  rays_d的1024维度 + 此时k不要第一的1024的组合维度#todo真的有维度不对应吗
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ["rgb_map", "disp_map", "acc_map"]
    ret_list = [all_ret[k] for k in k_extract]#3个tensor
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}#5的字典
    return ret_list + [ret_dict]#[3个tensor,5个元素字典]构成的列表


def render_path(#todo 渲染结果未看
    render_poses,
    hwf,
    K,
    chunk,
    render_kwargs,
    gt_imgs=None,
    savedir=None,
    render_factor=0,
):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(
            H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs
        )
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, "{:03d}.png".format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model."""
    embed_fn, input_ch = get_embedder(  ## function channel
        args.multires, args.i_embed
    )  # multi-res决定了正余弦log的数量和频率

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4#第五个是重要度 前三颜色 4σ
    skips = [4]
    model = NeRF(
        D=args.netdepth,  # 8
        W=args.netwidth,  # 256
        input_ch=input_ch,#63  10*2*3+3
        output_ch=output_ch,
        skips=skips,
        input_ch_views=input_ch_views,  # 27 (4*2+1)*3
        use_viewdirs=args.use_viewdirs,
    ).to(device)
    grad_vars = list(model.parameters()) #24

    model_fine = None  # 精细模型 如果N_importance=>number of additional fine samples per ray
    if args.N_importance > 0:
        model_fine = NeRF(
            D=args.netdepth_fine,  # 8
            W=args.netwidth_fine,  # 256
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,  # 27
            use_viewdirs=args.use_viewdirs,
        ).to(device)
        grad_vars += list(model_fine.parameters())  ##24+24

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
        inputs,
        viewdirs,
        network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk,  # 1024*64神经网络块大小 加速计算 N_samples 64 N_importance 128
    )#todo 这里调试看看batchfy

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f)
            for f in sorted(os.listdir(os.path.join(basedir, expname)))
            if "tar" in f
        ]
    # 看ckpts是不是存在
    print("Found ckpts", ckpts)
    ##有ckpt且不是no_reload不重新加载 也就是 有且重新加载
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Load model
        model.load_state_dict(ckpt["network_fn_state_dict"])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt["network_fine_state_dict"])

    ##########################

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "perturb": args.perturb,  # 1.0
        "N_importance": args.N_importance,  # 128
        "network_fine": model_fine,
        "N_samples": args.N_samples,  # 64
        "network_fn": model,
        "use_viewdirs": args.use_viewdirs,
        "white_bkgd": args.white_bkgd,  # True
        "raw_noise_std": args.raw_noise_std,  # 0
    }
    ##todo 不是forward facing的过程所需要的采用方式 lindisp的理解
    # NDC only good for LLFF-style forward facing data LLFF的同意坐标话表示 Norm Device Cooordinates
    if args.dataset_type != "llff" or args.no_ndc:
        print("Not ndc!")
        render_kwargs_train["ndc"] = False
        render_kwargs_train["lindisp"] = args.lindisp
    # sampling linearly in disparity rather than depth lin-disp 视差向量 视差计算相机视角方向和3D坐标之间的距离
    # 将相机视角方向和 3D 点坐标转换为归一化设备坐标系中的坐标，然后计算它们之间的差值 俩个不同的角度的差值呗
    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train
    }  # copy了一份train
    render_kwargs_test["perturb"] = False  # todo perturb变成0是什么意思 train是1
    render_kwargs_test["raw_noise_std"] = 0.0  # 这个没变

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]#(1024,63) 到尾巴减去头起的 2~6 delta
    dists = torch.cat(#值都是0.0X左右了
        [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1
    )  # [N_rays, N_samples]  # dists (1024,63+1)
    #rays_d[..., None, :] (1024,3) norm前(1024,1,3)->(1024,1)  #dists (1024,64)/L2 平方和的平方根
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)  ##todo算出来是什么东西呀 是那个delta
   

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3](1024,64,3) raw (1024,64,4)取了前三个
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]   #dists(1024,64) alpha(1024)拿出来 #todo求出来Wi/Ti的累和 1-exp(-xx)的部分
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)##todo累计积从上到下积 这个求T 1-alpha得到exp(-xx)的部分的T
    weights = (#alpha(1024,64)  #cumprod(1024,65)第一列全是1,沿着列计算累计积 算出来是(1024,65)的 应该严格递减 第一列补1来累乘
        alpha* torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.0 - alpha + 1e-10], -1), -1)[:, :-1]
    )
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]   rgb (1024,64,3) weights (1024,64)  64维度求和->(1024,3)

    depth_map = torch.sum(weights * z_vals, -1)#(1024,64)->(1024) 2~6weight与z_val的累和 5.8 加权深度值
    disp_map = 1.0 / torch.max(
        1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)#1024个1 997的和部分不是1 基本上都是后面的吧 得到逆深度
    )
    acc_map = torch.sum(weights, -1)#(1024,64)->(1024) #todo权重和 997,5731部分权重和不是1 密度和delta不均匀

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])#(1024,3)+(1024,1) 26.4269/1024基本上都是很小的1e-7数量

    return rgb_map, disp_map, acc_map, weights, depth_map#(1024,3) (1024) (1024) (1024,64) (1024)


def render_rays(
    ray_batch,
    network_fn,
    network_query_fn,
    N_samples,
    retraw=False,
    lindisp=False,
    perturb=0.0,
    N_importance=0,
    network_fine=None,
    white_bkgd=False,
    raw_noise_std=0.0,
    verbose=False,
    pytest=False,
):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]  # 1024
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = (
        ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    )  # viewdirs(1024,3)
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])  # (1024,2)->(1024,1,2)多了一维度 后面是二维的了
    near, far = bounds[..., 0], bounds[..., 1]  # 2&6重新取出来 (1024,1)
    ##todo 线性深度值和对数深度值
    t_vals = torch.linspace(0.0, 1.0, steps=N_samples)  # 64  64长的线性取样
    if not lindisp:
        z_vals = near * (1.0 - t_vals) + far * (t_vals)  # (1024,64)
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])  # 扩张到指定大小咋还是(1024,64) 好像本来就是这个

    if perturb > 0.0:
        # get intervals between samples
        mids = 0.5 * (
            z_vals[..., 1:] + z_vals[..., :-1]
        )  # 去头+去尾  2~6的线性64列的变化->(1024,63)
        upper = torch.cat([mids, z_vals[..., -1:]], -1)  # +尾
        lower = torch.cat([z_vals[..., :1], mids], -1)  # +头
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)
        # 随机扰动 不是啥delta
        z_vals = lower + (upper - lower) * t_rand  # 求每个点的差值 然后乘以随机数01 加上头的基础值

    pts = (  ## 竟然乘成了这样
        rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    )  # [N_rays, N_samples, 3]#(1024,1,3)*(1024,64,1)->(1024,64,3)   (1024,1,3)+(1024,64,3)广播复制->(1024,64,3)
    ##todo这里可能是转化成了世界坐标系中的点了 平移d+坐标原点o的位置

    #     raw = run_network(pts)   
    raw = network_query_fn(pts, viewdirs, network_fn)#raw (1024,64,4) 返回完了rgb+alpha的采样N_sample=64了
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
    )#z_vals (1024,64)  rays_d (1024,3)  raw_noise_std 0.0  white_bkgd True pytest False

    if N_importance > 0:#128 可能是额外采样的点？
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        #在 Nerf 模型中，det 变量表示是否进行确定性渲染。具体来说，当 perturb 变量的值为 0.0 时，表示不进行随机扰动，即进行确定性渲染，此时 det 变量的值为 True。当 perturb 变量的值不为 0.0 时，表示进行随机扰动，即进行随机渲染
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])#z_vals(1024,64)已经不再是2起6尾了 对于采样次数再取均值可能
        z_samples = sample_pdf(#todo 没看pdf太复杂
            z_vals_mid,
            weights[..., 1:-1],#todo为什么去头去尾的权重
            N_importance,
            det=(perturb == 0.0),
            pytest=pytest,
        )
        z_samples = z_samples.detach()#(1024,128)采样深度值 计算图分离 #todo二次采样得到的位置
        
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)#(1024,64+128)排序 sort范围一个排序结果 和一个indices 应当是单增 可能是与之前的对应关系 后面的要插到前面去
        pts = (
            rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        )  # [N_rays, N_samples + N_importance, 3] (1024,64+128,3)

        run_fn = network_fn if network_fine is None else network_fine
        #         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)#fine

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
        )

    ret = {"rgb_map": rgb_map, "disp_map": disp_map, "acc_map": acc_map}#return values 0代表第一次采样此处已经是fine后的结果了
    if retraw:
        ret["raw"] = raw
    if N_importance > 0:
        ret["rgb0"] = rgb_map_0#(1024,3)
        ret["disp0"] = disp_map_0#1024
        ret["acc0"] = acc_map_0#1024
        ret["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays] #z_samples (1024,128)

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():
    import configargparse

    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument(
        "--basedir", type=str, default="./logs/", help="where to store ckpts and logs"
    )
    parser.add_argument(
        "--datadir", type=str, default="./data/llff/fern", help="input data directory"
    )

    # training options
    parser.add_argument("--netdepth", type=int, default=8, help="layers in network")
    parser.add_argument("--netwidth", type=int, default=256, help="channels per layer")
    parser.add_argument(
        "--netdepth_fine", type=int, default=8, help="layers in fine network"
    )
    parser.add_argument(
        "--netwidth_fine",
        type=int,
        default=256,
        help="channels per layer in fine network",
    )
    parser.add_argument(
        "--N_rand",
        type=int,
        default=32 * 32 * 4,
        help="batch size (number of random rays per gradient step)",
    )
    parser.add_argument("--lrate", type=float, default=5e-4, help="learning rate")
    parser.add_argument(
        "--lrate_decay",
        type=int,
        default=250,
        help="exponential learning rate decay (in 1000 steps)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=1024 * 32,
        help="number of rays processed in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--netchunk",
        type=int,
        default=1024 * 64,
        help="number of pts sent through network in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--no_batching",
        action="store_true",  # config里面出现了 train的选项 指定了的就是true no_batching = True 如果写no的false那是不是纯nt行为
        help="only take random rays from 1 image at a time",
    )
    parser.add_argument(
        "--no_reload", action="store_true", help="do not reload weights from saved ckpt"
    )  ##todo为什么用no_reload 可能是因为不输入的话就是false 尝试重新加载 没出现就是false
    parser.add_argument(
        "--ft_path",
        type=str,
        default=None,
        help="specific weights npy file to reload for coarse network",
    )

    # rendering options
    parser.add_argument(
        "--N_samples", type=int, default=64, help="number of coarse samples per ray"
    )
    parser.add_argument(
        "--N_importance",
        type=int,
        default=0,
        help="number of additional fine samples per ray",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="set to 0. for no jitter, 1. for jitter",
    )
    parser.add_argument(
        "--use_viewdirs", action="store_true", help="use full 5D input instead of 3D"
    )
    parser.add_argument(
        "--i_embed",
        type=int,
        default=0,
        help="set 0 for default positional encoding, -1 for none",
    )
    parser.add_argument(
        "--multires",
        type=int,
        default=10,
        help="log2 of max freq for positional encoding (3D location)",
    )
    parser.add_argument(
        "--multires_views",
        type=int,
        default=4,
        help="log2 of max freq for positional encoding (2D direction)",
    )
    parser.add_argument(
        "--raw_noise_std",
        type=float,
        default=0.0,
        help="std dev of noise added to regularize sigma_a output, 1e0 recommended",
    )

    parser.add_argument(
        "--render_only",
        action="store_true",
        help="do not optimize, reload weights and render out render_poses path",
    )
    parser.add_argument(
        "--render_test",
        action="store_true",
        help="render the test set instead of render_poses path",
    )
    parser.add_argument(
        "--render_factor",
        type=int,
        default=0,
        help="downsampling factor to speed up rendering, set 4 or 8 for fast preview",
    )

    # training options
    parser.add_argument(
        "--precrop_iters",
        type=int,
        default=0,
        help="number of steps to train on central crops",
    )
    parser.add_argument(
        "--precrop_frac",
        type=float,
        default=0.5,
        help="fraction of img taken for central crops",
    )

    # dataset options
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="llff",
        help="options: llff / blender / deepvoxels",
    )
    parser.add_argument(
        "--testskip",
        type=int,
        default=8,
        help="will load 1/N images from test/val sets, useful for large datasets like deepvoxels",
    )

    ## deepvoxels flags
    parser.add_argument(
        "--shape",
        type=str,
        default="greek",
        help="options : armchair / cube / greek / vase",
    )

    ## blender flags
    parser.add_argument(
        "--white_bkgd",
        action="store_true",
        help="set to render synthetic data on a white bkgd (always use for dvoxels)",
    )
    parser.add_argument(
        "--half_res",
        action="store_true",
        help="load blender synthetic data at 400x400 instead of 800x800",
    )

    ## llff flags
    parser.add_argument(
        "--factor", type=int, default=8, help="downsample factor for LLFF images"
    )
    parser.add_argument(
        "--no_ndc",
        action="store_true",
        help="do not use normalized device coordinates (set for non-forward facing scenes)",
    )
    parser.add_argument(
        "--lindisp",
        action="store_true",
        help="sampling linearly in disparity rather than depth",
    )
    parser.add_argument(
        "--spherify", action="store_true", help="set for spherical 360 scenes"
    )
    parser.add_argument(
        "--llffhold",
        type=int,
        default=8,
        help="will take every 1/N images as LLFF test set, paper uses 8",
    )

    # logging/saving options
    parser.add_argument(
        "--i_print",
        type=int,
        default=100,
        help="frequency of console printout and metric loggin",
    )
    parser.add_argument(
        "--i_img", type=int, default=500, help="frequency of tensorboard image logging"
    )
    parser.add_argument(
        "--i_weights", type=int, default=10000, help="frequency of weight ckpt saving"
    )
    parser.add_argument(
        "--i_testset", type=int, default=50000, help="frequency of testset saving"
    )
    parser.add_argument(
        "--i_video",
        type=int,
        default=50000,
        help="frequency of render_poses video saving",
    )

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()

    # Load data如何在tmux使用F5执行vscode里面的debug
    K = None
    if args.dataset_type == "llff":
        images, poses, bds, render_poses, i_test = load_llff_data(
            args.datadir,
            args.factor,
            recenter=True,
            bd_factor=0.75,
            spherify=args.spherify,
        )
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print("Loaded llff", images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print("Auto LLFF holdout,", args.llffhold)
            i_test = np.arange(images.shape[0])[:: args.llffhold]

        i_val = i_test
        i_train = np.array(
            [
                i
                for i in np.arange(int(images.shape[0]))
                if (i not in i_test and i not in i_val)
            ]
        )

        print("DEFINING BOUNDS")
        if args.no_ndc:
            near = np.ndarray.min(bds) * 0.9
            far = np.ndarray.max(bds) * 1.0

        else:
            near = 0.0
            far = 1.0
        print("NEAR FAR", near, far)

    elif args.dataset_type == "blender":
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip
        )#skip 8 half_res true
        print("Loaded blender", images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.0
        far = 6.0

        if args.white_bkgd:#RGBA转为RGB
            images = images[..., :3] * images[..., -1:] + (
                1.0 - images[..., -1:]
            )  # 白色3channel 透明度处理
        else:
            images = images[..., :3]

    elif args.dataset_type == "LINEMOD":
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(
            args.datadir, args.half_res, args.testskip
        )
        print(f"Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}")
        print(f"[CHECK HERE] near: {near}, far: {far}.")
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == "deepvoxels":
        images, poses, render_poses, hwf, i_split = load_dv_data(
            scene=args.shape, basedir=args.datadir, testskip=args.testskip
        )

        print("Loaded deepvoxels", images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.0
        far = hemi_R + 1.0

    else:
        print("Unknown dataset type", args.dataset_type, "exiting")
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:#内参矩阵 转换为像素
        K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

    if args.render_test:
        render_poses = np.array(poses[i_test])#外参矩阵 render 40个 poses共138个(4,4) 相机转世界
  
    # Create log dir and copy the config file 抄config保存
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, "config.txt")
        with open(f, "w") as file:
            file.write(open(args.config, "r").read()) 

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(
        args
    )
    global_step = start

    bds_dict = {
        "near": near,
        "far": far,
    }  # "near"：相机视锥体的近平面距离。 #"far"：相机视锥体的远平面距离。
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)  # (40,4,4)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print("RENDER ONLY")
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(
                basedir,
                expname,
                "renderonly_{}_{:06d}".format(
                    "test" if args.render_test else "path", start
                ),
            )
            os.makedirs(testsavedir, exist_ok=True)
            print("test poses shape", render_poses.shape)

            rgbs, _ = render_path(
                render_poses,
                hwf,
                K,
                args.chunk,
                render_kwargs_test,
                gt_imgs=images,
                savedir=testsavedir,
                render_factor=args.render_factor,
            )
            print("Done rendering", testsavedir)
            imageio.mimwrite(
                os.path.join(testsavedir, "video.mp4"), to8b(rgbs), fps=30, quality=8
            )

            return
    #todo 这里开始N_rand光线
    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand  # 1024
    use_batching = not args.no_batching  #todo 原来没有用batching 这个batching的是图片感觉  
    if use_batching:
        # For random ray batching
        print("get rays")#todo ro旋转 rd平移 rgb
        rays = np.stack(
            [get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0
        )  # [N, ro+rd, H, W, 3](138,2,400,400,3) #ro旋转 rd平移 2维度列表 stack 138个
        print("done, concats")#世界坐标系
        rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+new, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+new, 3]
        rays_rgb = np.stack(
            [rays_rgb[i] for i in i_train], 0
        )  # train images only(100,400,400,3,3)
        rays_rgb = np.reshape(
            rays_rgb, [-1, 3, 3]
        )  # [(N-1)*H*W, ro+rd+rgb, 3](16000000,3,3)N-1?
        rays_rgb = rays_rgb.astype(np.float32)
        print("shuffle rays")
        np.random.shuffle(rays_rgb)  

        print("done")
        i_batch = 0

    # Move training data to GPU
    if use_batching:#前面是NP后面to tensor
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    N_iters = 200000 + 1
    print("Begin")
    print("TRAIN views are", i_train)
    print("TEST views are", i_test)
    print("VAL views are", i_val)

    ##todo 我改看看tensorboard
    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, "summaries", expname))
    step0 = 0
    ##todo
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:            # Random over   all images# [Batch, 2+1, 3*?](1024,ro+rd+rgb,3)
            batch = rays_rgb[i_batch : i_batch + N_rand]  #每次取1024个就完事 1600000shuffle后
            batch = torch.transpose(
                batch, 0, 1
            )  # (3,1024,3) 后面俩才是要换是维度 多个可以用列表传入 rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
            batch_rays, target_s = batch[:2], batch[2]  # 取rt rd(2,1024,3) (1024,3)左闭右开感觉 2是ro+rd 3是rgb的数据 1024*3的大小

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:  # 重新刷新
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])  # 随机排列完整的张量
                rays_rgb = rays_rgb[rand_idx]  # (16000000,3,3)
                i_batch = 0

        else:#todo单个
            # Random from one image
            img_i = np.random.choice(i_train)#选择里面的一个index
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3, :4]#(3,4) 外参

            if N_rand is not None:#外参矩阵转化后的 旋转变化的ray
                rays_o, rays_d = get_rays(
                    H, W, K, torch.Tensor(pose)
                )  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:#i是循环的次数
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW),
                        ),
                        -1,
                    )#(200,200,2)的一个坐标轴
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}"
                        )
                else:
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)
                        ),
                        -1,
                    )  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(
                    coords.shape[0], size=[N_rand], replace=False
                )  # (N_rand,) 挑选坐标轴中1024的inds 一个维度的数字 挑选出坐标 1024个然后得到光线
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)#(2,N_rand,3) 就是每一行堆在一起了
                target_s = target[
                    select_coords[:, 0], select_coords[:, 1]
                ]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(
            H,
            W,
            K,
            chunk=args.chunk,
            rays=batch_rays,#no batching是一张图片里面的坐标和点 batching
            verbose=i < 10,
            retraw=True,
            **render_kwargs_train,
        )#return了[3个tensor和5个元素大小的字典] #todo[]+[]最后竟然能分开成这俩个

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras["raw"][..., -1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if "rgb0" in extras:
            img_loss0 = img2mse(extras["rgb0"], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)
        ##todo
        writer.add_scalar("loss", loss, global_step=step0)
        step0 += 1
        ##todo
        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update le arning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate##todo每次迭代会有很多的group吗 第一次发现只有一个list[0] 100还是1诶
        ################################

        dt = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, "{:06d}.tar".format(i))
            torch.save(
                {
                    "global_step": global_step,
                    "network_fn_state_dict": render_kwargs_train[
                        "network_fn"
                    ].state_dict(),
                    "network_fine_state_dict": render_kwargs_train[
                        "network_fine"
                    ].state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                path,
            )
            print("Saved checkpoints at", path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode  渲染图片然后合成30帧数的1秒视频
            with torch.no_grad():
                rgbs, disps = render_path(
                    render_poses, hwf, K, args.chunk, render_kwargs_test
                )
            print("Done, saving", rgbs.shape, disps.shape)
            moviebase = os.path.join(
                basedir, expname, "{}_spiral_{:06d}_".format(expname, i)
            )
            imageio.mimwrite(moviebase + "rgb.mp4", to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(
                moviebase + "disp.mp4", to8b(disps / np.max(disps)), fps=30, quality=8
            )

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, "testset_{:06d}".format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print("test poses shape", poses[i_test].shape)
            with torch.no_grad():
                render_path(
                    torch.Tensor(poses[i_test]).to(device),
                    hwf,
                    K,
                    args.chunk,
                    render_kwargs_test,
                    gt_imgs=images[i_test],
                    savedir=testsavedir,
                )
            print("Saved test set")

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    train()
