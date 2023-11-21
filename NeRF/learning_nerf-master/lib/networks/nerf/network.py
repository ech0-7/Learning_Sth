import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.encoding import get_encoder
from lib.config import cfg

class Network(nn.Module):
    def __init__(self,):#V_D appearance depth是什么
        super(Network, self).__init__()
        net_cfg = cfg.network
        self.skips=[4]
        output_ch=self.output_ch #采样重要度&rgb+sigma #todo有意思 这样就能规避传入了 在参数里改就行
        self.V_D=net_cfg.nerf.V_D #appearance depth
        self.xyz_encoder, input_ch = get_encoder(net_cfg.xyz_encoder)#freq  
        self.dir_encoder, input_ch_views = get_encoder(net_cfg.dir_encoder)#freq
        D, W  = net_cfg.nerf.D, net_cfg.nerf.W
        self.use_viewdirs = cfg.use_viewdirs#todo 有事没事全存self里面
        self.pts_linears = nn.ModuleList(##8个线性层的网络
            [nn.Linear(input_ch, W)]
            + [
                nn.Linear(W, W)
                if i not in self.skips
                else nn.Linear(W + input_ch, W) 
                for i in range(D - 1)
            ]
        )
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
        if cfg.use_viewdirs:  # 有无相机
            self.feature_linear = nn.Linear(W, W)#W=256 橘色直连
            self.alpha_linear = nn.Linear(W, 1)#密度
            self.rgb_linear = nn.Linear(W // 2, 3)#RGB
        else:
            self.output_linear = nn.Linear(W, output_ch)
    #todo 看不懂这个output是5 且之后会采样的部分
    def test_render(self,x,batch):#todo 感觉batch就是前面需要 这里意思一下罢了
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1
        )#这样传入 把俩个区分出来#todo 原来是在外面encoding的限制这个encoder在模型里面 可以写个函数封装在这
        h=input_pts#todo x 不能这样传
        for i, l in enumerate(self.pts_linears): #i:0 l:Linear(in_features=63, out_features=256, bias=True)
            h = self.pts_linears[i](h)
            h = F.relu(h)#relu过线性层
            if i in self.skips:
                h = torch.cat([x, h], -1)#后面的都是256->256 除了skip后是63+256->256 i=4
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
        
    def render(self, uv, batch):
        uv_encoding = self.uv_encoder(uv)
        x = uv_encoding
        for i, l in enumerate(self.backbone_layer):
            x = self.backbone_layer[i](x)
            x = F.relu(x)
        rgb = self.output_layer(x)
        return {'rgb': rgb}

    def batchify(self, uv, batch):
        all_ret = {}
        chunk = cfg.task_arg.chunk_size
        for i in range(0, uv.shape[0], chunk):
            ret = self.render(uv[i:i + chunk], batch)#8192,3
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=0) for k in all_ret}#字典推导 每个chunk合体cat
        return all_ret

    def forward(self, batch):
        B, N_pixels, C = batch['uv'].shape#1 8192 2
        ret = self.batchify(batch['uv'].reshape(-1, C), batch)
        return {k:ret[k].reshape(B, N_pixels, -1) for k in ret}#(1,8192,3)
