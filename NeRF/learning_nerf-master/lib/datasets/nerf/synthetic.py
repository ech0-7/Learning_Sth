import torch
import torch.utils.data as data
import numpy as np
import os
from lib.utils import data_utils
from lib.config import cfg
from torchvision import transforms as T
import imageio
import json
import cv2
#todo 方便理解 不写在utils里了
trans_t = lambda t: torch.Tensor(
    [[1, 0, 0, 0], 
     [0, 1, 0, 0], 
     [0, 0, 1, t], 
     [0, 0, 0, 1]]
).float()

rot_phi = lambda phi: torch.Tensor(
    [
        [1, 0, 0, 0], 
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
).float()

rot_theta = lambda th: torch.Tensor(#Y不动
    [
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ]
).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.Tensor(
            np.array([[-1, 0, 0, 0], 
                      [0, 0, 1, 0], 
                      [0, 1, 0, 0], 
                      [0, 0, 0, 1]])
        )
        @ c2w
    )
    return c2w

def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )  # (400,400)[0,1,2,3,...,400]~[0,0,0,0,...,0] 400行列
    dirs = np.stack(
        [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1
    )  
    # Rotate ray directions from camera frame  to the world frame
    rays_d = np.sum(  
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1  
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))  # (400,400,3)
    return rays_o, rays_d


class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene#synthetic train lego
        #view = eval(kwargs['cams'])
        self.input_ratio = kwargs['input_ratio']#1
        self.data_root = os.path.join(data_root, scene)#logo
        self.split = split#train
        skips = kwargs['cams']  
        #my edition
        self.white_bkgd=cfg.task_arg.white_bkgd
        self.use_viewdirs=cfg.use_viewdirs
        self.N_rays=cfg.task_arg.N_rays
        # read image
        imgs = []
        poses = []
        image_paths = []
        json_info = json.load(open(os.path.join(self.data_root, 'transforms_{}.json'.format(self.split))))
        for frame in json_info['frames']:#dict
            image_paths.append(os.path.join(self.data_root, frame['file_path'][2:] + '.png'))
            poses.append(np.array(frame['transform_matrix']))
        # skip&set
        for path in image_paths[skips[0]:None:skips[2]]:
            img= (imageio.imread(path)/255.0).astype(np.float32)
            if self.white_bkgd:
                img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])    
            if self.input_ratio != 1.:
                img = cv2.resize(img, None, fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_AREA)
            imgs.append(img)
        imgs=np.array(imgs)
        self.counts=imgs.shape[0]
        self.poses = np.array(poses).astype(np.float32)#100,4,4 rt rd
        self.imgs = imgs.astype(np.float32)
        self.H, self.W = imgs[0].shape[:2]
        camera_angle_x=float(json_info['camera_angle_x'])
        self.focal = 0.5 * self.W / np.tan(0.5 * camera_angle_x)
        self.render_poses = torch.stack(
        [
            pose_spherical(angle, -30.0, 4.0)#半径4 俯仰角-30度
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )
        self.K = np.array([[self.focal, 0, 0.5 * self.W], [0, self.focal, 0.5 * self.H], [0, 0, 1]])
        #use_allbatch
        print('get rays')
        rays = np.stack(
            [get_rays_np(self.H, self.W, self.K, p) for p in self.poses[:, :3, :4]], 0
        )#(100,2,800,800,3)
        rays_rgb = np.concatenate([rays, self.imgs[:, None]], 1)  # [N, ro+rd+new的rgb维度, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+new的rgb维度, 3]
        #print(rays_rgb.shape)
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        print("shuffled rays")
        #np.random.shuffle(rays_rgb)  
        print("done")
        self.size=rays_rgb.shape[0]
        self.rays_rgb=rays_rgb

    #todo 把using Batching的那些东西都拿过来 先K的内参矩阵得到rays_d rays_o直接是trans最后一列得到1024的rays
    #todo 生成一个维度用于 xyz与rgb的 stack
    #todo using Batching的方法可以通过 iterableDataset或者BatchSampler一下1024个实现 这里直接自己写了

    def __getitem__(self, index):
        if self.split == 'train':
            ids = np.random.choice(len(self.rays_rgb), self.N_rays, replace=False)
            batch = self.rays_rgb[ids]  
            batch = torch.Tensor(batch)
            batch = torch.transpose(
                batch, 0, 1
            )  
            batch_rays, target_s = batch[:2], batch[2]  
        else:
            batch = self.rays_rgb
            batch = torch.Tensor(batch)
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]
        ret={'rays':batch_rays,'rgb':target_s}
        ret.update({'meta': {'H': self.H, 'W': self.W}}) 
        return ret

    def __len__(self):#数据集对象的 __len__ 方法来确定每个 epoch 的迭代次数。这是因为 DataLoader 需要知道在每个 epoch 中有多少批次（batch）的数据
        # we only fit 1 images, so we return 1
        return 100