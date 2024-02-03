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


class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene#synthetic train lego
        #view = eval(kwargs['cams'])
        self.input_ratio = kwargs['input_ratio']#1
        self.data_root = os.path.join(data_root, scene)#logo
        self.split = split#train
        #todo my edition
        self.white_bkgd=kwargs['white_bkgd']
        skips = kwargs['cams']
        # read image
        imgs = []
        poses = []
        image_paths = []
        json_info = json.load(open(os.path.join(self.data_root, 'transforms_{}.json'.format(self.split))))
        for frame in json_info['frames']:#dict
            image_paths.append(os.path.join(self.data_root, frame['file_path'][2:] + '.png'))
            poses.append(np.array(frame['transform_matrix']))
        # skip&set
        for path in image_paths[skips[0]:skips[1]:skips[2]]:
            img= imageio.imread(path)/255.0.astype(np.float32)
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
    #todo 把using Batching的那些东西都拿过来 先K的内参矩阵得到rays_d rays_o直接是trans最后一列得到1024的rays
    #todo 想要多一点的话可以 L2正则化的作为另外的输入
    #todo 期间一直保留rgb在最后一个位置上(,3)的形式 PE是网络的部分了就
    def __getitem__(self, index):
        if self.split == 'train':
            ids = np.random.choice(len(self.uv), self.batch_size, replace=False)
            uv = self.uv[ids]
            rgb = self.img.reshape(-1, 3)[ids]
        else:
            uv = self.uv
            rgb = self.img.reshape(-1, 3)
        ret = {'uv': uv, 'rgb': rgb} # input and output. they will be sent to cuda
        ret.update({'meta': {'H': self.img.shape[0], 'W': self.img.shape[1]}}) # meta means no need to send to cuda
        return ret

    def __len__(self):#数据集对象的 __len__ 方法来确定每个 epoch 的迭代次数。这是因为 DataLoader 需要知道在每个 epoch 中有多少批次（batch）的数据
        # we only fit 1 images, so we return 1
        return 1
