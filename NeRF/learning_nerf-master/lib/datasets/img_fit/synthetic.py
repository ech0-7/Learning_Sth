import torch.utils.data as data
import numpy as np
import os
from lib.utils import data_utils
from lib.config import cfg
from torchvision import transforms as T
import imageio
import json
import cv2


class Dataset(data.Dataset):
    def __init__(self, **kwargs):#4个参数 前4个取出来的
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene#synthetic train lego
        view = kwargs['view']#0
        self.input_ratio = kwargs['input_ratio']#1
        self.data_root = os.path.join(data_root, scene)#logo
        self.split = split#train
        self.batch_size = cfg.task_arg.N_pixels#8192

        # read image
        image_paths = []
        json_info = json.load(open(os.path.join(self.data_root, 'transforms_{}.json'.format('train'))))
        for frame in json_info['frames']:#dict
            image_paths.append(os.path.join(self.data_root, frame['file_path'][2:] + '.png'))#截取字符串 转换下路径

        img = imageio.imread(image_paths[view])/255.#读取归一化 (800,800,4)
        img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])#1是不透明 0是透明 1-0得到的是白色255/255 最后是3维度了 怎么算都没有大于1的还是归一的 小于1*后-同一个值肯定还是小于0最后+1
        if self.input_ratio != 1.:
            img = cv2.resize(img, None, fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_AREA)
        # set image
        self.img = np.array(img).astype(np.float32)
        # set uv
        H, W = img.shape[:2]
        X, Y = np.meshgrid(np.arange(W), np.arange(H))#(800,800)
        u, v = X.astype(np.float32) / (W-1), Y.astype(np.float32) / (H-1)#0~799归一
        self.uv = np.stack([u, v], -1).reshape(-1, 2).astype(np.float32)#(800,800,2)->(640000,2) 0~1的一列 和相同值的一列 
        #UV 映射是一种将 2D 图像坐标（像素位置）映射到 3D 模型表面的方法。在这里，它首先创建一个网格，然后将网格的坐标归一化，并将结果存储在 self.uv 中。最后，在 __getitem__ 方法中，它从 UV 映射中随机选择一些点
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
        return 1#todo dataloader的时候为什么调用这里的len呢 说的位置是torch.utils.data.sampler.BatchSampler
