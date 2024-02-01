import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


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


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ["train", "val", "test"]
    metas = {}#读出来角度和100图片的 角度和4参数矩阵 rotation好像怎么没用到
    for s in splits:
        with open(os.path.join(basedir, "transforms_{}.json".format(s)), "r") as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == "train" or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta["frames"][::skip]:
            fname = os.path.join(basedir, frame["file_path"] + ".png")
            imgs.append(imageio.imread(fname))  # RGBA 4channel 800*800 3channels+alpha
            poses.append(np.array(frame["transform_matrix"]))  ## 4*4 列表转np append
        imgs = (np.array(imgs) / 255.0).astype(
            np.float32
        )  # keep all 4 channels (RGBA) 归一  (100stack的,800,800,4)
        poses = np.array(poses).astype(np.float32)  #(100,4,4)
        counts.append(counts[-1] + imgs.shape[0])  # 0,100,113,138
        all_imgs.append(imgs)
        all_poses.append(poses)  # train(100) val(13,800,800,4) test (25)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)#100+13+25=138列表行接 (138,800,800,4)
    poses = np.concatenate(all_poses, 0)  # (138,4,4)  0维度拼接

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])#todo 是哪一个呀
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    render_poses = torch.stack(
        [
            pose_spherical(angle, -30.0, 4.0)#半径4 俯仰角-30度
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )  # (40,4,4) 0维度拼接

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.0

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split
