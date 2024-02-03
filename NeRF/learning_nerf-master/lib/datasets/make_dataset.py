from . import samplers
from .dataset_catalog import DatasetCatalog
import torch
import torch.utils.data
import imp
import os
from .collate_batch import make_collator
import numpy as np
import time
from lib.config.config import cfg
from torch.utils.data import DataLoader, ConcatDataset
##todo light latent catalog没使用

torch.multiprocessing.set_sharing_strategy('file_system')

def _dataset_factory(is_train, is_val):
    if is_val:
        module = cfg.val_dataset_module
        path = cfg.val_dataset_path
    elif is_train:
        module = cfg.train_dataset_module
        path = cfg.train_dataset_path
    else:
        module = cfg.test_dataset_module
        path = cfg.test_dataset_path
    dataset = imp.load_source(module, path).Dataset
    return dataset#私有函数 没有args


def make_dataset(cfg, is_train=True):
    if is_train:
        args = cfg.train_dataset
        module = cfg.train_dataset_module
        path = cfg.train_dataset_path
    else:
        args = cfg.test_dataset#这里传入yaml里面的具体解包参数
        module = cfg.test_dataset_module
        path = cfg.test_dataset_path
    dataset = imp.load_source(module, path).Dataset#这里是一个类与之前Network()不同,前面调用init了
    dataset = dataset(**args)## 这里才初始化了这个dataset(Dataset)
    return dataset


def make_data_sampler(dataset, shuffle, is_distributed):
    if is_distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)#todo 调用len?
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter,
                            is_train):
    if is_train:
        batch_sampler = cfg.train.batch_sampler
        sampler_meta = cfg.train.sampler_meta
    else:
        batch_sampler = cfg.test.batch_sampler
        sampler_meta = cfg.test.sampler_meta

    if batch_sampler == 'default':#走这个了
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last)
    elif batch_sampler == 'image_size':
        batch_sampler = samplers.ImageSizeBatchSampler(sampler, batch_size,
                                                       drop_last, sampler_meta)

    if max_iter != -1:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, max_iter)
    return batch_sampler


def worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))


def make_data_loader(cfg, is_train=True, is_distributed=False, max_iter=-1):
    if is_train:
        batch_size = cfg.train.batch_size
        # shuffle = True
        shuffle = cfg.train.shuffle
        drop_last = False
    else:
        batch_size = cfg.test.batch_size
        shuffle = True if is_distributed else False
        drop_last = False

    dataset = make_dataset(cfg, is_train)
    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(cfg, sampler, batch_size,
                                            drop_last, max_iter, is_train)#max_iter=500 batch size=1
    num_workers = cfg.train.num_workers#4
    collator = make_collator(cfg, is_train)#数据集中获取的单个数据样本组合成一个批次，以便在训练或推理过程中一次处理多个样本
    data_loader = DataLoader(dataset,
                            batch_sampler=batch_sampler,
                            num_workers=num_workers,
                            collate_fn=collator,
                            worker_init_fn=worker_init_fn,#生成的随机数序列是不同的  分布式可能？ 4个进程？
                            pin_memory=True)

    return data_loader
