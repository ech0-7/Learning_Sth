from collections import deque, defaultdict
import torch
from tensorboardX import SummaryWriter
import os
from lib.config.config import cfg

from termcolor import colored


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


def process_volsdf(image_stats):
    for k, v in image_stats.items():
        image_stats[k] = torch.clamp(v[0].permute(2, 0, 1), min=0., max=1.)
    return image_stats

process_neus = process_volsdf

class Recorder(object):
    def __init__(self, cfg):
        if cfg.local_rank > 0:
            return

        log_dir = cfg.record_dir
        if not cfg.resume:#不保存的话就删除 并告诉删除了哪一个
            print(colored('remove contents of directory %s' % log_dir, 'red'))
            os.system('rm -r %s' % log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

        # scalars
        self.epoch = 0
        self.step = 0
        self.loss_stats = defaultdict(SmoothedValue)#当你尝试访问字典中不存在的键时，它会自动创建一个新的条目，其值是 SmoothedValue 类的一个新实例跟踪不同类型的损失值，每种类型的损失值都有一个 SmoothedValue 实例来记录其平滑处理后的值。
        self.batch_time = SmoothedValue()
        self.data_time = SmoothedValue()#平滑处理批处理时间

        # images
        self.image_stats = defaultdict(object)
        if 'process_' + cfg.task in globals():#nerf 全局函数 20好像只有这个文件里面包括的recoder
            self.processor = globals()['process_' + cfg.task]
        else:
            self.processor = None#处理特定任务的数据的处理器，并创建一个用于跟踪图像统计信息的字典。具体的行为取决于 cfg.task 的值，以及存在哪些全局函数。

    def update_loss_stats(self, loss_dict):
        if cfg.local_rank > 0:
            return
        for k, v in loss_dict.items():
            self.loss_stats[k].update(v.detach().cpu())

    def update_image_stats(self, image_stats):
        if cfg.local_rank > 0:
            return
        if self.processor is None:
            return
        image_stats = self.processor(image_stats)
        for k, v in image_stats.items():
            self.image_stats[k] = v.detach().cpu()

    def record(self, prefix, step=-1, loss_stats=None, image_stats=None):
        if cfg.local_rank > 0:
            return

        pattern = prefix + '/{}'
        step = step if step >= 0 else self.step
        loss_stats = loss_stats if loss_stats else self.loss_stats

        for k, v in loss_stats.items():
            if isinstance(v, SmoothedValue):
                self.writer.add_scalar(pattern.format(k), v.median, step)
            else:
                self.writer.add_scalar(pattern.format(k), v, step)

        if self.processor is None:
            return
        image_stats = self.processor(image_stats) if image_stats else self.image_stats
        for k, v in image_stats.items():
            self.writer.add_image(pattern.format(k), v, step)

    def state_dict(self):
        if cfg.local_rank > 0:
            return
        scalar_dict = {}
        scalar_dict['step'] = self.step
        return scalar_dict

    def load_state_dict(self, scalar_dict):
        if cfg.local_rank > 0:
            return
        self.step = scalar_dict['step']

    def __str__(self):
        if cfg.local_rank > 0:
            return
        loss_state = []
        for k, v in self.loss_stats.items():
            loss_state.append('{}: {:.4f}'.format(k, v.avg))
        loss_state = '  '.join(loss_state)

        recording_state = '  '.join(['epoch: {}', 'step: {}', '{}', 'data: {:.4f}', 'batch: {:.4f}'])
        return recording_state.format(self.epoch, self.step, loss_state, self.data_time.avg, self.batch_time.avg)


def make_recorder(cfg):
    return Recorder(cfg)
