from lib.config import cfg, args
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network, save_trained_config, load_pretrain
from lib.evaluators import make_evaluator
import torch.multiprocessing
import torch
import torch.distributed as dist
import os
torch.autograd.set_detect_anomaly(True)

if cfg.fix_random:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(cfg, network):
    train_loader = make_data_loader(cfg,
                                    is_train=True,#处于才
                                    is_distributed=cfg.distributed,
                                    max_iter=cfg.ep_iter)#最大迭代次数500
    val_loader = make_data_loader(cfg, is_train=False)
    trainer = make_trainer(cfg, network, train_loader)#todo 初始化的时候 train_loader好像没咋调用

    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)

    begin_epoch = load_model(network,
                            optimizer,
                            scheduler,
                            recorder,
                            cfg.trained_model_dir,
                            resume=cfg.resume)#保存model trained model不存在return 0了
    if begin_epoch == 0 and cfg.pretrain != '':#没训练但又pretrain的
        load_pretrain(network, cfg.pretrain)

    set_lr_scheduler(cfg, scheduler)

    for epoch in range(begin_epoch, cfg.train.epoch):#20个epoch img
        recorder.epoch = epoch
        if cfg.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        train_loader.dataset.epoch = epoch#todo 这个调用设置有点神奇 没有这个参数呀 可能是基类相关吧

        trainer.train(epoch, train_loader, optimizer, recorder)#trainer利用网络初始化 调用函数train是后面这些参数 train_loader后续在这里enum好像前面都没有init作为一个参数
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0 and cfg.local_rank == 0:
            save_model(network, optimizer, scheduler, recorder,
                    cfg.trained_model_dir, epoch)

        if (epoch + 1) % cfg.save_latest_ep == 0 and cfg.local_rank == 0:
            save_model(network,
                    optimizer,
                    scheduler,
                    recorder,
                    cfg.trained_model_dir,
                    epoch,
                    last=True)

        if (epoch + 1) % cfg.eval_ep == 0 and cfg.local_rank == 0:
            trainer.val(epoch, val_loader, evaluator, recorder)

    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    epoch = load_network(network,
                        cfg.trained_model_dir,
                        resume=cfg.resume,
                        epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator)

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()#分布式进程数
    if world_size == 1:
        return
    dist.barrier()#阻塞进程，所有进程一起 直到所有进程都调用此函数

def main():
    if cfg.distributed:
        cfg.local_rank = int(os.environ['RANK']) % torch.cuda.device_count()
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl",init_method="env://")
        synchronize()

    network = make_network(cfg)#Network初始化后的4个参数的网络类
    if args.test:
        test(cfg, network)
    else:
        trained_network=train(cfg, network)#为什么return却不接受
    if cfg.local_rank == 0:#0是主进程 只要执行一次的比如 打印日记保存模型
        print('Success!')
        print('='*80)#80个等号输出
    os.system('kill -9 {}'.format(os.getpid()))#杀死进程释放


if __name__ == "__main__":
    main()
