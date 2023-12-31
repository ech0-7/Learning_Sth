from .trainer import Trainer
import imp


def _wrapper_factory(cfg, network, train_loader=None):
    module = cfg.loss_module
    path = cfg.loss_path
    network_wrapper = imp.load_source(module, path).NetworkWrapper(network, train_loader)
    return network_wrapper#包装一下网络，加上loss 等对象


def make_trainer(cfg, network, train_loader=None):
    network = _wrapper_factory(cfg, network, train_loader)#loss&loss的forward
    return Trainer(network)##
