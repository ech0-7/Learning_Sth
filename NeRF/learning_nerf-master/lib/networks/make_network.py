import os
import imp


def make_network(cfg):
    module = cfg.network_module
    path = cfg.network_path#todo path是啥 就是module最后.py了
    network = imp.load_source(module, path).Network()# module(内含cfg)
    return network
