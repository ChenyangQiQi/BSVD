import os.path as osp
from Experimental_root.scripts.train_pipeline import train_pipeline


import Experimental_root.archs
import Experimental_root.models
import Experimental_root.data
import Experimental_root.scripts


def train(runtime_root=None, cmd=None):
    if runtime_root is None:
        runtime_root = osp.abspath(osp.join(__file__, osp.pardir, '..', '..'))
    print(f'current root path: {runtime_root}')
    train_pipeline(runtime_root, cmd=cmd)
