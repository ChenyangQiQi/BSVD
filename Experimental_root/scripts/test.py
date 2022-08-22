import os.path as osp


import Experimental_root.archs
import Experimental_root.models
import Experimental_root.data
import Experimental_root.scripts


from basicsr import test_pipeline

def test(runtime_root=None):
    if runtime_root is None:
        runtime_root = osp.abspath(osp.join(__file__, osp.pardir, '..', '..'))
    print(f'current root path: {runtime_root}')
    test_pipeline(runtime_root)
