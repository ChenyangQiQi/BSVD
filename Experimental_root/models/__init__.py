import importlib
from os import path as osp
from basicsr.utils import scandir

model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]

# import all the model modules
_model_modules = [importlib.import_module(f'Experimental_root.models.{file_name}') for file_name in model_filenames]