from pathlib import Path

import cached_conv as cc
import gin
import torch
import acids_dataset
from acids_dataset.utils import GinEnv


BASE_PATH: Path = Path(__file__).parent
CUSTOM_PATH: Path = Path(__file__).parent.parent / "configs"

gin.add_config_file_search_path(BASE_PATH)
gin.add_config_file_search_path(CUSTOM_PATH)
gin.add_config_file_search_path(BASE_PATH.joinpath('configs'))
if not BASE_PATH.joinpath('configs', 'augmentations').exists():
    BASE_PATH.joinpath('configs', 'augmentations').symlink_to(Path(acids_dataset.__file__).parent / "configs" / "transforms")

gin.add_config_file_search_path(BASE_PATH.joinpath('configs', 'augmentations'))



def __safe_configurable(name):
    try: 
        setattr(cc, name, gin.get_configurable(f"cc.{name}"))
    except ValueError:
        setattr(cc, name, gin.external_configurable(getattr(cc, name), module="cc"))

# cc.get_padding = gin.external_configurable(cc.get_padding, module="cc")
# cc.Conv1d = gin.external_configurable(cc.Conv1d, module="cc")
# cc.ConvTranspose1d = gin.external_configurable(cc.ConvTranspose1d, module="cc")

__safe_configurable("get_padding")
__safe_configurable("Conv1d")
__safe_configurable("ConvTranspose1d")

from .blocks import *
from .discriminator import *
from .model import RAVE, BetaWarmupCallback
from .pqmf import *
from .balancer import *
from . import core


def get_run_name(run_path): 
    checkpoint_path = (Path(run_path) / "..").resolve()
    if checkpoint_path.stem != "checkpoints": 
        return None
    version_path = (checkpoint_path / "..").resolve()
    if checkpoint_path.stem.startswith("version_"):
        return None
    return ((version_path / "..").resolve()).stem

def load_rave_checkpoint(model_path, n_channels=1, ema=False, name="last.ckpt"):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))
    if model_path.suffix == ".ts":
        return torch.jit.load(model_path)
    with GinEnv():
        config_file = core.search_for_config(model_path) 
        if config_file is None:
            print('no configuration file found at address :'%model_path)
        gin.parse_config_file(config_file)
        run_path = core.search_for_run(model_path, name=name)
        rave_model = RAVE()
        checkpoint = torch.load(run_path, map_location='cpu')
        if ema and "EMA" in checkpoint["callbacks"]:
            rave_model.load_state_dict(
                checkpoint["callbacks"]["EMA"],
                strict=False,
            )
        else:
            rave_model.load_state_dict(
                checkpoint["state_dict"],
                strict=False,
            )
    return rave_model, run_path
