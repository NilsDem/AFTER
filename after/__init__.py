from pathlib import Path

import gin

from .diffusion import *

BASE_PATH: Path = Path(__file__).parent

gin.add_config_file_search_path(BASE_PATH)
gin.add_config_file_search_path(BASE_PATH.joinpath('diffusion/configs'))
gin.add_config_file_search_path(BASE_PATH.joinpath('autoencoder/configs'))
gin.add_config_file_search_path(BASE_PATH.joinpath('prior/configs'))
