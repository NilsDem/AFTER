from .diffusion import *

from pathlib import Path

BASE_PATH: Path = Path(__file__).parent

gin.add_config_file_search_path(BASE_PATH)
gin.add_config_file_search_path(BASE_PATH.joinpath('diffusion/configs'))
gin.add_config_file_search_path(BASE_PATH.joinpath('autoencoder/configs'))
gin.add_config_file_search_path(BASE_PATH.joinpath('encoder_ssl/configs'))
