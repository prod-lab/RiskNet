import logging
import os
import pathlib
from os.path import join
from typing import List, Any, Dict

import yaml

logger = logging.getLogger(__name__)


class Config(object):

	def __init__(self) -> None:
		self._config: Dict = self.read_config()

	@staticmethod
	def read_config() -> Dict[str, Any]:

		abs_config_path = os.getenv("RISKNET_CONFIG_PATH")

		path: str = ""

		if abs_config_path:
			path += abs_config_path
		else:
			path += join(pathlib.Path(__file__).parent.resolve(), "conf.yaml")
		with open(path, 'r') as stream:
			try:
				return yaml.full_load(stream)
			except yaml.YAMLError as exc:
				logger.exception(exc)
		return {}


class GeneralConfig(Config):
    @property
    def app_name(self) -> str:
        return self._config["general"]["app_name"]

    @property
    def log_bucket(self) -> str:
        return self._config["general"]["log_bucket"]

    @property
    def dask_cluster_type(self) -> str:
        return self._config["general"]["dask_cluster_type"]

    @property
    def gcp_dask_workers(self) -> str:
        return self._config["general"]["dask_cluster_type"]


class DataConfig(Config):

	@property
	def titanic_root(self) -> str:
		return self._config["data"]["titanic_root"]

	@property 
	def fm_root(self) -> str:
		return self._config["data"]["fm_root"]
	@property
	def data(self) -> str:
		return self._config["data"]["files"]

class ModelConfig(Config):
	@property
	def cv_strategy(self) -> str:
		return self._config["model"]["cv_strategy"]

	@property
	def imputation(self) -> List[str]:
		return self._config["model"]["imputation"]

	@property
	def selection(self) -> str:
		return self._config["model"]["selection"]

	@property
	def exclude_features(self) -> List[str]:
		return self._config["model"]["exclude_features"]
