from src.config import Config
from src import models, trainer, meta, dataset, handcrafted, estimator, transforms, utils

config = Config().get_yaml()
utils.set_random_seeds(config.get("seed"))