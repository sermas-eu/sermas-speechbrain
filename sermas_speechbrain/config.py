import logging
import pathlib
import dotenv
import os

logging.basicConfig(level=logging.INFO)


def load_env():

    logger = logging.getLogger(__name__)

    _root_dir = pathlib.Path(__file__).parents[1]
    _config_dir = os.path.join(_root_dir, "config")

    root_dir_env = os.path.join(_root_dir, ".env")
    logger.info(f"Loading {root_dir_env} ")
    dotenv.load_dotenv(root_dir_env)

    # Load from ./config path
    if os.path.isdir(_config_dir):
        config_dir_env = os.path.join(_config_dir, ".env")
        logger.info(f"Loading {config_dir_env} ")
        dotenv.load_dotenv(config_dir_env)

    if "LOG_LEVEL" in os.environ.keys():
        log_level = logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO"))
        logging.basicConfig(level=log_level)
