from risknet.sys.log import logger
from pathlib import Path


def create_dir(path: str) -> None:
    logger.info(f"Creating path at {path}")
    Path(path).mkdir(parents=True, exist_ok=True)


def delete_dir(path: str) -> None:
    Path(path).rmdir()


def delete_file(file: str) -> None:
    Path(file).unlink(missing_ok=True)


