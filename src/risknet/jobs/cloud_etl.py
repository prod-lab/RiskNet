from risknet.config import data_cfg
from risknet.sys.managers import DaskManager
import dask.dataframe as dd
from risknet.sys.log import logger

def execute(bucket_root: str = data_cfg.titanic_root) -> None:
    """
    This job is critical to indicate whether you have sufficient permissions to
     read/write from/to object storage.
    :param bucket_root: path to titanic.csv - you should stage this in advance
    :return: No python return value, writes to object storage
    """
    
    with DaskManager() as d:
        logger.info("Read titanic csv to dask dataframe")
        df = dd.read_csv(f"{bucket_root}titanic.csv")

        logger.info("Write summary file as parquet")
        df.describe().to_parquet(f"{bucket_root}dask_pq.parquet")

        logger.info("Cloud ETL job done")


if __name__ == "__main__":
    execute()
