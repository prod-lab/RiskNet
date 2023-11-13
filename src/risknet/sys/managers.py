from risknet.sys.log import logger
from risknet.config import cfg
from dask.distributed import LocalCluster, Client
from dask_kubernetes import KubeCluster
from dask_yarn import YarnCluster
from dask_cloudprovider.gcp import GCPCluster
from multiprocessing import cpu_count

class DaskManager:
    
    def __init__(self, cluster_type: str = cfg.dask_cluster_type):
        if cluster_type == "local":
            self.cluster = LocalCluster(n_workers=cpu_count(), 
                                        threads_per_worker=1, 
                                        processes=True)
            self.client = Client(self.cluster)
        elif cluster_type == "k8s":
            self.cluster = KubeCluster('manifests/worker-spec.yml')
        elif cluster_type == "gcp":
            self.cluster = GCPCluster(n_workers=cfg.gcp_dask_workers)
        else: 
            raise ValueError("""Please supply a valid cluster type, i.e. local,
                             k8s, or gcp""")
        self.client = Client(self.cluster)

    def __enter__(self):
        return self.client

    def __exit__(self, type, value, traceback):
        self.cluster.close()
        self.client.close()



