import os
import sys

#Note: for some reason risknet.proc.[package_name] didn't work so I'm updating this yall :D
risknet_proc_path = risknet_run_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'run')
sys.path.append(risknet_proc_path) #reorient directory to access proc .py files
from risknet.run import pipeline

pipeline.main()