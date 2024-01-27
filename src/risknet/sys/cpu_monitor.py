#Imports
import psutil #Note: psutil should be automatically installed with python >= 3.9 via site-packages

'''
Determine and print CPU/resource baseline for the pipeline.
This SHOULD be the same for every DSMLP instance assuming the same setup, 
but we'll print it just in case.
'''
def baseline():
    #Return the number of logical CPUs (physical cores * # of threads that can run on each core)
    logic = psutil.cpu_count()

    #Determine number of physical cores:
    phys = psutil.cpu_count(logical=False)

    #Determine total memory available
    mem = psutil.virtual_memory()
    total = mem[0]

    return {"num_logical_cpus:": logic, "num_physical_cores": phys, "total_avail_memory": total}

'''
Determine the disk status (how much space is left on disk) at a given time.

Input: fm_root as defined in pipeline.py
'''
def disk_status(fm_root):
    #How much % of the disk is being used?
    disks = psutil.disk_usage(fm_root)
    #This returns a named tuple of total, used, free, and % of disk

    #How much memory is being used?
    mem = psutil.virtual_memory() #returns a named tuple
    used_mem = mem[0:3] 
    #Returns named tuple of total, available, and percent of memory used

'''
Determine how much CPU and time has been used since it was last called.
Use this before/after methods, training, etc.

Note: the first time you call this, it will return 0
'''
def cpu_check():
    elapsed = psutil.cpu_percent(interval=None)
    return elapsed