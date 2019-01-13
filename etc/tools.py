import numpy as np
import time

def flatten(l):
    return np.array(l).flatten()


start_time = None

def timer_start():
    global start_time
    start_time = time.time()

def timer_stop():
    print("Run time: %s seconds" % (time.time() - start_time))


