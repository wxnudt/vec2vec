#!/usr/bin/env python
# -*- coding: utf-8 -*-
import threading
import time

def add(value1, value2=None):
    time.sleep(2)
    print(value1+value2)
    return value1+value2

def print_result(future):
    print(future.result())

if __name__ == "__main__":
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    threadPool = ThreadPoolExecutor(max_workers=4)
    for i in range(0,10):
        future =threadPool.submit(add, i, i+1)


    threadPool.shutdown(wait=True)