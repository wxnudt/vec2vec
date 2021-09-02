#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

if comm_rank == 0:
    a = np.array([[1, 2], [3, 4], [5, 6]])
    b = np.array([[1, 2, 3], [4, 5, 6]])
    b = b.transpose()
    print(a)
    print(b)

else:
    a = None
    b = None

# color_row = comm_rank / 3
color_row = comm_rank // 3     # python3 中除法必须是a//b，而不是 a/b, a/b获得的是未取整的小数，导致进程号不是整数，通信组中只有0号进程
color_col = comm_rank % 3
print(comm_rank, "color_row:", color_row, "color_col:", color_col)


comm_row = comm.Split(color_row)
comm_col = comm.Split(color_col)


row = comm_col.scatter(a, root=0) if color_col == 0 else None
row = comm_row.bcast(row, root=0)

col = comm_row.scatter(b, root=0) if color_row == 0 else None
col = comm_col.bcast(col, root=0)


print(comm_rank, "row:", row, "col:", col)

ret = sum([x * y for x, y in zip(row, col)])

c = comm.gather(ret, root=0)

if comm_rank == 0:
    c = np.array(c).reshape(3, 3)
    print(c)

