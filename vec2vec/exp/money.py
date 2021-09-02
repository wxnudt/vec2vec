#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

rate=0.3
earnEveryYear=0
numberOfYears=20
money=120.0

for year in range(numberOfYears):
    money=money*(1+rate)+earnEveryYear
    print("第 "+ str(year+1)+" 年："+ str(money))



print(int(math.log(756,2)))