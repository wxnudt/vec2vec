#!/usr/bin/env python

import numpy as np
import netCDF4 as nc
import xarray as xr

#
# file_path = '/Users/renxl/Desktop/todo/re4/data-clear/data_tanjm/ecwave.nc'
# file_obj = nc.Dataset(file_path)
#
# # print(file_obj)
# # odict_keys(['longitude', 'latitude', 'time', 'swh', 'mwd', 'mwp'])
#
# print(file_obj.variables.keys())
#
#
# longitudes = file_obj.variables['longitude']
# print(longitudes)
# print('-----------------')
#
# latitudes = file_obj.variables['latitude']
# print(latitudes)
# lat = latitudes[:]
# print(lat)
# print('-----------------')
#
# times = file_obj.variables['time']
# print(times)
# print('-----------------')
# print(times[0])     #1016088
# print(nc.num2date(times[0],'hours since 1900-01-01 00:00:00.0'))    #2015-12-01 00:00:00
# print('-----------------')
#
# times_arr = nc.num2date(times[:], times.units)
# print(times_arr[:12])
# print('-----------------')
#
#
# swh = file_obj.variables['swh']
# print(swh)          # int16 swh(time, latitude, longitude) # (124, 241, 480)
# print('-----------------')
# swh_arr = swh[:]
# print(swh_arr.shape)
# print(swh.scale_factor)
# print(swh.add_offset)
#
# print(swh_arr[:]*swh.scale_factor+swh.add_offset) # the output is --, why?
#
#
# mwd = file_obj.variables['mwd']
# print(mwd)
# mwd_arr = mwd[:]
# print(mwd_arr[0][0][0])
# print('-----------------')
#
# mwp = file_obj.variables['mwp']
# print(mwp)
# print('-----------------')

print('***************************************')
#
# file_path2 = '/Users/renxl/Desktop/todo/re4/data-clear/data_tanjm/MPP20151203000240.nc'
# file_obj2 = nc.Dataset(file_path2)
# # odict_keys(['SWH_GDS0_MSL', 'MWD_GDS0_MSL', 'PP1D_GDS0_MSL', 'MWP_GDS0_MSL',
# # 'SHWW_GDS0_MSL', 'MDWW_GDS0_MSL', 'MPWW_GDS0_MSL', 'SHTS_GDS0_MSL',
# # 'MDTS_GDS0_MSL', 'MPTS_GDS0_MSL', 'DWI_GDS0_HTGL', 'g0_lat_1', 'g0_lon_2', 'forecast_time0'])
#
# print(file_obj2)
# print('-----------------')
# print(file_obj2.variables.keys())
#
# SHWW_GDS0_MSL = file_obj2.variables['SHWW_GDS0_MSL']
# print(SHWW_GDS0_MSL)
# print('-----------------')
# SHWW_GDS0_MSL_arr = SHWW_GDS0_MSL[:]
# print(SHWW_GDS0_MSL_arr[:])
# print(SHWW_GDS0_MSL_arr[0])           # the output is --
# print(SHWW_GDS0_MSL_arr[240][500][999])     # the output is 0.0


print('***************************************')

file_path3 = '/Users/renxl/Desktop/todo/re4/data-clear/taifeng/WP_data_solo/data_seq1/tp_1_sf.nc'
file_obj3 = nc.Dataset(file_path3)

print(file_obj3)
print('-----------------')
print(file_obj3.variables.keys())

sst = file_obj3.variables['sst']
print(sst)
print('-----------------')

sst_arr = sst[:]

print(sst_arr[0])
# print(sst_arr[0]*sst.scale_factor+sst.add_offset)

print('========================================')
ds = xr.open_dataset("/Users/renxl/Desktop/todo/re4/data-clear/taifeng/WP_data_solo/data_seq1/tp_1_sf.nc")
# print(ds)
temp = ds['sst'][0,...] #- 273.15  #把温度取为二维数组(原为三维数组，time,lon,lat)，并转化为℃
# print(temp.values)
temp = temp.values
temp = np.reshape(temp, (-1, 1))
print(temp)
print(temp.shape)


print('========================================')
ds = xr.open_dataset("/Users/renxl/Desktop/todo/re4/data-clear/taifeng/WP_data_solo/data_seq1/tp_1_pl.nc")
print(ds)
z = ds['z']
t = ds['t']
r = ds['r']
u = ds['u']
v = ds['v']

print('-----------------z:')
z = z.values
z = np.reshape(z, (-1, 1))
print(z)
print(z.shape)

print('-----------------t:')
t = t.values
t = np.reshape(t, (-1, 1))
print(t)
print(t.shape)

print('-----------------r:')
r = r.values
r = np.reshape(r, (-1, 1))
print(r)
print(r.shape)

print('-----------------u:')
# print(u.values)
u = u.values
u = np.reshape(u, (-1, 1))
print(u)
print(u.shape)

print('-----------------v:')
v = v.values
v = np.reshape(v, (-1, 1))
print(v)
print(v.shape)


print('-----------------arr:')
arr = np.hstack((z, t, r, u, v))
print(arr)
print(arr.shape)





