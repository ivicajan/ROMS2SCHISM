#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from munch import Munch as Bunch
from netCDF4 import Dataset, num2date

def roms_depth_point(zeta, h, vtransform, sc_r, Cs_r, hc):
    N = len(sc_r)
    r = range(N)
    z = np.zeros(np.hstack((N, zeta.shape)))
    if vtransform == 1:
        for k in r:
            z0 = (sc_r[k] - Cs_r[k]) * hc + Cs_r[k] * h
            z[k,:] = z0 + zeta * (1.0 + z0/h)
    elif vtransform == 2:
        for k in r:
            z0 = (hc * sc_r[k] + Cs_r[k] * h) / (hc + h)
            z[k,:] = zeta + (zeta + h) * z0
    return z

def roms_bbox(lon, lat, bbox):
    from matplotlib import path
    #bbox = np.array([xmin, xmax, ymin, ymax])
    mypath = np.array([bbox[[0,1,1,0]], bbox[[2,2,3,3]]]).T
    p = path.Path(mypath)
    points = np.vstack((lon.flatten(), lat.flatten())).T   
    n, m = np.shape(lon)
    inside = p.contains_points(points).reshape((n, m))
    ii, jj = np.meshgrid(list(range(m)), list(range(n)))
    i0, i1, j0, j1 = min(ii[inside])-1, max(ii[inside]), min(jj[inside])-1, max(jj[inside])+3
    ny, nx = np.shape(lon)
    if i0<0 : i0=0
    if i1>nx-1 : i0 = nx
    if j0<0 : j0=0
    if j1>ny-1 : j1 = ny
    return i0, i1, j0, j1     
    
def read_roms_grid(filein, bbox):
    """Reads ROMS grid data from ROMS grid file or output."""
    roms = Bunch()
    nc = Dataset(filein,'r')
    lonr = nc.variables['lon_rho'][:]
    latr = nc.variables['lat_rho'][:]
    roms.i0, roms.i1, roms.j0, roms.j1 = roms_bbox(lonr, latr, bbox)
    print('bbox subset i0=%d, i1=%d, j0=%d, j1=%d' %(roms.i0,roms.i1,roms.j0,roms.j1))
    roms.h = nc.variables['h'][(roms.j0+1):(roms.j1-1), (roms.i0+1):(roms.i1-1)]
    # if east/north velocities not present, need to rotate and process from staggered velocities:
    roms.rotate = not all([var in nc.variables for var in ['u_eastward', 'v_northward']])
    if roms.rotate:
        roms.angle = nc.variables['angle'][(roms.j0+1):(roms.j1-1), (roms.i0+1):(roms.i1-1)]
    roms.lonr = lonr[(roms.j0+1):(roms.j1-1), (roms.i0+1):(roms.i1-1)]
    roms.latr = latr[(roms.j0+1):(roms.j1-1), (roms.i0+1):(roms.i1-1)]
    roms.maskr = nc.variables['mask_rho'][(roms.j0+1):(roms.j1-1), (roms.i0+1):(roms.i1-1)]
    nc.close()
    print('Done with reading roms grid')
    return roms

def read_roms_data(filein, grid, num_times = None):
    roms = Bunch()
    nc = Dataset(filein,'r')
    times = nc.variables['ocean_time']
    nt = np.size(times) if num_times is None else num_times
    roms.date = num2date(times[:nt], units=times.units, calendar='proleptic_gregorian')
    i0, i1, j0, j1 = grid.i0, grid.i1, grid.j0, grid.j1
    #print('loading subset i0=%d, i1=%d, j0=%d, j1=%d' %(i0,i1,j0,j1))
    roms.zeta = nc.variables['zeta'][:nt,(j0+1):(j1-1), (i0+1):(i1-1)]
    #print(np.shape(roms.zeta))
    if grid.rotate:
        # rotate and de-stagger velocities:
        u = nc.variables['u'][:nt,:,(j0+1):(j1-1), i0:(i1-1)]
        v = nc.variables['v'][:nt,:,j0:(j1-1), (i0+1):(i1-1)]
        ur = 0.5*(u[:,:,:,:-1]+u[:,:,:,1:])
        vr = 0.5*(v[:,:,:-1,:]+v[:,:,1:,:])
        roms.u, roms.v = rot2d(ur, vr, grid.angle)
    else:
        roms.u = nc.variables['u_eastward'][:nt,:,(j0+1):(j1-1), (i0+1):(i1-1)]
        roms.v = nc.variables['v_northward'][:nt,:,(j0+1):(j1-1), (i0+1):(i1-1)]
    roms.temp = nc.variables['temp'][:nt,:,(j0+1):(j1-1), (i0+1):(i1-1)]
    roms.salt = nc.variables['salt'][:nt,:,(j0+1):(j1-1), (i0+1):(i1-1)]
    roms.vtransform = nc.variables['Vtransform'][:]
    roms.sc_r = nc.variables['s_rho'][:]
    roms.Cs_r = nc.variables['Cs_r'][:]
    roms.sc_w = nc.variables['s_w'][:]
    roms.Cs_w = nc.variables['Cs_w'][:]
    roms.hc = nc.variables['hc'][:]
    nc.close()
    #print(np.shape(roms.temp))
    print('Done reading roms data file: %s' %filein)
    return roms

def roms_append(old, new):
    ''' 
    appends variables from new dictionary into old along axis 0 (time)
    '''
    out = old
    out.date = np.append(old.date, new.date, axis=0)
    out.zeta = np.append(old.zeta, new.zeta, axis=0)
    out.u = np.append(old.u, new.u, axis=0)
    out.v = np.append(old.v, new.v, axis=0)
    out.temp = np.append(old.temp, new.temp, axis=0)
    out.salt = np.append(old.salt, new.salt, axis=0)
    return out

def read_roms_files(roms_dir, roms_grid, template, dates):
    # part for loading ROMS data for the subset
    for date in dates:
        fname = os.path.join(roms_dir, date.strftime(template))
        try: 
            new = read_roms_data(fname, roms_grid)
            if date == dates[0]:
                roms_data = new
            else:
                roms_data = roms_append(roms_data, new)
        except:
            continue        

    print('Done with reading roms files')
    return roms_data

