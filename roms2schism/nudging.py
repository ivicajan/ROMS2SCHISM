#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from progressbar import progressbar
from netCDF4 import Dataset, date2num
from roms2schism import schism as sm
from roms2schism import roms as rs
from roms2schism import interpolation as itp

def save_nudging_nc(outfile, data, date, sponge_nodes):
    '''
    nComp = 1 for zeta, temp and salt
    nvrt = schism.nvrt for all 3D variables
    date are datetime veariable for time records (not used in SCHISM)
    data is holding data to save
    sponge_nodes is holding node id in the mesh
    '''    
    dst = Dataset(outfile, "w", format="NETCDF4")
    #dimensions
    dst.createDimension('node', data.shape[1])
    dst.createDimension('one', 1)
    dst.createDimension('time', None)
    dst.createDimension('nLevels', data.shape[2])
    dst.createDimension('nComponents', data.shape[3])
    #variables
    dst.createVariable('map_to_global_node', 'i4', ('node',))
    dst['map_to_global_node'][:] = sponge_nodes+1
    dst.createVariable('time', 'f', ('time',))
    dst['time'][:] = date2num(date[:],'seconds since 1900-1-1') - date2num(date[0],'seconds since 1900-1-1')
    dst.createVariable('tracer_concentration', 'f', ('time', 'node', 'nLevels', 'nComponents'))
    dst['tracer_concentration'][:,:,:,:] = data
    dst.close() 
    return

def make_nudging(schism, template, dates, dcrit = 700, roms_dir = './',
                 roms_grid_filename = None, lonc = 175., latc = -37.):
    # ## Part with nudging zone, 
    # ### it needs more points (defined in nudge.gr3) and that file is made using gen_nudge.f90

    sponge = sm.readgr3('nudge.gr3')
    OK = np.where(sponge.z != 0)
    sponge_x = sponge.x[OK]; sponge_y = sponge.y[OK]; sponge_depth = schism.depth[OK]; 
    np.shape(sponge_x), np.shape(sponge_depth)

    # repeat all that we had for boundaries but now for "OK" points
    sponge_bbox = sm.schism_bbox(sponge_x, sponge_y)

    # part to load ROMS grid for given subset
    if roms_grid_filename is not None:
        fname = roms_grid_filename
    else:
        fname = os.path.join(roms_dir, dates[0].strftime(template))
    roms_grid = rs.read_roms_grid(fname, sponge_bbox)
    mask_OK = roms_grid.maskr == 1  # this is the case to avoid interp with masked land values

    roms_data = rs.read_roms_files(roms_dir, roms_grid, template, dates)
      
    interp = itp.spatial_interp(roms_grid, mask_OK, sponge_x, sponge_y, dcrit, lonc, latc)

    # initi outputs nudgining
    nt = len(roms_data.date)  # need to loop over time for each record
    Nz = len(roms_data.Cs_r)  # number of ROMS levels
    Np = np.size(sponge_x)
    schism_zeta = np.zeros((nt, Np,1,1))              # zeta is also needed to compute ROMS depths
    schism_temp = np.zeros((nt, Np, schism.nvrt, 1))  # schims is using (time, node, vert, 1) 
    schism_salt = np.zeros((nt, Np, schism.nvrt, 1))  # schims is using (time, node, vert, 1)

    print('Total steps: %d' %nt, end='>')
    for it in progressbar(range(0, nt)):
        # get first zeta as I need it for depth calculation
        schism_zeta[it,:,0,0] = itp.interp2D(roms_data.zeta[it, mask_OK], interp)
        # compute depths for each ROMS levels at the specific SCHISM locations
        roms_depths_at_schism_node = roms_depth_point(schism_zeta[it,:,0,0], interp.depth_interp,
                                                      roms_data.vtransform, roms_data.sc_r,
                                                      roms_data.Cs_r, roms_data.hc)
        # start with temperature variable for each ROMS layer, need to do that for all 3D variables (temp, salt, u, v)
        temp_interp = np.zeros((Nz, Np))   # this is temp at ROMS levels
        for k in range(0, Nz):   
            temp_interp[k,:] = itp.interp2D(roms_data.temp[it,k,][mask_OK], interp)
        # interpolate in vertical to SCHISM depths
        schism_temp[it,:,:,0] = itp.vert_interp(temp_interp, roms_depths_at_schism_node, -sponge_depth)
        # interp salt variable 
        temp_interp = np.zeros((Nz, Np))
        for k in range(0,Nz):
            temp_interp[k,:] = itp.interp2D(roms_data.salt[it,k,][mask_OK], interp)
        # now you need to interp temp for each NOP at SCHISM depths
        schism_salt[it,:,:,0] = itp.vert_interp(temp_interp, roms_depths_at_schism_node, -sponge_depth)

    os.system('rm -f TEM_nu.nc SAL_nu.nc')
    # now you need to save them in the boundary files
    save_nudging_nc('TEM_nu.nc', schism_temp, roms_data.date, np.array(OK))
    save_nudging_nc('SAL_nu.nc', schism_salt, roms_data.date, np.array(OK))