#!/usr/bin/env python
# coding: utf-8

"""Copyright 2023 University of Western Australia.

This file is part of ROMS2SCHISM.

ROMS2SCHISM is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

ROMS2SCHISM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with ROMS2SCHISM.  If not, see <http://www.gnu.org/licenses/>."""

import os
import numpy as np
from netCDF4 import Dataset, num2date, date2num
from progressbar import progressbar
from roms2schism import roms as rs
from roms2schism import interpolation as itp

def save_boundary_nc(outfile, data, date, schism):
    '''
    nComp = 1 for zeta, temp and salt
    nComp = 2 for uv
    nvrt = schism.nvrt for all 3D variables
    nvrt = 1 for zeta
    date are datetime veariable for time records (not used in SCHISM)
    data is holding data to save
    
    '''  
    dst = Dataset(outfile, "w", format="NETCDF4")
    #dimensions
    dst.createDimension('nOpenBndNodes', data.shape[1])
    dst.createDimension('one', 1)
    dst.createDimension('time', None)
    dst.createDimension('nLevels', data.shape[2])
    dst.createDimension('nComponents', data.shape[3])
    #variables
    dst.createVariable('time_step', 'f', ('one',))
    dst['time_step'][:] = (date[2]-date[1]).total_seconds()
    # time should start with 0. and increase with step (in secs)
    dst.createVariable('time', 'f', ('time',))
    dst['time'][:] = date2num(date[:],'seconds since 1900-1-1') - date2num(date[0],'seconds since 1900-1-1')
    dst.createVariable('time_series', 'f', ('time', 'nOpenBndNodes', 'nLevels', 'nComponents'))
    dst['time_series'][:,:,:,:] = data
    dst.close()     
    return

def make_boundary(schism, template, dates, start = None, end = None, roms_dir = './',
                  roms_grid_filename = None, roms_grid_dir = None,
                  dcrit = 700):
    # ## Part for boundary conditions ROMS -> SCHISM

    # part to load ROMS grid for given subset
    if roms_grid_filename is not None:
        fname = roms_grid_filename
    else:
        roms_grid_dir = roms_dir
        fname = dates[0].strftime(template)
    roms_grid = rs.roms_grid(fname, roms_grid_dir, schism.b_bbox, schism.lonc, schism.latc)

    mask_OK = roms_grid.maskr == 1  # this is the case to avoid interp with masked land values

    roms_data = rs.roms_data(roms_grid, roms_dir, template, dates, start, end)
    
    interp = itp.interpolator(roms_grid, mask_OK, schism.b_xi, schism.b_yi, dcrit)

    # init outputs 
    nt = len(roms_data.date)  # need to loop over time for each record
    Nz = len(roms_data.Cs_r)  # number of ROMS levels
    schism_depth = schism.b_depth                             # schism depths at the open bounday nodes [NOP, nvrt]
    schism_zeta = np.zeros((nt, schism.NOP,1,1))              # zeta is also needed to compute ROMS depths
    schism_temp = np.zeros((nt, schism.NOP, schism.nvrt, 1))  # schism is using (time, node, vert, 1)
    schism_salt = np.zeros((nt, schism.NOP, schism.nvrt, 1))  # schism is using (time, node, vert, 1)
    schism_uv = np.zeros((nt, schism.NOP, schism.nvrt, 2))    # schism is using (time, node, vert, 2)

    print('Interpolating...')
    for it in progressbar(range(0, nt)):
        # get first zeta as I need it for depth calculation
        schism_zeta[it,:,0,0] = interp.interpolate(roms_data.zeta[it, mask_OK])
        # compute depths for each ROMS levels at the specific SCHISM locations
        roms_depths_at_schism_node = roms_data.depth_point(schism_zeta[it,:,0,0], interp.depth_interp)
        # start with temperature variable for each ROMS layer, need to do that for all 3D variables (temp, salt, u, v)
        temp_interp = np.zeros((Nz, schism.NOP))   # this is temp at ROMS levels
        for k in range(0, Nz):   
            temp_interp[k,:] = interp.interpolate(roms_data.temp[it,k,][mask_OK])
        # interpolate in vertical to SCHISM depths
        schism_temp[it,:,:,0] = itp.vert_interp(temp_interp, roms_depths_at_schism_node, -schism_depth)

        # interp salt variable 
        temp_interp = np.zeros((Nz, schism.NOP))
        for k in range(0,Nz):
            temp_interp[k,:] = interp.interpolate(roms_data.salt[it,k,][mask_OK])
        # now you need to interp temp for each NOP at SCHISM depths
        schism_salt[it,:,:,0] = itp.vert_interp(temp_interp, roms_depths_at_schism_node, -schism_depth)

        # interp u variable 
        temp_interp = np.zeros((Nz, schism.NOP))
        for k in range(0,Nz):
            temp_interp[k,:] = interp.interpolate(roms_data.u[it,k,][mask_OK])
        # now you need to interp temp for each NOP at SCHISM depths
        schism_uv[it,:,:,0] = itp.vert_interp(temp_interp, roms_depths_at_schism_node, -schism_depth)

        # interp v variable 
        temp_interp = np.zeros((Nz, schism.NOP))
        for k in range(0,Nz):
            temp_interp[k,:] = interp.interpolate(roms_data.v[it,k,][mask_OK])
        # now you need to interp temp for each NOP at SCHISM depths
        schism_uv[it,:,:,1] = itp.vert_interp(temp_interp, roms_depths_at_schism_node, -schism_depth)
    # now you need to save them in the boundary files
    os.system('rm  -f elev2D.th.nc TEM_3D.th.nc SAL_3D.th.nc uv3D.th.nc')
    save_boundary_nc('elev2D.th.nc', schism_zeta, roms_data.date, schism)
    save_boundary_nc('TEM_3D.th.nc', schism_temp, roms_data.date, schism)
    save_boundary_nc('SAL_3D.th.nc', schism_salt, roms_data.date, schism)
    save_boundary_nc('uv3D.th.nc', schism_uv, roms_data.date, schism)


