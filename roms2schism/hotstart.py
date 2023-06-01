#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from roms2schism import roms as rs
from roms2schism import interpolation as itp
from progressbar import progressbar

def save_hotstart_nc(outfile, eta2_data, temp_data, salt_data,
                     su2_data, sv2_data,
                     schism):

    dst = Dataset(outfile, "w", format="NETCDF4")

    #dimensions
    dst.createDimension('nNodes', len(eta2_data))
    dst.createDimension('nSides', su2_data.shape[0])
    dst.createDimension('nLevels', su2_data.shape[1])

    #variables
    dst.createVariable('eta2', 'f', ('nNodes'))
    dst['eta2'][:] = eta2_data
    dst.createVariable('su2', 'f', ('nSides, nLevels'))
    dst['su2'][:,:] = su2_data
    dst.createVariable('sv2', 'f', ('nSides, nLevels'))
    dst['sv2'][:,:] = sv2_data

    # TODO

    dst.close()

def make_hotstart(schism, roms_data_filename, dcrit = 700,
                  roms_dir = './', roms_grid_filename = None,
                  lonc = 175., latc = -37.):
    """Creates hotstart.nc from initial results in ROMS output file"""

    if roms_grid_filename is not None:
        fname = roms_grid_filename
    else:
        fname = os.path.join(roms_dir, roms_data_filename)
    roms_grid = rs.read_roms_grid(fname, schism.bbox)

    mask_OK = roms_grid.maskr == 1  # this is the case to avoid interp with masked land values
    nt = 1                          # number of times

    # read initial roms data:
    roms_data = rs.read_roms_data(fname, roms_grid, num_times = nt)
    
    node_interp = itp.spatial_interp(roms_grid,mask_OK, schism.xi, schism.yi, dcrit, lonc, latc)
    side_x = 0.5 * (schism.xi[schism.sides[:,0]] + schism.xi[schism.sides[:,1]])
    side_y = 0.5 * (schism.yi[schism.sides[:,0]] + schism.yi[schism.sides[:,1]])
    side_interp = itp.spatial_interp(roms_grid,mask_OK, side_x, side_y, dcrit, lonc, latc)

    Nz = len(roms_data.Cs_r)  # number of ROMS levels
    nnodes = len(schism.xi)   # number of SCHISM nodes
    nsides = len(schism.sides) # number of SCHISM sides 
    schism_node_depth = schism.depth # schism depths at the nodes [nnodes, nvrt]
    schism_zeta = np.zeros(nnodes) # zeta is also needed to compute ROMS depths
    schism_temp = np.zeros((nnodes, schism.nvrt))
    schism_salt = np.zeros((nnodes, schism.nvrt))

    schism_side_zeta = np.zeros(nsides)
    schism_side_depth = 0.5 * (schism_node_depth[schism.sides[:,0]] + \
                               schism_node_depth[schism.sides[:,1]])
    schism_su2 = np.zeros((nsides, schism.nvrt))
    schism_sv2 = np.zeros((nsides, schism.nvrt))

    schism_zeta = itp.interp2D(roms_data.zeta[0, mask_OK], node_interp)
    roms_depths_at_schism_node = rs.roms_depth_point(schism_zeta, node_interp.depth_interp,
                                                      roms_data.vtransform,
                                                      roms_data.sc_r,roms_data.Cs_r, roms_data.hc)
    schism_side_zeta = itp.interp2D(roms_data.zeta[0, mask_OK], side_interp)
    roms_depths_at_schism_side = rs.roms_depth_point(schism_side_zeta,
                                                      side_interp.depth_interp,
                                                      roms_data.vtransform,
                                                      roms_data.sc_r,roms_data.Cs_r, roms_data.hc)

    print('Interpolate temps:')
    val = np.zeros((Nz, nnodes))
    for k in progressbar(range(0, Nz)):
        val[k,:] = itp.interp2D(roms_data.temp[0,k,][mask_OK], node_interp)
    schism_temp = itp.vert_interp(val, roms_depths_at_schism_node, -schism_node_depth)

    print('Interpolate salt:')
    val = np.zeros((Nz, nnodes))
    for k in progressbar(range(0, Nz)):
        val[k,:] = itp.interp2D(roms_data.salt[0,k,][mask_OK], node_interp)
    schism_salt = itp.vert_interp(val, roms_depths_at_schism_node, -schism_node_depth)

    print('Interpolate u:')
    val = np.zeros((Nz, nsides))
    for k in progressbar(range(0, Nz)):
        val[k,:] = itp.interp2D(roms_data.u[0,k,][mask_OK], side_interp)
    schism_su2 = itp.vert_interp(val, roms_depths_at_schism_side, -schism_side_depth)

    print('Interpolate v:')
    val = np.zeros((Nz, nsides))
    for k in progressbar(range(0, Nz)):
        val[k,:] = itp.interp2D(roms_data.v[0,k,][mask_OK], side_interp)
    schism_sv2 = itp.vert_interp(val, roms_depths_at_schism_side, -schism_side_depth)

    # TODO ...

    outfile = 'hotstart.nc'
    save_hotstart_nc(outfile, schism_zeta, schism_temp, schism_salt,
                     schism_su2, schism_sv2,
                     schism)
