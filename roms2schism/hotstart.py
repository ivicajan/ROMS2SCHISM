#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from roms2schism import roms as rs
from roms2schism import interpolation as itp
from progressbar import progressbar

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
    schism_zeta = np.zeros((nt, nnodes,1,1)) # zeta is also needed to compute ROMS depths
    schism_temp = np.zeros((nt, nnodes, schism.nvrt, 1))  # SCHISM is using (time, node, vert, 1) 
    schism_salt = np.zeros((nt, nnodes, schism.nvrt, 1))

    schism_side_zeta = np.zeros((nt, nsides, 1, 1))
    schism_side_depth = 0.5 * (schism_node_depth[schism.sides[:,0]] + \
                               schism_node_depth[schism.sides[:,1]])
    schism_uv = np.zeros((nt, nsides, schism.nvrt, 2))

    schism_zeta[0,:,0,0] = itp.interp2D(roms_data.zeta[0, mask_OK], node_interp)
    roms_depths_at_schism_node = rs.roms_depth_point(schism_zeta[0,:,0,0], node_interp.depth_interp,
                                                      roms_data.vtransform,
                                                      roms_data.sc_r,roms_data.Cs_r, roms_data.hc)
    schism_side_zeta[0,:,0,0] = itp.interp2D(roms_data.zeta[0, mask_OK], side_interp)
    roms_depths_at_schism_side = rs.roms_depth_point(schism_side_zeta[0,:,0,0],
                                                      side_interp.depth_interp,
                                                      roms_data.vtransform,
                                                      roms_data.sc_r,roms_data.Cs_r, roms_data.hc)

    print('Interpolate temps:')
    val = np.zeros((Nz, nnodes))
    for k in progressbar(range(0, Nz)):
        val[k,:] = itp.interp2D(roms_data.temp[0,k,][mask_OK], node_interp)
    schism_temp[0,:,:,0] = itp.vert_interp(val, roms_depths_at_schism_node, -schism_node_depth)

    print('Interpolate salt:')
    val = np.zeros((Nz, nnodes))
    for k in progressbar(range(0, Nz)):
        val[k,:] = itp.interp2D(roms_data.salt[0,k,][mask_OK], node_interp)
    schism_salt[0,:,:,0] = itp.vert_interp(val, roms_depths_at_schism_node, -schism_node_depth)

    print('Interpolate u:')
    val = np.zeros((Nz, nsides))
    for k in progressbar(range(0, Nz)):
        val[k,:] = itp.interp2D(roms_data.u[0,k,][mask_OK], side_interp)
    schism_uv[0,:,:,0] = itp.vert_interp(val, roms_depths_at_schism_side, -schism_side_depth)

    print('Interpolate v:')
    val = np.zeros((Nz, nsides))
    for k in progressbar(range(0, Nz)):
        val[k,:] = itp.interp2D(roms_data.v[0,k,][mask_OK], side_interp)
    schism_uv[0,:,:,1] = itp.vert_interp(val, roms_depths_at_schism_side, -schism_side_depth)

    # TBC ...
