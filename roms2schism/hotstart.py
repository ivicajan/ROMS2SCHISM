#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from netCDF4 import Dataset
from roms2schism import roms as rs
from roms2schism import interpolation as itp
from progressbar import progressbar

def save_hotstart_nc(outfile, eta2_data, temp_data, salt_data,
                     su2_data, sv2_data, w_data,
                     schism, h0):

    dst = Dataset(outfile, "w", format="NETCDF4")

    # dimensions:
    dst.createDimension('one_new', 1)
    nnodes = len(eta2_data)
    dst.createDimension('node', nnodes)
    dst.createDimension('elem', w_data.shape[0])
    dst.createDimension('side', su2_data.shape[0])
    dst.createDimension('nVert', su2_data.shape[1])
    dst.createDimension('ntracers', 2)

    time = dst.createVariable('time', 'f8', ('one_new'))
    dst['time'][:] = 0
    time.long_name = 'time'

    ths = dst.createVariable('iths', 'i4', ('one_new'))
    dst['iths'][:] = 0
    ths.long_name = 'iteration number'

    ifile = dst.createVariable('ifile', 'i4', ('one_new'))
    dst['ifile'][:] = 0
    ifile.long_name = 'file number'

    nfc = dst.createVariable('nsteps_from_cold', 'i4', ('one_new'))
    dst['nsteps_from_cold'][:] = 0
    nfc.long_name = 'number of steps from cold start'

    # dry nodes, sides, elements:

    idry = dst.createVariable('idry', 'i4', ('node'))
    dry_node = np.zeros(nnodes)
    dry_node[np.where(eta2_data < -schism.depth[:,0] + h0)] = 1
    dst['idry'][:] = dry_node
    idry.long_name = "wet/dry flag at nodes"

    idry_s = dst.createVariable('idry_s', 'i4', ('side'))
    dst['idry_s'][:] = 0
    dst['idry_s'][np.where([np.any(dry_node[s]) for s in schism.sides])] = 1
    idry_s.long_name = "wet/dry flag at sides"

    idry_e = dst.createVariable('idry_e', 'i4', ('elem'))
    dst['idry_e'][:] = 0
    dst['idry_e'][np.where([np.any(dry_node[nodes.compressed()])
                                  for nodes in schism.elements])] = 1
    idry_e.long_name = "wet/dry flag at elements"

    # elevations and velocities:

    eta2 = dst.createVariable('eta2', 'f8', ('node'))
    dst['eta2'][:] = eta2_data
    eta2.long_name = "elevation at nodes at current timestep"

    cse = dst.createVariable('cumsum_eta', 'f8', ('node'))
    dst['cumsum_eta'][:] = 0
    cse.long_name = 'cumsum eta'

    su2 = dst.createVariable('su2', 'f8', ('side', 'nVert'))
    dst['su2'][:,:] = su2_data
    su2.long_name = "u-velocity at side centres"

    sv2 = dst.createVariable('sv2', 'f8', ('side', 'nVert'))
    dst['sv2'][:,:] = sv2_data
    sv2.long_name = "v-velocity at side centres"

    we = dst.createVariable('we', 'f8', ('elem', 'nVert'))
    dst['we'][:,:] = w_data
    we.long_name = "vertical velocity at element centres"

    # tracers:

    tr_nd = dst.createVariable('tr_nd', 'f8', ('node', 'nVert', 'ntracers'))
    dst['tr_nd'][:,:,0] = temp_data
    dst['tr_nd'][:,:,1] = salt_data
    tr_nd.long_name = "tracer concentration at nodes"

    tr_nd0 = dst.createVariable('tr_nd0', 'f8', ('node', 'nVert', 'ntracers'))
    dst['tr_nd0'][:,:,:] = dst['tr_nd'][:,:,:]
    tr_nd0.long_name = "initial tracer concentration at nodes"

    tr_el = dst.createVariable('tr_el', 'f8', ('elem', 'nVert', 'ntracers'))
    temp_el = np.array([np.average(temp_data[nodes.compressed(),:], axis = 0)
                                    for nodes in schism.elements])
    salt_el = np.array([np.average(salt_data[nodes.compressed(),:], axis = 0)
                                    for nodes in schism.elements])
    dst['tr_el'][:,:,0] = temp_el
    dst['tr_el'][:,:,1] = salt_el
    tr_nd.long_name = "tracer concentration at elements"

    # other variables (turbulence, viscosity etc.) set to zero:

    q2 = dst.createVariable('q2', 'f8', ('node', 'nVert'))
    dst['q2'][:,:] = 0
    q2.long_name = "turbulent kinetic energy at sides and half levels"

    xl = dst.createVariable('xl', 'f8', ('node', 'nVert'))
    dst['xl'][:,:] = 0
    xl.long_name = "turbulent mixing length at sides and half levels"

    dfv = dst.createVariable('dfv', 'f8', ('node', 'nVert'))
    dst['dfv'][:,:] = 0
    dfv.long_name = "viscosity at nodes"

    dfh = dst.createVariable('dfh', 'f8', ('node', 'nVert'))
    dst['dfh'][:,:] = 0
    dfh.long_name = "diffusivity at nodes"

    dfq1 = dst.createVariable('dfq1', 'f8', ('node', 'nVert'))
    dst['dfq1'][:,:] = 0
    dfq1.long_name = "diffmin"

    dfq2 = dst.createVariable('dfq2', 'f8', ('node', 'nVert'))
    dst['dfq2'][:,:] = 0
    dfq2.long_name = "diffmax"

    dst.close()

def make_hotstart(schism, roms_data_filename, dcrit = 700,
                  roms_dir = './', roms_grid_filename = None,
                  lonc = 175., latc = -37., h0 = 0.01):
    """Creates hotstart.nc from initial results in ROMS output file.
    h0 is the minimum depth for wet nodes."""

    if roms_grid_filename is not None:
        fname = roms_grid_filename
    else:
        fname = os.path.join(roms_dir, roms_data_filename)
    roms_grid = rs.roms_grid(fname, schism.bbox)

    mask_OK = roms_grid.maskr == 1  # this is the case to avoid interp with masked land values
    nt = 1                          # number of times

    # read initial roms data:
    roms_data = rs.read_roms_data(fname, roms_grid, num_times = nt, get_w = True)
    
    node_interp = itp.interpolator(roms_grid,mask_OK, schism.xi, schism.yi, dcrit, lonc, latc)

    elt_x = np.array([np.average(schism.xi[nodes.compressed()]) for nodes in schism.elements])
    elt_y = np.array([np.average(schism.yi[nodes.compressed()]) for nodes in schism.elements])
    elt_interp = itp.interpolator(roms_grid,mask_OK, elt_x, elt_y, dcrit, lonc, latc)

    side_x = 0.5 * (schism.xi[schism.sides[:,0]] + schism.xi[schism.sides[:,1]])
    side_y = 0.5 * (schism.yi[schism.sides[:,0]] + schism.yi[schism.sides[:,1]])
    side_interp = itp.interpolator(roms_grid,mask_OK, side_x, side_y, dcrit, lonc, latc)

    Nz = len(roms_data.Cs_r)  # number of ROMS rho levels
    Nw = len(roms_data.Cs_w)  # number of ROMS w levels
    nnodes = len(schism.xi)   # number of SCHISM nodes
    nsides = len(schism.sides) # number of SCHISM sides 
    nelts = len(schism.elements) # number of SCHISM elements

    schism_node_depth = schism.depth # schism depths at the nodes [nnodes, nvrt]
    schism_side_depth = 0.5 * (schism_node_depth[schism.sides[:,0],:] + \
                               schism_node_depth[schism.sides[:,1],:])
    schism_elt_depth = np.array([np.average(schism_node_depth[nodes.compressed(),:], axis = 0)
                                 for nodes in schism.elements])

    schism_zeta = node_interp.interpolate(roms_data.zeta[0, mask_OK])
    schism_elt_zeta = elt_interp.interpolate(roms_data.zeta[0, mask_OK])
    schism_side_zeta = side_interp.interpolate(roms_data.zeta[0, mask_OK])

    roms_depths_at_schism_node = rs.roms_depth_point(schism_zeta, node_interp.depth_interp,
                                                      roms_data.vtransform,
                                                      roms_data.sc_r,roms_data.Cs_r, roms_data.hc)
    roms_w_depths_at_schism_elt = rs.roms_depth_point(schism_elt_zeta, elt_interp.depth_interp,
                                                      roms_data.vtransform,
                                                      roms_data.sc_w,roms_data.Cs_w, roms_data.hc)
    roms_depths_at_schism_side = rs.roms_depth_point(schism_side_zeta,
                                                      side_interp.depth_interp,
                                                      roms_data.vtransform,
                                                      roms_data.sc_r,roms_data.Cs_r, roms_data.hc)

    print('Interpolate temps:')
    val = np.zeros((Nz, nnodes))
    for k in progressbar(range(0, Nz)):
        val[k,:] = node_interp.interpolate(roms_data.temp[0,k,][mask_OK])
    schism_temp = itp.vert_interp(val, roms_depths_at_schism_node, -schism_node_depth)

    print('Interpolate salt:')
    val = np.zeros((Nz, nnodes))
    for k in progressbar(range(0, Nz)):
        val[k,:] = node_interp.interpolate(roms_data.salt[0,k,][mask_OK])
    schism_salt = itp.vert_interp(val, roms_depths_at_schism_node, -schism_node_depth)

    print('Interpolate u:')
    val = np.zeros((Nz, nsides))
    for k in progressbar(range(0, Nz)):
        val[k,:] = side_interp.interpolate(roms_data.u[0,k,][mask_OK])
    schism_su2 = itp.vert_interp(val, roms_depths_at_schism_side, -schism_side_depth)

    print('Interpolate v:')
    val = np.zeros((Nz, nsides))
    for k in progressbar(range(0, Nz)):
        val[k,:] = side_interp.interpolate(roms_data.v[0,k,][mask_OK])
    schism_sv2 = itp.vert_interp(val, roms_depths_at_schism_side, -schism_side_depth)

    print('Interpolate w:')
    val = np.zeros((Nw, nelts))
    for k in progressbar(range(0, Nw)):
        val[k,:] = elt_interp.interpolate(roms_data.w[0,k,][mask_OK])
    schism_w = itp.vert_interp(val, roms_w_depths_at_schism_elt, -schism_elt_depth)

    print('Writing hotstart.nc...')
    outfile = 'hotstart.nc'
    save_hotstart_nc(outfile, schism_zeta, schism_temp, schism_salt,
                     schism_su2, schism_sv2, schism_w,
                     schism, h0)
