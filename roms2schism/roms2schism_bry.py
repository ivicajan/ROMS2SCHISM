#!/usr/bin/env python
# coding: utf-8

from progressbar import progressbar

import warnings
warnings.filterwarnings("ignore")

import os, sys
from munch import Munch as Bunch

from netCDF4 import Dataset, num2date, date2num
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay, cKDTree

from datetime import datetime, timedelta

from pyschism.mesh import Hgrid
from pyschism.mesh.vgrid import Vgrid

# define functions

def readgr3(filename):
    from itertools import islice;
    out = Bunch()
    with open(filename,'r') as fid:
        # grid  name
        out.name=fid.readline().strip();        
        # number of elements and nodes
        tmp=fid.readline().split();
        out.ne=int(tmp[0]);
        out.nn=int(tmp[1]);
        # first load nodes and values 
        # not using nn
        tmp=list(islice(fid,out.nn));
        node_id,out.x,out.y,out.z=np.loadtxt(tmp,
                                             dtype={'names':('n','x','y','z'),
                                                    'formats':('i4','f8','f8','f8')},
                                             usecols = (0,1,2,3),
                                             unpack=True)
        del node_id;
        # elements
        tmp=list(islice(fid,out.ne));
        tmp_e=np.loadtxt(tmp,dtype='i4');
        out.e=tmp_e[:,2:]-1;
        fid.close();
        return out

def transform_ll_to_cpp(lon, lat, lonc, latc):
    # harcoded central location for projection and degrees into meters
    # but can compute mean lonc: lonc=(np.max(lon)+np.min(lon))/2.0
    longitude=lon/180*np.pi
    latitude=lat/180*np.pi
    radius=6378206.4
    loncc=lonc/180*np.pi
    latcc=latc/180*np.pi
    lon_new=[radius*(longitude[i]-loncc)*np.cos(latcc) for i in np.arange(len(longitude))]
    lat_new=[radius*latitude[i] for i in np.arange(len(latitude))]
    return np.array(lon_new), np.array(lat_new)

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

def rot2d(x, y, ang): #rotate vectors by geometric angle
    dims = x.shape
    #print(np.shape(x))
    #print(np.shape(y))
    #print(np.shape(ang))
    if len(dims)==3:
        ang = np.tile(ang,[dims[0],1,1])
    if len(dims)==4:
        ang = np.tile(ang,[dims[0],dims[1], 1,1])
    xr = x*np.cos(ang) - y*np.sin(ang)
    yr = x*np.sin(ang) + y*np.cos(ang)
    return xr, yr

def schism_bbox(blon, blat):
    """
    Calculate boundary box of schism grid
    """
    # add small offeset for interpolation
    offset = 0.01
    xmin, xmax = np.min(blon)-offset, np.max(blon)+offset
    ymin, ymax = np.min(blat)-offset, np.max(blat)+offset        
    return np.array([xmin, xmax, ymin, ymax])

def schism_grid(schism_grid_file, schism_vgrid_file, schism_grid_dir = './',
                lonc = 175., latc = -37.):
    schism = Bunch()
    # get schism mesh
    schism_mesh = os.path.join(schism_grid_dir, schism_grid_file)
    hgrid = Hgrid.open(schism_mesh,  crs='EPSG:4326')   
    # get schism depths
    schism_vgrid = os.path.join(schism_grid_dir, schism_vgrid_file)
    vd=Vgrid.open(schism_vgrid)
    sigma = vd.sigma              # sigma values for vertical grid
    depth = hgrid.values          # this is grid bathymery
    zcor = depth[:,None]*sigma    # this is 2D array with layer depths at [nodes, layers]
    nvrt = zcor.shape[1]          # number of SCHISM layers
    x, y = transform_ll_to_cpp(hgrid.coords[:,0], hgrid.coords[:,1],
                               lonc, latc) # transform them to meters
    
    # get SCHISM open boundaries from grid file
    gdf = hgrid.boundaries.open.copy()    
    opbd = gdf.indexes[0]       # need only first open boundary as 2nd is river
    zcor2 = zcor[opbd,:]        # depths at the boundary nodes    
    blon = hgrid.coords[opbd,0]  # OB lons
    blat = hgrid.coords[opbd,1]  # OB lats
    NOP = len(blon)              # number of open boundary nodes    
    xi, yi = x[opbd], y[opbd]  # only at the bry nodes    
    schism.b_bbox = schism_bbox(blon, blat)
    schism.NOP = NOP
    schism.nvrt = nvrt     
    schism.b_lon = blon
    schism.b_lat = blat
    schism.b_depth = zcor2
    schism.b_xi = xi
    schism.b_yi = yi
    schism.lon = hgrid.coords[:,0]
    schism.lat = hgrid.coords[:,1]
    schism.xi = x
    schism.yi = y
    schism.triangles = hgrid.triangles    
    schism.depth = zcor
    schism.bbox = schism_bbox(hgrid.coords[:,0], hgrid.coords[:,1])
    print('Computing SCHISM zcor is done!')    
    return schism

def calc_weights(xyin, xyout):
    tri = Delaunay(xyin)    
    s = tri.find_simplex(xyout)
    # Compute the barycentric coordinates (these are the weights)
    X = tri.transform[s,:2]
    Y = xyout - tri.transform[s,2]
    b = np.einsum('ijk,ik->ij', X, Y)
    weights = np.c_[b, 1 - b.sum(axis=1)]    
    # These are the vertices of the output points
    verts = tri.simplices[s]
    return weights, verts, s

def interp2D(z, weights, verts, simplices, XY, kdtree, XYout, dcrit):
    """
    Perform the interpolation
    """    
    out = (z[verts]*weights).sum(axis=1)

    npt = np.shape(XYout)[0]    # number of output locations
    for i in range(0, npt):
        if simplices[i] >=0:
            # check for the crtical distance
            dx = np.min(np.abs(XY[verts[i]][:,0]- XYout[i][0]))
            dy = np.min(np.abs(XY[verts[i]][:,1]- XYout[i][1]))
            use_closest = np.logical_or(dx>dcrit, dy>dcrit)
        else: # point is outside the triangulation
            use_closest = True
        if use_closest:
            r, closest = kdtree.query(XYout[i])
            out[i] = z[closest]

    return out

def vert_interp(temp_interp, roms_depths_at_schism_node, schism_depth):
    schism_temp = np.zeros((np.size(schism_depth,0), np.size(schism_depth,1)))  # schism is using (node, level)
    tmp_depth = np.zeros((roms_depths_at_schism_node.shape[0]))
    tmp_var  = np.zeros((roms_depths_at_schism_node.shape[0]))
    for n in range(0, np.size(schism_depth,0)):
        tmp_depth = roms_depths_at_schism_node[:,n]
        tmp_var = temp_interp[:,n]
        f = interp1d(tmp_depth, tmp_var, kind='linear', bounds_error = False,
                     fill_value = (tmp_var[0], tmp_var[-1]))
        schism_temp[n,:] = f(schism_depth[n,:])
    return schism_temp

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

def spatial_interp(roms_grid, mask, coord_x, coord_y, dcrit, lonc, latc):
    # Prepare for spatial (2d) interpolation
    x2, y2 = transform_ll_to_cpp(roms_grid.lonr, roms_grid.latr,
                                 lonc, latc) # transform to [m], the same projection as SCHISM
    XY = np.vstack((x2[mask], y2[mask])).T
    kdtree = cKDTree(XY)
    XYout = np.vstack((coord_x.ravel(),coord_y.ravel())).T   # the same for SCHISM sponge nodes
    weights, verts, simplices = calc_weights(XY, XYout)
    
    # interp 2D depth which is time invariant
    depth_interp = interp2D(roms_grid.h[mask], weights, verts, simplices, XY, kdtree, XYout, dcrit)
    
    return weights, verts, simplices, XY, kdtree, XYout, depth_interp

def make_boundary(schism, template, dates, dcrit = 700,
                  roms_dir = './', roms_grid_filename = None,
                  lonc = 175., latc = -37.):
    # ## Part for boundary conditions ROMS -> SCHISM

    # part to load ROMS grid for given subset
    if roms_grid_filename is not None:
        fname = roms_grid_filename
    else:
        fname = os.path.join(roms_dir, dates[0].strftime(template))
    roms_grid = read_roms_grid(fname, schism.b_bbox)

    mask_OK = roms_grid.maskr == 1  # this is the case to avoid interp with masked land values

    roms_data = read_roms_files(roms_dir, roms_grid, template, dates)
    
    weights, verts, simplices, XY, kdtree, XYout, depth_interp = spatial_interp(roms_grid, mask_OK, schism.b_xi, schism.b_yi, dcrit, lonc, latc)

    # init outputs 
    nt = len(roms_data.date)  # need to loop over time for each record
    Nz = len(roms_data.Cs_r)  # number of ROMS levels
    schism_depth = schism.b_depth                             # schism depths at the open bounday nodes [NOP, nvrt]
    schism_zeta = np.zeros((nt, schism.NOP,1,1))              # zeta is also needed to compute ROMS depths
    schism_temp = np.zeros((nt, schism.NOP, schism.nvrt, 1))  # schims is using (time, node, vert, 1) 
    schism_salt = np.zeros((nt, schism.NOP, schism.nvrt, 1))  # schims is using (time, node, vert, 1) 
    schism_uv = np.zeros((nt, schism.NOP, schism.nvrt, 2))    # schims is using (time, node, vert, 2) 

    print('total steps: %d ' %nt, end='>')
    for it in progressbar(range(0, nt)):
        # get first zeta as I need it for depth calculation
        schism_zeta[it,:,0,0] = interp2D(roms_data.zeta[it, mask_OK], weights, verts, XY, kdtree,
                                         XYout, dcrit)
        # compute depths for each ROMS levels at the specific SCHISM locations
        roms_depths_at_schism_node = roms_depth_point(schism_zeta[it,:,0,0], depth_interp,
                                                      roms_data.vtransform, roms_data.sc_r,
                                                      roms_data.Cs_r, roms_data.hc)
        # start with temperature variable for each ROMS layer, need to do that for all 3D variables (temp, salt, u, v)
        temp_interp = np.zeros((Nz, schism.NOP))   # this is temp at ROMS levels
        for k in range(0, Nz):   
            temp_interp[k,:] = interp2D(roms_data.temp[it,k,][mask_OK], weights, verts,
                                        XY, kdtree, XYout, dcrit)
        # interpolate in vertical to SCHISM depths
        schism_temp[it,:,:,0] = vert_interp(temp_interp, roms_depths_at_schism_node, -schism_depth)

        # interp salt variable 
        temp_interp = np.zeros((Nz, schism.NOP))
        for k in range(0,Nz):
            temp_interp[k,:] = interp2D(roms_data.salt[it,k,][mask_OK], weights, verts,
                                        XY, kdtree, XYout, dcrit)
        # now you need to interp temp for each NOP at SCHISM depths
        schism_salt[it,:,:,0] = vert_interp(temp_interp, roms_depths_at_schism_node, -schism_depth)

        # interp u variable 
        temp_interp = np.zeros((Nz, schism.NOP))
        for k in range(0,Nz):
            temp_interp[k,:] = interp2D(roms_data.u[it,k,][mask_OK], weights, verts, XY,
                                        kdtree, XYout, dcrit)
        # now you need to interp temp for each NOP at SCHISM depths
        schism_uv[it,:,:,0] = vert_interp(temp_interp, roms_depths_at_schism_node, -schism_depth)

        # interp v variable 
        temp_interp = np.zeros((Nz, schism.NOP))
        for k in range(0,Nz):
            temp_interp[k,:] = interp2D(roms_data.v[it,k,][mask_OK], weights, verts, XY,
                                        kdtree, XYout, dcrit)
        # now you need to interp temp for each NOP at SCHISM depths
        schism_uv[it,:,:,1] = vert_interp(temp_interp, roms_depths_at_schism_node, -schism_depth)
    print('Done interpolating')
    # now you need to save them in the boundary files
    os.system('rm  -f elev2D.th.nc TEM_3D.th.nc SAL_3D.th.nc uv3D.th.nc')
    save_boundary_nc('elev2D.th.nc', schism_zeta, roms_data.date, schism)
    save_boundary_nc('TEM_3D.th.nc', schism_temp, roms_data.date, schism)
    save_boundary_nc('SAL_3D.th.nc', schism_salt, roms_data.date, schism)
    save_boundary_nc('uv3D.th.nc', schism_uv, roms_data.date, schism)
        
    return


def make_nudging(schism, template, dates, dcrit = 700, roms_dir = './',
                 roms_grid_filename = None, lonc = 175., latc = -37.):
    # ## Part with nudging zone, 
    # ### it needs more points (defined in nudge.gr3) and that file is made using gen_nudge.f90

    sponge = readgr3('nudge.gr3')
    OK = np.where(sponge.z != 0)
    sponge_x = sponge.x[OK]; sponge_y = sponge.y[OK]; sponge_depth = schism.depth[OK]; 
    np.shape(sponge_x), np.shape(sponge_depth)

    # repeat all that we had for boundaries but now for "OK" points
    sponge_bbox = schism_bbox(sponge_x, sponge_y)

    # part to load ROMS grid for given subset
    if roms_grid_filename is not None:
        fname = roms_grid_filename
    else:
        fname = os.path.join(roms_dir, dates[0].strftime(template))
    roms_grid = read_roms_grid(fname, sponge_bbox)
    mask_OK = roms_grid.maskr == 1  # this is the case to avoid interp with masked land values

    roms_data = read_roms_files(roms_dir, roms_grid, template, dates)
      
    weights, verts, simplices, XY, kdtree, XYout, depth_interp = spatial_interp(roms_grid, mask_OK, sponge_x, sponge_y, dcrit, lonc, latc)

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
        schism_zeta[it,:,0,0] = interp2D(roms_data.zeta[it, mask_OK], weights, verts, XY,
                                         kdtree, XYout, dcrit)
        # compute depths for each ROMS levels at the specific SCHISM locations
        roms_depths_at_schism_node = roms_depth_point(schism_zeta[it,:,0,0], depth_interp,
        roms_data.vtransform, roms_data.sc_r, roms_data.Cs_r, roms_data.hc)
        # start with temperature variable for each ROMS layer, need to do that for all 3D variables (temp, salt, u, v)
        temp_interp = np.zeros((Nz, Np))   # this is temp at ROMS levels
        for k in range(0, Nz):   
            temp_interp[k,:] = interp2D(roms_data.temp[it,k,][mask_OK], weights, verts, XY,
                                        kdtree, XYout, dcrit)
        # interpolate in vertical to SCHISM depths
        schism_temp[it,:,:,0] = vert_interp(temp_interp, roms_depths_at_schism_node, -sponge_depth)
        # interp salt variable 
        temp_interp = np.zeros((Nz, Np))
        for k in range(0,Nz):
            temp_interp[k,:] = interp2D(roms_data.salt[it,k,][mask_OK], weights, verts, XY,
                                        kdtree, XYout, dcrit)
        # now you need to interp temp for each NOP at SCHISM depths
        schism_salt[it,:,:,0] = vert_interp(temp_interp, roms_depths_at_schism_node, -sponge_depth)

    os.system('rm -f TEM_nu.nc SAL_nu.nc')
    # now you need to save them in the boundary files
    save_nudging_nc('TEM_nu.nc', schism_temp, roms_data.date, np.array(OK))
    save_nudging_nc('SAL_nu.nc', schism_salt, roms_data.date, np.array(OK))


def main(dates, template, bry=False, nudge=False, dcrit = 700, schism_grid_dir = './',
         roms_dir = './', roms_grid_filename = None, lonc = 175., latc = -37.):
    # ## Actual start of the roms2schism interpolation
    
    # part with reading SCHISM mesh
    schism_grid_file = 'hgrid.ll'
    schism_vgrid_file = 'vgrid.in'
    schism = schism_grid(schism_grid_file, schism_vgrid_file, schism_grid_dir, lonc, latc)
    
    if bry is True:
        print('Making bry files for SCHISM')
        make_boundary(schism, template, dates, dcrit, roms_dir, roms_grid_filename)
        
    if nudge is True:
        print('Making nudging files for SCHISM')
        make_nudging(schism, template, dates, dcrit, roms_dir, roms_grid_filename)
        
    return    
        

if __name__=='__main__':
    
    from argparse import ArgumentParser    
    parser = ArgumentParser()
    parser.add_argument('--start_date', default='20200101', help='First history date (yyyymmdd)')
    parser.add_argument('--ndays', default=30,  type=int, help='number of days to process')
    parser.add_argument('--dcrit', default=700,  type=float, help='maximum distance for interpolation - if distance larger than dcrit, use closest value from ROMS grid, to avoid interpolating over land (should be slightly larger than ROMS grid resolution)')
    parser.add_argument('--schism_grid_dir', default='./', help='SCHISM grid directory')
    parser.add_argument('--roms_dir', default='./', help='ROMS output directory')
    parser.add_argument('--roms_grid_filename', default=None, help='ROMS grid filename (None if no grid required)')
    parser.add_argument('--lonc', default=175.,  type=float, help='reference longitude for converting coordinates to metres')
    parser.add_argument('--latc', default=-37.,  type=float, help='reference latitude for converting coordinates to metres')
    parser.add_argument('--bry', default=False, help='make boundary file')
    parser.add_argument('--nudge', default=False, help='make nudging file')
    parser.add_argument('--template', default='model_avg_%Y%m%d.nc', help='roms output filename template')
    # For nudging you don't need hourly (his) data and faster approach (and better) is to use avg file
    # First call the prog to create bry file with template for his, and then again for nudge but now using template for avg
    args = parser.parse_args()
    dates = datetime.strptime(args.start_date,'%Y%m%d') + np.arange(args.ndays)*timedelta(days=1)
    main(dates, args.template, args.bry, args.nudge, args.dcrit,
         args.schism_grid_dir, args.roms_dir, args.roms_grid_filename,
         args.lonc, args.latc)
