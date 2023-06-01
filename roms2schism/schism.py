#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from munch import Munch as Bunch
from pyschism.mesh import Hgrid
from pyschism.mesh.vgrid import Vgrid
from roms2schism.geometry import transform_ll_to_cpp

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
    schism.sides = hgrid.elements.sides
    schism.depth = zcor
    schism.bbox = schism_bbox(hgrid.coords[:,0], hgrid.coords[:,1])
    print('Computing SCHISM zcor is done!')    
    return schism
    
def schism_bbox(blon, blat):
    """
    Calculate boundary box of schism grid
    """
    # add small offeset for interpolation
    offset = 0.01
    xmin, xmax = np.min(blon)-offset, np.max(blon)+offset
    ymin, ymax = np.min(blat)-offset, np.max(blat)+offset        
    return np.array([xmin, xmax, ymin, ymax])

