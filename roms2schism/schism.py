#!/usr/bin/env python
# coding: utf-8

"""Copyright 2023 University of Western Australia.

This file is part of ROMS2SCHISM.

ROMS2SCHISM is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

ROMS2SCHISM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with ROMS2SCHISM.  If not, see <http://www.gnu.org/licenses/>."""

import os
from itertools import islice
import numpy as np
from pyschism.mesh import Hgrid
from pyschism.mesh.vgrid import Vgrid
from roms2schism.geometry import transform_ll_to_cpp, bbox

class gr3(object):
    """Class for gr3 grid"""

    def __init__(self, filename):

        with open(filename,'r') as fid:
            # grid  name
            self.name = fid.readline().strip()
            # number of elements and nodes
            tmp = fid.readline().split()
            self.ne = int(tmp[0])
            self.nn = int(tmp[1])
            # first load nodes and values
            # not using nn
            tmp = list(islice(fid, self.nn))
            node_id, self.x, self.y, self.z = np.loadtxt(tmp,
                                                         dtype = {'names':('n','x','y','z'),
                                                                  'formats':('i4','f8','f8','f8')},
                                                         usecols = (0,1,2,3),
                                                         unpack=True)
            del node_id
            # elements
            tmp = list(islice(fid, self.ne))
            tmp_e = np.loadtxt(tmp, dtype='i4')
            self.e = tmp_e[:,2:] - 1

class schism_grid(object):
    """Class for SCHISM grid"""

    def __init__(self, schism_grid_file = 'hgrid.ll', schism_vgrid_file = 'vgrid.in',
                 schism_grid_dir = './', lonc = None, latc = None):

        bbox_offset = 0.01
        print('Reading SCHISM grid %s, %s...' % (schism_grid_file, schism_vgrid_file))
        # get schism mesh
        hgrid_filename = os.path.join(schism_grid_dir, schism_grid_file)
        hgrid = Hgrid.open(hgrid_filename,  crs = 'EPSG:4326')
        # get schism depths
        vgrid_filename = os.path.join(schism_grid_dir, schism_vgrid_file)
        vd = Vgrid.open(vgrid_filename)
        sigma = vd.sigma              # sigma values for vertical grid
        depth = hgrid.values          # this is grid bathymery
        zcor = depth[:,None] * sigma  # this is 2D array with layer depths at [nodes, layers]
        nvrt = zcor.shape[1]          # number of SCHISM layers

        self.lon = hgrid.coords[:,0]
        self.lat = hgrid.coords[:,1]
        self.lonc = np.average(self.lon) if lonc is None else lonc # reference coords
        self.latc = np.average(self.lat) if latc is None else latc # for conversion
        x, y = transform_ll_to_cpp(self.lon, self.lat,
                                   self.lonc, self.latc) # transform them to meters

        # get SCHISM open boundaries from grid file
        gdf = hgrid.boundaries.open.copy()
        opbd = gdf.indexes[0]       # need only first open boundary as 2nd is river
        zcor2 = zcor[opbd,:]        # depths at the boundary nodes
        blon = hgrid.coords[opbd,0]  # OB lons
        blat = hgrid.coords[opbd,1]  # OB lats
        NOP = len(blon)              # number of open boundary nodes
        self.b_xi, self.b_yi = x[opbd], y[opbd]  # only at the bry nodes
        self.b_bbox = bbox(blon, blat, offset = bbox_offset)
        self.NOP = NOP
        self.nvrt = nvrt
        self.b_lon = blon
        self.b_lat = blat
        self.b_depth = zcor2
        self.xi = x
        self.yi = y
        self.triangles = hgrid.triangles
        self.elements = hgrid.elements.array
        self.sides = hgrid.elements.sides
        self.depth = zcor
        self.bbox = bbox(self.lon, self.lat, offset = bbox_offset)
