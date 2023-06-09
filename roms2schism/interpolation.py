#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.interpolate import interp1d

class interpolator(object):
    """Class for spatial interpolator"""

    def __init__(self, roms_grid, mask, coord_x, coord_y, dcrit):

        self.XY = np.vstack((roms_grid.x[mask], roms_grid.y[mask])).T
        self.kdtree = cKDTree(self.XY)
        self.XYout = np.vstack((coord_x.ravel(),coord_y.ravel())).T   # the same for SCHISM sponge nodes
        self.dcrit = dcrit
        self.calc_weights()

        # interpolate 2D depth which is time invariant
        self.depth_interp = self.interpolate(roms_grid.h[mask])

    def calc_weights(self):
        """Calculate interpolation weights"""

        tri = Delaunay(self.XY)
        s = tri.find_simplex(self.XYout)
        # Compute the barycentric coordinates (these are the weights)
        X = tri.transform[s,:2]
        Y = self.XYout - tri.transform[s,2]
        b = np.einsum('ijk,ik->ij', X, Y)
        self.weights = np.c_[b, 1 - b.sum(axis=1)]
        # These are the vertices of the output points
        self.verts = tri.simplices[s]

        npt = np.shape(self.XYout)[0] # number of output locations
        self.use_closest = []    # list of point indices to do nearest neighbour interpolation on
        self.closest_in = []     # indices of closest input points
        for i in range(0, npt):
            if s[i] >=0:
                # check for the crtical distance
                dx = np.min(np.abs(self.XY[self.verts[i]][:,0]- self.XYout[i][0]))
                dy = np.min(np.abs(self.XY[self.verts[i]][:,1]- self.XYout[i][1]))
                closest = np.logical_or(dx > self.dcrit, dy > self.dcrit)
            else: # point is outside the triangulation
                closest = True
            if closest:
                self.use_closest.append(i)
                r, c = self.kdtree.query(self.XYout[i])
                self.closest_in.append(c)

    def interpolate(self, z):
        """
        Perform the interpolation
        """
        out = (z[self.verts]*self.weights).sum(axis=1)
        for i, closest in zip(self.use_closest, self.closest_in):
            out[i] = z[closest]
        return out

def vert_interp(temp_interp, roms_depths_at_schism_pt, schism_depth):
    """Vertical interpolation"""

    schism_temp = np.zeros((np.size(schism_depth,0), np.size(schism_depth,1)))  # schism is using (node, level)
    tmp_depth = np.zeros((roms_depths_at_schism_pt.shape[0]))
    tmp_var  = np.zeros((roms_depths_at_schism_pt.shape[0]))
    for n in range(0, np.size(schism_depth,0)):
        tmp_depth = roms_depths_at_schism_pt[:,n]
        tmp_var = temp_interp[:,n]
        f = interp1d(tmp_depth, tmp_var, kind='linear', bounds_error = False,
                     fill_value = (tmp_var[0], tmp_var[-1]))
        schism_temp[n,:] = f(schism_depth[n,:])
    return schism_temp

