#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.interpolate import interp1d
from munch import Munch as Bunch
from roms2schism.geometry import transform_ll_to_cpp

def calc_weights(xyin, xyout, dcrit, kdtree):
    tri = Delaunay(xyin)    
    s = tri.find_simplex(xyout)
    # Compute the barycentric coordinates (these are the weights)
    X = tri.transform[s,:2]
    Y = xyout - tri.transform[s,2]
    b = np.einsum('ijk,ik->ij', X, Y)
    weights = np.c_[b, 1 - b.sum(axis=1)]    
    # These are the vertices of the output points
    verts = tri.simplices[s]

    npt = np.shape(xyout)[0]    # number of output locations
    use_closest = []            # list of point indices to do nearest neighbour interpolation on
    closest_in = []             # indices of closest input points
    for i in range(0, npt):
        if s[i] >=0:
            # check for the crtical distance
            dx = np.min(np.abs(xyin[verts[i]][:,0]- xyout[i][0]))
            dy = np.min(np.abs(xyin[verts[i]][:,1]- xyout[i][1]))
            closest = np.logical_or(dx>dcrit, dy>dcrit)
        else: # point is outside the triangulation
            closest = True
        if closest:
            use_closest.append(i)
            r, c = kdtree.query(xyout[i])
            closest_in.append(c)

    return weights, verts, use_closest, closest_in

def spatial_interp(roms_grid, mask, coord_x, coord_y, dcrit, lonc, latc):

    interp = Bunch()
    # Prepare for spatial (2d) interpolation
    x2, y2 = transform_ll_to_cpp(roms_grid.lonr, roms_grid.latr,
                                 lonc, latc) # transform to [m], the same projection as SCHISM
    interp.XY = np.vstack((x2[mask], y2[mask])).T
    interp.kdtree = cKDTree(interp.XY)
    interp.XYout = np.vstack((coord_x.ravel(),coord_y.ravel())).T   # the same for SCHISM sponge nodes
    interp.weights, interp.verts, interp.use_closest, interp.closest_in = calc_weights(interp.XY, interp.XYout, dcrit, interp.kdtree)
    
    # interp 2D depth which is time invariant
    interp.depth_interp = interp2D(roms_grid.h[mask], interp)
    
    return interp

def interp2D(z, interp):
    """
    Perform the interpolation
    """    
    out = (z[interp.verts]*interp.weights).sum(axis=1)
    for i, closest in zip(interp.use_closest, interp.closest_in):
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

