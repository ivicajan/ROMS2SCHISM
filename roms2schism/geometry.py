#!/usr/bin/env python
# coding: utf-8

"""Copyright 2023 University of Western Australia.

This file is part of ROMS2SCHISM.

ROMS2SCHISM is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

ROMS2SCHISM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with ROMS2SCHISM.  If not, see <http://www.gnu.org/licenses/>."""

import numpy as np

def transform_ll_to_cpp(lon, lat, lonc, latc):
    longitude=lon/180*np.pi
    latitude=lat/180*np.pi
    radius=6378206.4
    loncc=lonc/180*np.pi
    latcc=latc/180*np.pi
    lon_new=[radius*(longitude[i]-loncc)*np.cos(latcc) for i in np.arange(len(longitude))]
    lat_new=[radius*latitude[i] for i in np.arange(len(latitude))]
    return np.array(lon_new), np.array(lat_new)

def rot2d(x, y, ang): #rotate vectors by geometric angle
    dims = x.shape
    if len(dims)==3:
        ang = np.tile(ang,[dims[0],1,1])
    if len(dims)==4:
        ang = np.tile(ang,[dims[0],dims[1], 1,1])
    xr = x*np.cos(ang) - y*np.sin(ang)
    yr = x*np.sin(ang) + y*np.cos(ang)
    return xr, yr

def bbox(x, y, offset = 0):
    """
    Calculate boundary box of specified points, with optional offset
    """
    xmin, xmax = np.min(x)-offset, np.max(x)+offset
    ymin, ymax = np.min(y)-offset, np.max(y)+offset
    return np.array([xmin, xmax, ymin, ymax])

def u2rho(u):
    """
    Put the u field onto the rho field for the c-grid
    Parameters
    ----------
    u : masked array like
    Input u field

    Returns
    -------
    rho : masked array
    """
    u = np.ma.array(u, copy=False)
    shp = np.array(u.shape)
    nshp = shp.copy()
    nshp[-1] = nshp[-1] + 1
    fore = np.product([shp[i] for i in np.arange(0, u.ndim - 1)]).astype(int)
    nfld = np.ones([fore, nshp[-1]])
    nfld[:, 1:-1] = 0.5 * \
        (u.reshape([fore, shp[-1]])[:, 0:-1].filled(np.nan) +
         u.reshape([fore, shp[-1]])[:, 1:].filled(np.nan))
    nfld[:, 0] = nfld[:, 1] + (nfld[:, 2] - nfld[:, 3])
    nfld[:, -1] = nfld[:, -2] + (nfld[:, -2] - nfld[:, -3])
    return np.ma.fix_invalid(nfld.reshape(nshp), copy=False, fill_value=1e+37)
    
def v2rho(v):
    """
    Put the v field onto the rho field for the c-grid

    Parameters
    ----------
    v : masked array like
        Input v field

    Returns
    -------
    rho : masked array
    """
    v = np.ma.array(v, copy=False)
    shp = np.array(v.shape)
    nshp = shp.copy()
    nshp[-2] = nshp[-2] + 1
    fore = np.product([shp[i] for i in np.arange(0, v.ndim - 2)]).astype(int)
    nfld = np.ones([fore, nshp[-2], nshp[-1]])
    nfld[:, 1:-1, :] = 0.5 * \
        (v.reshape([fore, shp[-2], shp[-1]])[:, 0:-1, :].filled(np.nan) +
         v.reshape([fore, shp[-2], shp[-1]])[:, 1:, :].filled(np.nan))
    nfld[:, 0, :] = nfld[:, 1, :] + (nfld[:, 2, :] - nfld[:, 3, :])
    nfld[:, -1, :] = nfld[:, -2, :] + (nfld[:, -2, :] - nfld[:, -3, :])
    return np.ma.fix_invalid(nfld.reshape(nshp), copy=False, fill_value=1e+37)

