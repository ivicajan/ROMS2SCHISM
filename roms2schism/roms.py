#!/usr/bin/env python
# coding: utf-8

"""Copyright 2023 University of Western Australia.

This file is part of ROMS2SCHISM.

ROMS2SCHISM is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

ROMS2SCHISM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with ROMS2SCHISM.  If not, see <http://www.gnu.org/licenses/>."""

import os
import numpy as np
from netCDF4 import Dataset, num2date
from roms2schism import geometry as geom

class roms_grid(object):
    """Class for ROMS grid"""
    
    def __init__(self, filename, grid_dir, bbox, lonc, latc):
        """Reads ROMS grid data from ROMS grid file or output."""

        print('Reading roms grid %s...' % filename)
        fname = os.path.join(grid_dir, filename)
        nc = Dataset(fname,'r')
        lonr = nc.variables['lon_rho'][:]
        latr = nc.variables['lat_rho'][:]
        self.get_bbox_indices(lonr, latr, bbox)
        print('  bbox subset i0=%d, i1=%d, j0=%d, j1=%d' %(self.i0,self.i1,self.j0,self.j1))
        self.h = nc.variables['h'][self.j0:self.j1, self.i0:self.i1]
        # if east/north velocities not present, need to rotate and process from staggered velocities:
        self.rotate = not all([var in nc.variables for var in ['u_eastward', 'v_northward']])
        if self.rotate:
            self.angle = nc.variables['angle'][self.j0:self.j1, self.i0:self.i1]
        self.lonr = lonr[self.j0:self.j1, self.i0:self.i1]
        self.latr = latr[self.j0:self.j1, self.i0:self.i1]
        self.maskr = nc.variables['mask_rho'][self.j0:self.j1, self.i0:self.i1]
        self.x, self.y = geom.transform_ll_to_cpp(self.lonr, self.latr, lonc, latc)
        nc.close()

    def get_bbox_indices(self, lon, lat, bbox):
        """Gets i,j indices corresponding to specified bounding box"""

        from matplotlib import path
        mypath = np.array([bbox[[0,1,1,0]], bbox[[2,2,3,3]]]).T
        p = path.Path(mypath)
        points = np.vstack((lon.flatten(), lat.flatten())).T
        n, m = np.shape(lon)
        inside = p.contains_points(points).reshape((n, m))
        ii, jj = np.meshgrid(list(range(m)), list(range(n)))
        self.i0, self.i1, self.j0, self.j1 = min(ii[inside]) - 1, max(ii[inside]) + 2, \
            min(jj[inside]) - 1, max(jj[inside]) + 2
        ny, nx = np.shape(lon)
        self.i0, self.i1 = max(self.i0, 0), min(self.i1, nx)
        self.j0, self.j1 = max(self.j0, 0), min(self.j1, ny)

class roms_data(object):
    """Class for ROMS output data"""

    def __init__(self, grid, roms_dir, filename, dates = None, start = None,
                 single = False, get_w = False):

        if dates is None: # read single file
            self.read(grid, roms_dir, filename, start, single, get_w)
        else:
            fname = dates[0].strftime(filename) # filename interpreted as a template
            self.read(grid, roms_dir, fname, start, single = False, get_w = get_w)
            if len(dates) > 1:
                for date in dates[1:]:
                    try:
                        fname = date.strftime(filename)
                        new = roms_data(grid, roms_dir, fname, start, single = False,
                                        get_w = get_w)
                        self.append(new)
                    except:
                        continue

    def read(self, grid, roms_dir, filename, start = None, single = False, get_w = False):
        """Reads ROMS data from single file. If single is False, all time
        results are read after start (datetime), otherwise only one set of
        results for the time nearest to start (or the first set of
        results, if start is not specified).
        """

        if start is None: start = dates[0]
        timeword = 'at' if single else 'from'
        print('Reading roms data %s %s %s...' % (filename, timeword,
                                                 start.strftime('%H:%M:%S %d/%m/%Y')))
        fname = os.path.join(roms_dir, filename)
        nc = Dataset(fname,'r')
        times = nc.variables['ocean_time']
        dates = num2date(times[:], units = times.units, calendar = 'proleptic_gregorian')
        if single:
            if start is None: start = dates[0]
            nt1 = np.argmin(np.array([abs(d - start) for d in dates]))
            nt2 = nt1 + 1
        else:
            if start is None: nt1 = 0
            else: nt1 = np.searchsorted(dates, start)
            nt2 = np.size(times)
        self.date = dates[nt1: nt2]
        i0, i1, j0, j1 = grid.i0, grid.i1, grid.j0, grid.j1
        self.zeta = nc.variables['zeta'][nt1:nt2, j0:j1, i0:i1]
        if grid.rotate:
            # rotate and de-stagger velocities:
            u = nc.variables['u'][nt1:nt2, :, j0:j1, i0-1:i1)]
            v = nc.variables['v'][nt1:nt2, :, j0-1:j1, i0:i1)]
            ur = 0.5*(u[:,:,:,:-1] + u[:,:,:,1:])
            vr = 0.5*(v[:,:,:-1,:] + v[:,:,1:,:])
            self.u, self.v = geom.rot2d(ur, vr, grid.angle)
        else:
            self.u = nc.variables['u_eastward'][nt1:nt2, :, j0:j1, i0:i1]
            self.v = nc.variables['v_northward'][nt1:nt2, :, j0:j1, i0:i1]
        if get_w:
            self.w = nc.variables['w'][nt1:nt2, :, j0:j1, i0:i1]
        self.temp = nc.variables['temp'][nt1:nt2, :, j0:j1, i0:i1]
        self.salt = nc.variables['salt'][nt1:nt2, :, j0:j1, i0:i1]
        self.vtransform = nc.variables['Vtransform'][:]
        self.sc_r = nc.variables['s_rho'][:]
        self.Cs_r = nc.variables['Cs_r'][:]
        self.sc_w = nc.variables['s_w'][:]
        self.Cs_w = nc.variables['Cs_w'][:]
        self.hc = nc.variables['hc'][:]
        nc.close()

    def append(self, new, get_w = False):
        '''
        appends variables from new along axis 0 (time)
        '''
        self.date = np.append(self.date, new.date, axis=0)
        self.zeta = np.append(self.zeta, new.zeta, axis=0)
        self.u = np.append(self.u, new.u, axis=0)
        self.v = np.append(self.v, new.v, axis=0)
        if get_w: np.append(self.w, new.w, axis=0)
        self.temp = np.append(self.temp, new.temp, axis=0)
        self.salt = np.append(self.salt, new.salt, axis=0)

    def depth_point(self, zeta, h, w = False):
        """Depths for given zeta and h. If w is True, return w levels rather
        than rho."""
        sc, Cs = (self.sc_w, self.Cs_w) if w else (self.sc_r, self.Cs_r)
        N = len(sc)
        r = range(N)
        z = np.zeros(np.hstack((N, zeta.shape)))
        if self.vtransform == 1:
            for k in r:
                z0 = (sc[k] - Cs[k]) * self.hc + Cs[k] * h
                z[k,:] = z0 + zeta * (1.0 + z0 / h)
        elif self.vtransform == 2:
            for k in r:
                z0 = (self.hc * sc[k] + Cs[k] * h) / (self.hc + h)
                z[k,:] = zeta + (zeta + h) * z0
        return z

