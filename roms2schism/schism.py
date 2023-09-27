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
from roms2schism.geometry import transform_ll_to_cpp, bbox

class schism_vgrid:
    def __init__(self):
        pass

    def read_vgrid(self,fname):
        #read schism vgrid
        fid=open(fname,'r'); lines=fid.readlines(); fid.close()

        self.ivcor=int(lines[0].strip().split()[0]); self.nvrt=int(lines[1].strip().split()[0])
        if self.ivcor==1:
            #read vgrid info
            lines=lines[2:]; sline=np.array(lines[0].split()).astype('float')
            if sline.min()<0: #old format
               self.kbp=np.array([int(i.split()[1])-1 for i in lines]); self.np=len(self.kbp)
               self.sigma=-np.ones([self.np,self.nvrt])
               for i,line in enumerate(lines):
                   self.sigma[i,self.kbp[i]:]=np.array(line.strip().split()[2:]).astype('float')
            else:
              sline=sline.astype('int'); self.kbp=sline-1; self.np=len(sline)
              self.sigma=np.array([i.split()[1:] for i in lines[1:]]).T.astype('float')
              fpm=self.sigma<-1; self.sigma[fpm]=-1
        elif self.ivcor==2:
            self.kz, self.h_s = lines[1].strip().split()[1:3]; self.kz=int(self.kz); self.h_s=float(self.h_s)

            #read z grid
            self.ztot=[]; irec=2
            for i in np.arange(self.kz):
                irec=irec+1
                self.ztot.append(lines[irec].strip().split()[1])
            self.ztot=np.array(self.ztot).astype('float')

            #read s grid
            self.sigma=[]; irec=irec+2
            self.nsig=self.nvrt-self.kz+1
            self.h_c, self.theta_b, self.theta_f = np.array(lines[irec].strip().split()[:3]).astype('float')
            for i in np.arange(self.nsig):
                irec=irec+1
                self.sigma.append(lines[irec].strip().split()[1])
            self.sigma=np.array(self.sigma).astype('float')
        return self.sigma

    def compute_zcor(self, dp, eta=0, fmt=0, method=0, sigma=None, kbp=None, ifix=0):
        '''
        compute schism zcor (ivcor=1)
            dp:  depth at nodes (dim=[np] or [1])
            eta: surface elevation (dim=[np] or [1])
            fmt: output format of zcor
                 fmt=0: bottom depths byeond kbp are extended
                 fmt=1: bottom depths byeond kbp are nan
            method=1 and ivcor=1: used for computing zcor for subset of nodes (need sigma,kbp)
            method=1 and ivcor=2: return zcor and kbp
            ifix=1 and ivcor=2: using traditional sigma in shallow if error raise
        '''
        if self.ivcor==1:
           if method==0: return compute_zcor(self.sigma,dp,eta=eta,fmt=fmt,kbp=self.kbp)
           if method==1: return compute_zcor(sigma,dp,eta=eta,fmt=fmt,kbp=kbp)
        elif self.ivcor==2:
           zcor,kbp=compute_zcor(self.sigma,dp,eta=eta,fmt=fmt,ivcor=2,vd=self,method=1,ifix=ifix)
           if method==0: return zcor
           if method==1: return [zcor,kbp]

def read_schism_vgrid(fname):
    '''
    read schism vgrid information
    '''
    vd=schism_vgrid(); vd.read_vgrid(fname)
    return vd

def compute_zcor(sigma,dp,eta=0,fmt=0,kbp=None,ivcor=1,vd=None,method=0,ifix=0):
    '''
    compute schism zcor (ivcor=1)
        sigma: sigma cooridinate (dim=[np,nvrt])
        dp: depth at nodes (dim=[np] or [1])
        eta: surface elevation (dim=[np] or [1])
        fmt: output format of zcor
            fmt=0: bottom depths byeond kbp are extended
            fmt=1: bottom depths byeond kbp are nan
        kbp: index of bottom layer (not necessary, just to speed up if provided for ivcor=1)
        method=1 and ivcor=2: return zcor and kbp
        ifix=1 and ivcor=2: using traditional sigma in shallow if error raise
    '''

    if ivcor==1:
        npp=sigma.shape[0]
        if not hasattr(dp,'__len__'):  dp=np.ones(npp)*dp
        if not hasattr(eta,'__len__'): eta=np.ones(npp)*eta

        #get kbp
        if kbp is None:
            kbp=np.array([nonzero(abs(i+1)<1e-10)[0][-1] for i in sigma])

        #thickness of water column
        hw=dp+eta

        #add elevation
        zcor=hw[:,None]*sigma+eta[:,None]
        fpz=hw<0; zcor[fpz]=-dp[fpz][:,None]

        #change format
        if fmt==1:
            for i in np.arange(npp):
                zcor[i,:kbp[i]]=nan
        return zcor
    elif ivcor==2:
        #get dimension of pts
        if not hasattr(dp,'__len__'):
            npp=1; dp=np.array([dp])
        else:
            npp=len(dp)
        if not hasattr(eta,'__len__'): eta=np.ones(npp)*eta
        zcor=np.ones([vd.nvrt,npp])*np.nan

        cs=(1-vd.theta_b)*np.sinh(vd.theta_f*vd.sigma)/np.sinh(vd.theta_f)+ \
            vd.theta_b*(np.tanh(vd.theta_f*(vd.sigma+0.5))-np.tanh(vd.theta_f*0.5))/2/np.tanh(vd.theta_f*0.5)
        #for sigma layer: depth<=h_c
        hmod=dp.copy(); fp=hmod>vd.h_s; hmod[fp]=vd.h_s
        fps=hmod<=vd.h_c
        zcor[(vd.kz-1):,fps]=vd.sigma[:,None]*(hmod[fps][None,:]+eta[fps][None,:])+eta[fps][None,:]

        #depth>h_c
        fpc=eta<=(-vd.h_c-(hmod-vd.h_c)*vd.theta_f/np.sinh(vd.theta_f))
        if sum(fpc)>0:
            if ifix==0: sys.exit('Pls choose a larger h_c: {}'.format(vd.h_c))
            if ifix==1: zcor[(vd.kz-1):,~fps]=eta[~fps][None,:]+(eta[~fps][None,:]+hmod[~fps][None,:])*vd.sigma[:,None]
        else:
            zcor[(vd.kz-1):,~fps]=eta[~fps][None,:]*(1+vd.sigma[:,None])+vd.h_c*vd.sigma[:,None]+cs[:,None]*(hmod[~fps]-vd.h_c)

        #for z layer
        kbp=-np.ones(npp).astype('int'); kbp[dp<=vd.h_s]=vd.kz-1
        fpz=dp>vd.h_s; sind=np.nonzero(fpz)[0]
        for i in sind:
            for k in np.arange(0,vd.kz-1):
                if (-dp[i]>=vd.ztot[k])*(-dp[i]<=vd.ztot[k+1]):
                    kbp[i]=k;
                    break
            #check
            if kbp[i]==-1:
                sys.exit('can not find a bottom level for node')
            elif kbp[i]<0 or kbp[i]>=(vd.kz-1):
                sys.exit('impossible kbp,kz: {}, {}'.format(kbp[i],vd.kz))

            #assign values
            zcor[kbp[i],i]=-dp[i]
            for k in np.arange(kbp[i]+1,vd.kz-1):
                zcor[k,i]=vd.ztot[k]
        zcor=zcor.T; vd.kbp=kbp

        #change format
        if fmt==0:
            for i in np.arange(npp):
                zcor[i,:kbp[i]]=zcor[i,kbp[i]]
        if method==0: return zcor
        if method==1: return [zcor,kbp]


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
        vd=read_schism_vgrid(schism_vgrid_file)
        zcor=vd.compute_zcor(-hgrid.values)
        nvrt = zcor.shape[1] 
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
