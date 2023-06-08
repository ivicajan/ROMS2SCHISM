#!/usr/bin/env python
# coding: utf-8

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
