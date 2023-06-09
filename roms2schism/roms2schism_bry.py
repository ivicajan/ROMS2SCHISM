#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from datetime import timedelta
from roms2schism import schism as sm
from roms2schism import boundary as bdy
from roms2schism import nudging as ndg

def main(dates, template, bry=False, nudge=False, dcrit = 700, schism_grid_dir = './',
         roms_dir = './', roms_grid_filename = None, lonc = 175., latc = -37.):
    # ## Actual start of the roms2schism interpolation
    
    # part with reading SCHISM mesh
    schism_grid_file = 'hgrid.ll'
    schism_vgrid_file = 'vgrid.in'
    schism = sm.schism_grid(schism_grid_file, schism_vgrid_file, schism_grid_dir, lonc, latc)
    
    if bry is True:
        print('Making bry files for SCHISM')
        bdy.make_boundary(schism, template, dates, dcrit, roms_dir,
                          roms_grid_filename, roms_grid_dir, lonc, latc)
        
    if nudge is True:
        print('Making nudging files for SCHISM')
        ndg.make_nudging(schism, template, dates, dcrit, roms_dir,
                         roms_grid_filename, roms_grid_dir, lonc, latc)
        
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
    parser.add_argument('--roms_grid_dir', default=None, help='ROMS grid directory (only needed if roms_grid_filename is not None)')
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
