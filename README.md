# ROMS2SCHISM
A Python package for creating [SCHISM](http://ccrm.vims.edu/schismweb/) boundary condition forcing and hotstart files from [ROMS](https://www.myroms.org/) output.

The `roms2schism` module contains four main sub-modules:

* `schism`: for loading the SCHISM grid (e.g. `hgrid.ll`, `vgrid.in`)
* `boundary`: for creating "type 4" SCHISM boundary conditions (`uv3D.th.nc` etc. files) from ROMS output 
* `nudging`: for creating temperature and salinity nudging files for the open boundary (`TEM_nu.nc` and `SAL_nu.nc`) from ROMS output 
* `hotstart`: for creating hotstart initial conditions (`hotstart.nc`) from ROMS output

## Loading the SCHISM grid

The SCHISM grid can be loaded like this:

```python
import roms2schism as r2s
schism = r2s.schism.schism_grid(schism_grid_file, schism_vgrid_file, schism_grid_dir, lonc, latc)
```

The defaults for the parameters are:

* `schism_grid_file`: `hgrid.ll`
* `schism_vgrid_file`: `vgrid.in`
* `schism_grid_dir`: `./`
* `iob`: `0`
* `lonc`, `latc`: `None`

The `iob` is list of open boundary segments you want to include for boundary conditions (default is 0 which is the first boundary only, but can be for example iobn = [0, 1, 2] for first 3 boundaries.  `lonc`, `latc` parameters define a reference longitude and latitude, used to convert the horizontal grid coordinates (assumed to be longitude, latitude) to an approximate projected coordinate system. If these are not specified, default values are calculated by averaging the grid vertex coordinates.

## Creating boundary conditions

Boundary conditions can be created like this:

```python
r2s.boundary.make_boundary(schism, template, dates, start, end, roms_dir,
                           roms_grid_filename, roms_grid_dir, dcrit)
```

The `schism` parameter is the SCHISM grid created above.

The `template` and `dates` parameters are used to define the sequence of filenames for the ROMS files used to create the boundary conditions. The `dates` parameter is an array of `datetime` objects. These are processed using the `template` string to create the filenames. All the ROMS date files must be in the directory specified by `roms_dir`.

The `start` and `end` parameters define the start and end `datetimes` of the simulation. Boundary conditions are created from ROMS results between these datetimes. They both default to `None`, in which case they are set equal to the first and last datetimes in the ROMS results.

The `roms_grid_filename` and `roms_grid_dir` parameters can be used to specify a ROMS grid file. This may be needed if the ROMS results do not contain the `u_eastward` and `v_northward` fields, in which case they are reconstructed (rotated and de-staggered) from the `u` and `v` fields. Otherwise, these parameters do not need to be specified.

The `dcrit` parameter specifies a critical distance used to avoid interpolating ROMS results over land. It should generally be slightly larger than the ROMS grid resolution. If SCHISM grid points are further than this distance from the nearest ROMS grid point, nearest neighbour interpolation is used instead of bilinear interpolation.

## Creating nudging files

Nudging files for temperature and salinity can be created like this:

```python
r2s.nudging.make_nudging(schism, template, dates, start, end, roms_dir,
                         roms_grid_filename, roms_grid_dir, dcrit)
```

The parameters for nudging are the same as those for creating the boundary conditions. The only difference is that typically nudging files are created from time-averaged ROMS results (e.g. daily averages), so the `template` parameter will be different.

## Creating hotstart initial conditions

Hotstart files can be created like this:

```python
r2s.hotstart.make_hotstart(schism, roms_data_filename, start, roms_dir,
                           roms_grid_filename, roms_grid_dir, dcrit, h0)
```

For the initial conditions, only one ROMS data file is needed, specified via the `roms_data_filename` parameter. Most of the other parameters are the same as for creating boundary conditions and nudging. The `h0` parameter (default 0.01 m) specifies the water depth below which SCHISM nodes will be considered dry at the start of the run.

## Example

```python
from datetime import datetime, timedelta
import numpy as np
import roms2schism as r2s

# set up dates corresponding to ROMS files to be read:
start_filedate = datetime(2017, 1, 1)
ndays = 30
dates = start_filedate + np.arange(ndays) * timedelta(days = 1)
start_date = datetime(2017, 1, 12)

roms_dir = '/path/to/roms/data/'
dcrit = 7e3 # should be slightly larger than ROMS grid resolution

# read SCHISM grid and only the first open boundary segment (in case you want other segments specify iob = [0, 1, 2] for example and first 3 segments):
schism = r2s.schism.schism_grid()

# create boundary forcing files:
template = "foo_his_%Y%m%d.nc"
r2s.boundary.make_boundary(schism, template, dates, start_date,
                           roms_dir = roms_dir, dcrit = dcrit)

# create boundary nudging files for T, S:
template = "foo_avg_%Y%m%d.nc"
r2s.nudging.make_nudging(schism, template, dates, start_date,
                         roms_dir = roms_dir, dcrit = dcrit)

# create hotstart.nc file:
roms_data_filename = "foo_his_20170101.nc"
r2s.hotstart.make_hotstart(schism, roms_data_filename, start_date,
                           roms_dir = roms_dir, dcrit = dcrit)

```
## Installation

ROMS2SCHISM can be installed from PyPI via `pip`:

```
pip install roms2schism
```
The latest version with all updates and features can be installed directly from Github repo using `pip` as:

```
pip install git+https://github.com/ivicajan/ROMS2SCHISM.git
```
