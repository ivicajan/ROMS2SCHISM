Tools for creating SCHISM boundary condition forcing and hotstart files from ROMS output.

Example:

```python
from datetime import datetime, timedelta
import numpy as np
from roms2schism import schism as sm
from roms2schism import boundary as bdy
from roms2schism import nudging as ng
from roms2schism import hotstart as hs

start_date = datetime(2017, 1, 1)
ndays = 30
dates = start_date + np.arange(ndays) * timedelta(days = 1)

roms_dir = '/path/to/roms/data/'
lonc, latc = 175., -37.
dcrit = 7e3
roms_grid_filename = None

schism = sm.schism_grid(lonc = lonc, latc = latc)

template = "foo_his_%Y%m.nc"
bdy.make_boundary(schism, template, dates, dcrit, roms_dir, roms_grid_filename,
                  lonc, latc)

template = "foo_avg_%Y%m.nc"
ng.make_nudging(schism, template, dates, dcrit, roms_dir, roms_grid_filename,
                lonc, latc)

roms_data_filename = "foo_his_201701.nc"
hs.make_hotstart(schism, roms_data_filename, dcrit, roms_dir, roms_grid_filename,
                  lonc, latc)

```
