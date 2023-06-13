import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='roms2schism',
    version='0.8.0',
    description='A Python package for creating SCHISM boundary condition forcing and hotstart files from ROMS output',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://gitlab.com/acroucher/roms2schism',
    author='Ivica Janekovic, Adrian Croucher',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"],
    python_requires='>=3.6',
    install_requires=['progressbar2', 'netCDF4', 'numpy', 'scipy',
                      'pyschism', 'matplotlib']
)
