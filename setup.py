import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='roms2schism',
    version='0.1.0',
    description='Tools for creating SCHISM boundary forcing files from ROMS output',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://gitlab.com/acroucher/roms2schism',
    author='Ivica Janekovic, Adrian Croucher',
    author_email='a.croucher@auckland.ac.nz',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"],
    python_requires='>=3.6',
    install_requires=['progressbar2', 'munch', 'netCDF4', 'numpy', 'scipy',
                      'pyschism', 'matplotlib']
)
