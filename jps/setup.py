from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name='Hello world app',
      ext_modules=cythonize(["jps_nc.pyx"]), #, "jps_nc_cython.pyx"]),
      include_dirs=[numpy.get_include()])

      # ext_modules=cythonize(["cython_test.pyx", "jps.py", "jps_no_cache.pyx"]))