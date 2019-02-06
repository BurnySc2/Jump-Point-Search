from distutils.core import setup
from Cython.Build import cythonize
import numpy, heapq

setup(name='Hello world app',
      ext_modules=cythonize(["pathfinding.pyx"]),
      include_dirs=[numpy.get_include()])