from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
      Extension("jps_cython_helper", ["jps_cython_helper.pyx"],
      include_dirs=[numpy.get_include()]),
]

setup(name='Hello world app',
      ext_modules=cythonize(extensions),
)