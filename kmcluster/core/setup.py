from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

ext_modules = [
    Extension(
        "cython_helper", ["cython_helper.pyx"], include_dirs=[numpy.get_include()]
    )
]
setup(ext_modules=cythonize("cython_helper.pyx"), include_dirs=[numpy.get_include()])


# import numpy
# import pyximport
# pyximport.install(setup_args={"include_dirs":numpy.get_include()},reload_support=True)
# import cython_helper"

# from cython_npm.cythoncompile import export
# export('cython_helper.pyx', 'cython_helper')
# import cython_helper
""""
from cython_npm.cythoncompile import install
Manymodules = [
    'cython_helper.pyx'
]
install(Manymodules)"""
