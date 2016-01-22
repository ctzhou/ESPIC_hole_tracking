# setup.py
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules=[
    Extension("infMagSim_cython",
              sources=["infMagSim_cython.pyx"],
              extra_objects=["infMagSim_c.o"],
              include_dirs=["/usr/include/gsl/"],
              libraries=["gsl","gslcblas",]) # Unix-like specific
]
setup(
    name = "infMagSim",
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules
)
