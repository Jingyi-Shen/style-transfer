from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# setup(
#     cmdclass = {'build_ext': build_ext},
#     ext_modules = [Extension("lic_internal", ["lic_internal.pyx"])]
# )

import numpy
# add include_dirs=[numpy.get_include()] in Extension by JY
setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("lic_internal", ["lic_internal.pyx"], include_dirs=[numpy.get_include()])]
)