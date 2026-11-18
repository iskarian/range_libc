import glob
import os
import platform
import shutil
import sys
from os.path import join as pjoin

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Obtain the numpy include directory.  This logic works across numpy versions.
numpy_include = numpy.get_include()

# CUDA is on by default. Set WITH_CUDA=OFF to build the CPU-only module.
use_cuda = os.environ.get("WITH_CUDA", "ON").lower() == "on"
trace = os.environ.get("TRACE", "OFF").lower() == "on"

# support for compiling in clang
if platform.system().lower() == "darwin":
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = platform.mac_ver()[0]
    os.environ["CC"] = "c++"

CUDA_ARCH = os.environ.get("RANGE_LIBC_CUDA_ARCH", "sm_87")  # Jetson Orin
CHUNK_SIZE = "262144"
NUM_THREADS = "256"


def find_nvcc():
    """nvcc from CUDAHOME if set, otherwise from PATH (the pixi build env)."""
    if "CUDAHOME" in os.environ:
        candidate = pjoin(os.environ["CUDAHOME"], "bin", "nvcc")
        if os.path.exists(candidate):
            return candidate
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise EnvironmentError("nvcc not found: add it to PATH or set CUDAHOME")
    return nvcc


def cuda_dirs():
    """CUDA include and library dirs for a conda-style prefix.

    conda-forge CUDA packages put headers and libraries under
    $PREFIX/targets/<arch>/ as well as $PREFIX/include and $PREFIX/lib.
    """
    prefix = sys.prefix
    include = [pjoin(prefix, "include")] + glob.glob(pjoin(prefix, "targets", "*", "include"))
    lib = [pjoin(prefix, "lib")] + glob.glob(pjoin(prefix, "targets", "*", "lib"))
    include = [d for d in include if os.path.isdir(d)]
    lib = [d for d in lib if os.path.isdir(d)]
    if not any(os.path.exists(pjoin(d, "cuda_runtime.h")) for d in include):
        raise EnvironmentError("cuda_runtime.h not found under %s" % prefix)
    return include, lib


##################### Configuration ############################

compiler_flags = ["-w", "-std=c++11", "-ffast-math", "-fno-math-errno", "-O3"]
nvcc_flags = ["-arch=" + CUDA_ARCH, "-std=c++11", "-O3", "-w",
              "--compiler-options", "-fPIC"]
include_dirs = ["../", numpy_include]
depends = ["../includes/*.h"]
sources = ["RangeLibc.pyx", "../vendor/lodepng/lodepng.cpp"]
define_macros = []
library_dirs = []
libraries = []

if use_cuda:
    define_macros += [("USE_CUDA", "1"), ("CHUNK_SIZE", CHUNK_SIZE), ("NUM_THREADS", NUM_THREADS)]
    cuda_include, cuda_lib = cuda_dirs()
    include_dirs += cuda_include
    library_dirs += cuda_lib
    libraries.append("cudart")

if trace:
    define_macros.append(("_MAKE_TRACE_MAP", "1"))


class build_ext_cuda(build_ext):
    """Compile includes/kernels.cu with nvcc and link it into the module."""

    def build_extensions(self):
        if use_cuda:
            nvcc = find_nvcc()
            for ext in self.extensions:
                cu_sources = [s for s in ext.sources if s.endswith(".cu")]
                ext.sources = [s for s in ext.sources if not s.endswith(".cu")]
                for src in cu_sources:
                    obj = pjoin(self.build_temp, os.path.splitext(src)[0] + ".o")
                    os.makedirs(os.path.dirname(obj), exist_ok=True)
                    cmd = [nvcc, "-c", src, "-o", obj] + nvcc_flags
                    cmd += ["-D%s=%s" % m for m in define_macros]
                    cmd += ["-I" + d for d in ext.include_dirs]
                    self.compiler.spawn(cmd)
                    ext.extra_objects = list(ext.extra_objects or []) + [obj]
        super().build_extensions()


if use_cuda:
    sources.append("../includes/kernels.cu")

ext = Extension(
    "range_libc",
    sources,
    extra_compile_args=compiler_flags,
    extra_link_args=["-std=c++11"],
    include_dirs=include_dirs,
    define_macros=define_macros,
    library_dirs=library_dirs,
    libraries=libraries,
    depends=depends,
    language="c++",
)

setup(
    name="range_libc",
    author="Corey Walsh",
    version="0.1",
    ext_modules=cythonize([ext], language_level=3),
    cmdclass={"build_ext": build_ext_cuda},
)
