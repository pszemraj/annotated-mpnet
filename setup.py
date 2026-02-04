#!/usr/bin/env python3

from pathlib import Path
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

if sys.platform == "darwin":
    extra_compile_args = ["-stdlib=libc++", "-O3"]
else:
    extra_compile_args = ["-std=c++11", "-O3"]

try:
    from Cython.Build import cythonize

    USE_CYTHON = True
except ImportError:
    cythonize = None
    USE_CYTHON = False


class BuildExt(_build_ext):
    def finalize_options(self) -> None:
        super().finalize_options()
        import numpy

        if self.include_dirs is None:
            self.include_dirs = []
        self.include_dirs.append(numpy.get_include())


source = Path("annotated_mpnet/utils/perm_utils_fast.pyx")
sources = [str(source if USE_CYTHON else source.with_suffix(".cpp"))]

extensions = [
    Extension(
        "annotated_mpnet.utils.perm_utils_fast",
        sources=sources,
        language="c++",
        extra_compile_args=extra_compile_args,
    )
]

if USE_CYTHON and cythonize is not None:
    ext_modules = cythonize(extensions, compiler_directives={"language_level": "3"})
else:
    ext_modules = extensions

setup(ext_modules=ext_modules, cmdclass={"build_ext": BuildExt})
