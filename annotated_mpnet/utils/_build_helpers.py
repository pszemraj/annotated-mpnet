"""Build helpers for packaging."""

from __future__ import annotations

import sys

from setuptools.command.build_ext import build_ext as _build_ext


class BuildExt(_build_ext):
    """Custom build_ext to inject NumPy headers and platform flags."""

    def finalize_options(self) -> None:
        super().finalize_options()
        import numpy

        if self.include_dirs is None:
            self.include_dirs = []
        self.include_dirs.append(numpy.get_include())

    def build_extensions(self) -> None:
        if sys.platform == "darwin":
            extra_compile_args = ["-stdlib=libc++", "-O3"]
        else:
            extra_compile_args = ["-std=c++11", "-O3"]

        for ext in self.extensions:
            existing = list(ext.extra_compile_args or [])
            for arg in extra_compile_args:
                if arg not in existing:
                    existing.append(arg)
            ext.extra_compile_args = existing

        super().build_extensions()
