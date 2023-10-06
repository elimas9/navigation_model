import setuptools
import site
import sys
from glob import glob
from pybind11.setup_helpers import Pybind11Extension, build_ext
from urllib.request import urlretrieve
import shutil
import os
import tarfile

site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

class build_ext_with_eigen(build_ext):

    def run(self):
        urlretrieve("https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz", "eigen.tar.gz")
        with tarfile.open("eigen.tar.gz") as eigen_sources:
            eigen_sources.extractall()
        self.extensions[0].include_dirs.append("eigen-3.4.0")
        build_ext.run(self)
        # remove eigen folder and sources
        shutil.rmtree("eigen-3.4.0")
        os.remove("eigen.tar.gz")


ext_modules = [
    Pybind11Extension(
        "_navigation_model",
        sorted(glob("src/_navigation_model/*.cpp")),  # Sort source files for reproducibility,
        include_dirs=["eigen-3.4.0"]
    ),
]

setuptools.setup(
    cmdclass={"build_ext": build_ext_with_eigen},
    ext_modules=ext_modules
)
