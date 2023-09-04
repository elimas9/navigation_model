import setuptools
import site
import sys
from glob import glob
from pybind11.setup_helpers import Pybind11Extension, build_ext

site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

ext_modules = [
    Pybind11Extension(
        "_navigation_model",
        sorted(glob("src/_navigation_model/*.cpp")),  # Sort source files for reproducibility,
        include_dirs=["/usr/include/eigen3"]
    ),
]

setuptools.setup(
    name="navigation-model",
    version="0.2.0",
    author="Elisa Massi",
    author_email="elymas93@gmail.com",
    description="A python library for analyzing, modelling, and simulating the spatial behavior and learning of rodents. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/elimas9/mavigation_model",
    project_urls={
        "Bug Tracker": "https://github.com/elimas9/navigation_model/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    license_files=("LICENSE",),
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ],
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules
)
