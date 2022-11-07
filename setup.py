import setuptools
import site
import sys

site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="navigation-model",
    version="0.0.1",
    author="Elisa Massi",
    author_email="elisa.massi9@gmail.com",
    description="A framework to model navigation behaviour from data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/elimas9/mavigation_model",
    project_urls={
        "Bug Tracker": "https://github.com/elimas9/navigation_model/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ]
)
