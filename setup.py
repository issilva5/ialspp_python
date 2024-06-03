from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

import os
import subprocess

# Get the Eigen path using Homebrew
eigen_path = subprocess.check_output(["brew", "--prefix", "eigen"]).strip().decode("utf-8")
eigen_include_dir = os.path.join(eigen_path, "include/eigen3")

ext_modules = [
    Extension(
        'ialspp',  # Name of the generated module
        [
            'ialspp/bindings.cc',
            # Add other source files here
        ],
        include_dirs=[pybind11.get_include(), eigen_include_dir],
        extra_compile_args=['-std=c++11', '-O3', '-Wall'],
        language='c++'
    ),
]

setup(
    name='ialspp',
    version='0.1',
    author='Ítallo Silva',
    author_email='itallo@copin.ufcg.edu.br',
    description='A Python package for a recommendation system using pybind11',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    package_data={'stubs': ['*.pyi']},
    install_requires=[
        'numpy==1.26.4',
        'pandas==2.2.2',
        'pybind11==2.12.0',
        'python-dateutil==2.9.0.post0',
        'pytz==2024.1',
        'setuptools==70.0.0',
        'six==1.16.0',
        'tzdata==2024.1'
    ],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
