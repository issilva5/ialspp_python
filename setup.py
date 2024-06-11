from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

import os
import subprocess
import platform

def get_eigen_include_dir():
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        try:
            eigen_path = subprocess.check_output(["brew", "--prefix", "eigen"]).strip().decode("utf-8")
            eigen_include_dir = os.path.join(eigen_path, "include/eigen3")
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Eigen not found via Homebrew. Make sure Eigen is installed.") from e
    elif system == 'Linux':
        try:
            # Trying to locate eigen in common install paths
            eigen_include_dir = "/usr/include/eigen3"
            if not os.path.exists(eigen_include_dir):
                raise RuntimeError("Eigen not found in /usr/include/eigen3. Make sure Eigen is installed.")
        except Exception as e:
            raise RuntimeError("Eigen not found via standard Linux paths. Ensure Eigen is installed.") from e
    else:
        raise RuntimeError("Unsupported operating system.")
    
    return eigen_include_dir


def get_json_include_dir():
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        try:
            nlohmann_path = subprocess.check_output(["brew", "--prefix", "nlohmann-json"]).strip().decode("utf-8")
            nlohmann_include_dir = os.path.join(nlohmann_path, "include")
        except subprocess.CalledProcessError as e:
            raise RuntimeError("nlohmann-json not found via Homebrew. Make sure nlohmann-json is installed.") from e
    elif system == 'Linux':
        raise NotImplementedError("Not implemented yet.")
    else:
        raise RuntimeError("Unsupported operating system.")
    
    return nlohmann_include_dir

# Get the Eigen path using Homebrew
eigen_include_dir = get_eigen_include_dir()
json_include_dir = get_json_include_dir()

ext_modules = [
    Extension(
        'ialspp',  # Name of the generated module
        [
            'ialspp/bindings.cc',
            # Add other source files here
        ],
        include_dirs=[pybind11.get_include(), eigen_include_dir, json_include_dir],
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
    packages=['ialspp-stubs'],
    package_data={'ialspp-stubs': ['*.pyi']},
    include_package_data=True,
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
