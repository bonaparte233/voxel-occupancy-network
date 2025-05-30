#!/usr/bin/env python3
"""
Setup script for Voxel Occupancy Network standalone module.

Install with: pip install -e .
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Standalone implementation of voxel-conditioned occupancy networks"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="voxelOccupancyNetwork",
    version="1.0.0",
    author="Extracted from Occupancy Networks by Mescheder et al.",
    author_email="",
    description="Standalone implementation of voxel-conditioned occupancy networks",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/autonomousvision/occupancy_networks",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "binvox": ["binvox-rw>=1.0.0"],
        "visualization": ["open3d>=0.15.0", "plotly>=5.0.0"],
        "dev": ["pytest>=6.0.0", "black>=21.0.0", "flake8>=4.0.0"],
    },
    entry_points={
        "console_scripts": [
            "voxel-to-mesh=voxelOccupancyNetwork.examples.load_pretrained:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
