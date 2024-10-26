from setuptools import setup, find_packages
from pathlib import Path

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyLoopSage',  # Package name
    version='0.0.2',  # Version of the software
    description='An energy-based stochastic model of loop extrusion in chromatin.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Sebastian Korsak',
    author_email='s.korsak@datascience.edu.pl',
    url='https://github.com/SFGLab/pyLoopSage',  # GitHub repository URL
    license='GNU General Public License v3.0',
    packages=find_packages(),  # Automatically finds all packages and sub-packages
    install_requires=[  # List your package dependencies here
        'numpy',
        'numba',
        'scipy',
        'pandas',
        'argparse',
        'matplotlib',
        'mdtraj',
        'seaborn',
        'scikit-learn',
        'configparser',
        'typing-extensions',
        'tqdm',
        'pyvista[all]',
        'OpenMM',
        'OpenMM-cuda',
        'statsmodels',
        'imageio',
        'pillow',
    ],
    entry_points={
        'console_scripts': [
            'loopsage=LoopSage.run:main',  # MultiMM command points to run.py's main function
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',  # General OS classifier
    ],
    python_requires='>=3.10',  # Specify Python version compatibility
)