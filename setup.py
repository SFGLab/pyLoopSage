from setuptools import setup, find_namespace_packages
import re
from pathlib import Path

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

VERSION = re.search(r'"([^"]+)"', Path('src/version.py').read_text().split('__version__')[1]).group(1)

setup(
    name='pyLoopSage',  # Package name
    version=VERSION,  # read from src/version.py
    description='An energy-based stochastic model of loop extrusion in chromatin.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Sebastian Korsak',
    author_email='s.korsak@datascience.edu.pl',
    url='https://github.com/SFGLab/pyLoopSage',  # GitHub repository URL
    license='GNU General Public License v3.0',
    package_dir={'': 'src'},
    packages=find_namespace_packages('src', exclude=['*__pycache__*']),
    py_modules=['run', 'logger', 'version'],
    include_package_data=True,
    package_data={
        'html': ['*.html', '*.css'],
        'molecular_dynamics.forcefields': ['*.xml'],
        'motif_finder.data': ['*.pfm'],
    },
    install_requires=[  # List your package dependencies here
        'numpy>1.2,<2.0',
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
        'imageio[ffmpeg]',
        'pillow',
        'pyBigWig',
        'topoly',
    ],
    entry_points={
        'console_scripts': [
            'loopsage=run:main',  # the `loopsage` command runs src/run.py
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',  # General OS classifier
    ],
    python_requires='>=3.10',  # Specify Python version compatibility
)