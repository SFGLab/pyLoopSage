from pathlib import Path
import re

from setuptools import find_packages, setup

VERSION = re.search(r'"([^"]+)"', Path('src/loopsage/version.py').read_text().split('__version__')[1]).group(1)

setup(
    name='pyLoopSage',  # Package name
    version=VERSION,  # read from src/loopsage/version.py
    description='An energy-based stochastic model of loop extrusion in chromatin.',
    long_description=Path('README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    author='Sebastian Korsak',
    author_email='s.korsak@datascience.edu.pl',
    url='https://github.com/SFGLab/pyLoopSage',  # GitHub repository URL
    license='GNU General Public License v3.0',
    package_dir={'': 'src'},
    packages=find_packages('src', include=['loopsage', 'loopsage.*']),
    include_package_data=True,
    package_data={
        'loopsage.html': ['*.html', '*.css'],
        'loopsage.molecular_dynamics': ['forcefields/*.xml'],
        'loopsage.motif_finder': ['data/*.pfm'],
    },
    install_requires=[
        'numpy>=1.26',
        'numba',
        'scipy',
        'pandas',
        'matplotlib',
        'mdtraj',
        'seaborn',
        'scikit-learn',
        'rich',
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
        'hic-straw',  # reads .hic files (--hic)
        'topoly',
    ],
    entry_points={
        'console_scripts': [
            'loopsage=loopsage.api:cli',  # the `loopsage` command
            'loopsage-scan=loopsage.sensitivity.cli:cli',  # sensitivity scans
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',  # General OS classifier
    ],
    python_requires='>=3.10',  # Specify Python version compatibility
)
