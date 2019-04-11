from __future__ import print_function
from setuptools import setup, Command
import os
import sys
import lsmtool._version


class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import sys,subprocess
        errno = subprocess.call([sys.executable, 'runtests.py'])
        raise SystemExit(errno)

# Handle Python 3-only dependencies
if sys.version_info < (3, 0):
    reqlist = ['numpy','astropy >= 0.4, <3.0']
else:
    reqlist = ['numpy','astropy >= 0.4']

setup(
    name='lsmtool',
    version=lsmtool._version.__version__,
    url='http://github.com/darafferty/lsmtool/',
    author='David Rafferty',
    author_email='drafferty@hs.uni-hamburg.de',
    description='The LOFAR Local Sky Model Tool',
    platforms='any',
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules',
        ],
    install_requires=reqlist,
    scripts=['bin/lsmtool'],
    packages=['lsmtool','lsmtool.operations'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
    )
