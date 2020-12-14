from __future__ import print_function
from setuptools import setup, Command, Extension, Distribution
from setuptools.command.build_ext import build_ext
import sys
import lsmtool._version


# Flag that determines whether to build the optional (but faster) C++
# extensions. Set to False to install only the pure Python versions.
if "--build_c_extentions" in sys.argv:
    build_c_extentions = True
    sys.argv.remove("--build_c_extentions")
else:
    build_c_extentions = False

# Handle Python 3-only dependencies
if sys.version_info < (3, 0):
    reqlist = ['numpy', 'astropy >= 0.4, <3.0']
else:
    # Require astropy v3.2 or later to get much faster copies (4.x can cause some
    # installation problems, so exclude for now)
    reqlist = ['numpy', 'astropy >= 3.2, <4.0']
if build_c_extentions:
    reqlist.append('pybind11>=2.2.0')
    ext_modules = [Extension('lsmtool.operations._grouper',
                             ['lsmtool/operations/_grouper.cpp'],
                             language='c++')]
else:
    ext_modules = []


class PyTest(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import sys
        import subprocess
        errno = subprocess.call([sys.executable, 'runtests.py'])
        raise SystemExit(errno)


class LSMToolDistribution(Distribution):

    def is_pure(self):
        if self.pure:
            return True

    def has_ext_modules(self):
        return not self.pure

    global_options = Distribution.global_options + [
        ('pure', None, "use pure Python code instead of C++ extensions")]
    pure = False


class BuildExt(build_ext):

    def build_extensions(self):
        opts = ['-std=c++11']
        if sys.platform == 'darwin':
            opts += ['-stdlib=libc++']
        else:
            # Enable OpenMP support only on non-Darwin platforms, as clang
            # does not support it without some work
            opts += ['-fopenmp']
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


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
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules',
        ],
    install_requires=reqlist,
    scripts=['bin/lsmtool'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    distclass=LSMToolDistribution,
    packages=['lsmtool', 'lsmtool.operations'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
    )
