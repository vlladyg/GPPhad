from setuptools import find_packages, setup
import sys

if sys.version_info < (3, 5, 0, 'final', 0):
    raise SystemExit('Python 3.5 or later is required!')

setup(
    name='GPPhad',
    packages=find_packages(include=['GPPhad', 'GPPhad.GP', 'GPPhad.GP.phase_diagram', 'GPPhad.two_stages']),
    version='1.1',
    description='Phase diagrams calculation via Gaussian process',
    author='Vladimir Ladygin, Alexander Shapeev',
    license='MIT',
    install_requires=['numpy','scipy','pandas', 'gmpy2', 'dill']
)
