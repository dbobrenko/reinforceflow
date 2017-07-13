import os.path
import sys
from setuptools import setup, find_packages

sys.path.append(os.path.join(os.path.dirname(__file__), 'reinforceflow'))
from version import __version__ as VERSION

deps = [l for l in open('requirements.txt').read().splitlines()]

setup(name='reinforceflow',
      version=VERSION,
      description='Reinforcement Learning framework based on TensorFlow and OpenAI Gym',
      url='https://github.com/dbobrenko/reinforceflow',
      author='Dmytro Bobrenko',
      author_email='d.bobrenko@gmail.com',
      license='MIT',
      packages=[package for package in find_packages()
                if package.startswith('reinforceflow')],
      install_requires=deps,
      zip_safe=False)
