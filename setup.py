import os.path
import sys
from setuptools import setup, find_packages

sys.path.append(os.path.join(os.path.dirname(__file__), 'reinforceflow'))
from reinforceflow.version import version


install_requires = [
    'numpy',
    'gym>=0.9.1',
    'scikit-image',
    'six',
    'matplotlib',
    'opencv-python'
]

extras_require = {
    'tf': ['tensorflow>=1.0.1'],
    'tf-gpu': ['tensorflow-gpu>=1.0.1'],
    'universe': ['universe>=0.21.3']
}

setup(name='reinforceflow',
      version=version,
      description='Reinforcement Learning framework based on TensorFlow and OpenAI Gym',
      url='https://github.com/dbobrenko/reinforceflow',
      author='Dmytro Bobrenko',
      author_email='d.bobrenko@gmail.com',
      license='MIT',
      packages=[package for package in find_packages()
                if package.startswith('reinforceflow')],
      install_requires=install_requires,
      extras_require=extras_require,
      zip_safe=False)
