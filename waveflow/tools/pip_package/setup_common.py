from setuptools import setup, find_packages
import os, glob


def setup_waveflow(name, install_requires):
  """
  Setups waveflow PIP package.

  This method defines common setup code for all waveflow pip packages.

  :param name: name of the package
  :param install_requires: package additional dependencies
  """
  ops_so = ['core/' + os.path.basename(x) for x in
            glob.glob('waveflow/core/*.so')]
  setup(
    name=name,
    version='0.1rc1',
    description='Signal processing software for tensorflow',
    url='http://github.com/waveflow-team/waveflow',
    author='Waveflow Team',
    author_email="waveflow-team@googlegroups.com",
    license='Apache License 2.0',
    classifiers=[
      'Development Status :: 1 - Planning',

      'Intended Audience :: Developers',
      'Intended Audience :: Science/Research',

      'License :: OSI Approved :: Apache Software License',

      'Programming Language :: Python :: 2',
      'Programming Language :: Python :: 2.7',
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 3.2',
      'Programming Language :: Python :: 3.3',
      'Programming Language :: Python :: 3.4',
      'Programming Language :: Python for :: 3.5',
      'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(),
    install_requires=[
      'scipy>=1.0.0',
    ] + install_requires,
    package_data={
      'waveflow': ops_so
    }
  )
