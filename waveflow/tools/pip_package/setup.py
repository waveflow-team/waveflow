from setuptools import setup, find_packages
import os
import glob

ops_so = ['core/' + os.path.basename(x) for x in glob.glob('waveflow/core/*.so')]

setup(name='waveflow',
      version='0.1rc1',
      description='Signal processing with tensorflow',
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
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
      ],
      packages=find_packages(),
      install_requires=['tensorflow>=1.6.0'],
      package_data={
        'waveflow': ops_so
      }
)
