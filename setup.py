from setuptools import setup

with open('README.md', 'r', encoding='utf8') as f:
    long_description = f.read()

__version__ = 'unknown'
exec(open('rll/version.py').read())

setup(
      name='rll',
      packages=['rll', 'tests'],
      version=__version__,
      description='Collection of different methods frequently used in the reiterlab across projects.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      install_requires=['numpy', 'matplotlib', ],
      setup_requires=['pytest-runner', 'flake8'],
      tests_require=['pytest', 'pytest-cov'],
      extras_require={'plotting': ['matplotlib']},
      url='https://github.com/reiterlab/rll',
      author='Johannes Reiter',
      author_email='johannes.reiter@stanford.edu',
      license='GNUv3',
      classifiers=[
        'Programming Language :: Python :: 3.6',
      ],
      test_suite='tests',
)
