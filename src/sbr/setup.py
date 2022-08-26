setup(
    ...
    test_suite='nose.collector',
    tests_require=['nose'],
)

from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='sbr',
      version='1.0.0',
      description='Machine learning tools for public biomolecular datasets, from Saboredge',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GPL-3.0-only License',
        'Programming Language :: Python :: 2.9.13',
        'Topic :: Text Processing :: Linguistic',
      ],
      keywords='gtex rna gene expression AI',
      url='http://github.com/saboredge/sbr',
      author='Kimberly Robasky',
      author_email='kimberly.robasky@gmail.com',
      license='GPL-3.0',
      packages=['sbr'],
      install_requires=[
          'markdown',
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      #entry_points={
      #    'console_scripts': ['funniest-joke=funniest.command_line:main'],
      #},
      include_package_data=True,
      zip_safe=False)

