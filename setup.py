from setuptools import setup, find_packages

setup(
    name='SurplusElement',
    version='1.0',
    packages=find_packages(),
    package_data={
        'SurplusElement': ['mathematics/integrationData.txt'],
    },
    include_package_data=True,
    install_requires=[
    'numpy',
    'scipy',
    'matplotlib',
    'scikit_tt @ git+https://github.com/PGelss/scikit_tt'],  
)