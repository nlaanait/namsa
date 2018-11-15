from setuptools import setup, find_packages

setup(
    name='namsa',
    version='0.2',
    packages=['namsa', 'tests'],
    url='',
    license='',
    author='Numan Laanait',
    author_email='laanaitn@ornl.gov',
    description='',
    install_requires=['scipy', 'pymatgen', 'numpy', 'pycuda', 'scikit-cuda'],
    #install_requires=['numpy', 'scipy', 'pymatgen', 'pybtex'],
    test_suite='tests',
    python_requires='>=3.6',
    package_dir = {'namsa': 'namsa'},
    package_data = {'namsa': 'scattering_database/*.npy'},
    include_package_data=True
)
