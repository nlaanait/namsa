# ``NAMSA`` 
(**N**ot **A**ny **M**ulti-**S**lice **A**lgorithm) 

Python Project for multi-slice algorithm based simulations of dynamical electron diffraction/microscopy.  
The computation heavy steps are implemented in CUDA C/C++ and just-in-time compiled into cuda kernels for maximal performance.  
Distribution of the Multi-Slice simulations across one material or multiple materials is performed using MPI.  

``Namsa`` __has been tested and benchmarked on up to 3000 Nvidia V100 GPUs__

# Major Dependencies
- __pymatgen__: Python package from materials genome project. Interfacing with pymatgen is done throughout.
- __pycuda__
- __scikit-cuda__
- __mpi4py__

# Install & Test
```
python setup.py install
python setup.py test
python -m unittest tests/*.py
```
