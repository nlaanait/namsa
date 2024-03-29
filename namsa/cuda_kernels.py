from pycuda.tools import dtype_to_ctype
from pycuda.compiler import SourceModule, DynamicSourceModule
from jinja2 import Template
import numpy as np
import namsa
import os

class ProbeKernels(object):
    def __init__(self, cu_file=None, sampling=np.array([512, 512])):
        if cu_file is None:
            cu_file = os.path.join(namsa.__path__[0], 'probe_kernels.cu')
        with open(cu_file, 'r') as f:
            cuda_kerns = Template(f.read())
        self.kernels = dict()
        self.x_sampling = np.int32(sampling[1])
        self.y_sampling = np.int32(sampling[0])
        kernels = cuda_kerns.render(type=dtype_to_ctype(np.complex64), y_sampling=self.y_sampling,
                                    x_sampling=self.x_sampling)
        src = SourceModule(kernels, cache_dir=False)
        self.kernels['hard_aperture'] = src.get_function('hard_aperture')
        self.kernels['soft_aperture'] = src.get_function('soft_aperture')
        self.kernels['spherical_phase_error'] = src.get_function('spherical_phase_error')
        self.kernels['mult_wise_c2d_re2d'] = src.get_function('mult_wise_c2d_re2d')
        self.kernels['fftshift_2d'] = src.get_function('fftshift_2d')
        self.kernels['fftshift_2d_stack'] = src.get_function('fftshift_2d_stack')
        self.kernels['k_grid'] = src.get_function('k_grid')
        self.kernels['probes_stack'] = src.get_function('build_probes_stack')
        self.kernels['mult_wise_c3d_c2d'] = src.get_function('mult_wise_c3d_c2d')
        self.kernels['mult_wise_c3d_c3d_ind'] = src.get_function('mult_wise_c3d_c3d_ind')
        self.kernels['propagator'] = src.get_function('propagator')
        self.kernels['norm_stack'] = src.get_function('norm_const_stack')
        self.kernels['norm'] = src.get_function('norm_const')
        self.kernels['normalize_stack'] = src.get_function('normalize_stack')
        self.kernels['normalize'] = src.get_function('normalize')
        self.kernels['mod_square_stack'] = src.get_function('mod_square_stack')

class PotentialKernels:
    def __init__(self, cu_file=None, sampling=np.array([512,512]),
    potential_shape=np.array([80,80]), num_slices=100, sites_size=1000):
        if cu_file is None:
            cu_file = os.path.join(namsa.__path__[0], 'potential_kernels.cu')
        with open(cu_file, 'r') as f:
            cuda_kerns = Template(f.read())
        self.kernels = dict()
        self.x_sampling = np.int32(sampling[1])
        self.y_sampling = np.int32(sampling[0])
        self.pot_shape_x = np.int32(potential_shape[1])
        self.pot_shape_y = np.int32(potential_shape[0])
        self.num_slices = np.int32(num_slices)
        self.sites_size = np.int32(sites_size)
        kernels = cuda_kerns.render(y_sampling= self.y_sampling, x_sampling=self.x_sampling,
                               pot_shape_y=self.pot_shape_y, pot_shape_x=self.pot_shape_x,
                               sites_size=self.sites_size, num_slices=self.num_slices)
                               # max_size=self.sites_size)
        src = SourceModule(kernels, cache_dir=False)
        self.kernels['build_potential'] = src.get_function('BuildScatteringPotential')
        # self.kernels['build_potential'] = src.get_function('makePotSlice')
