from pycuda.tools import dtype_to_ctype
# import pycuda.autoinit
from pycuda.compiler import SourceModule
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
        src = SourceModule(kernels)
        self.kernels['hard_aperture'] = src.get_function('hard_aperture')
        self.kernels['soft_aperture'] = src.get_function('soft_aperture')
        self.kernels['spherical_phase_error'] = src.get_function('spherical_phase_error')
        self.kernels['mult_wise_c2d_re2d'] = src.get_function('mult_wise_c2d_re2d')
        self.kernels['fftshift_2d'] = src.get_function('fftshift_2d')
        self.kernels['fftshift_2d_stack'] = src.get_function('fftshift_2d_stack')
        self.kernels['k_grid'] = src.get_function('k_grid')
        self.kernels['probes_stack'] = src.get_function('build_probes_stack')
        self.kernels['mult_wise_c3d_c2d'] = src.get_function('mult_wise_c3d_c2d')
        self.kernels['propagator'] = src.get_function('propagator')
        self.kernels['norm_const'] = src.get_function('norm_const')
        self.kernels['normalize'] = src.get_function('normalize')
        self.kernels['mod_square_stack'] = src.get_function('mod_square_stack')
