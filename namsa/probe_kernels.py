import pycuda.driver as cuda
from pycuda.tools import dtype_to_ctype
import pycuda.autoinit
from pycuda.compiler import SourceModule
from jinja2 import Template
import numpy as np

cuda_kerns = Template("""
    // WITH
    #include <pycuda-complex.hpp>
    #include <stdio.h>

    __device__ double calc_krad(float k_max, int size_x, int size_y, int col_idx, int row_idx){
        double kx = double(col_idx) * double(k_max)/double(size_x - 1) - double(k_max)/2. ;
        double ky = double(row_idx) * double(k_max)/double(size_y - 1) - double(k_max)/2. ;
        double k_rad = sqrt(kx * kx + ky * ky);
        return k_rad;
    }

    __device__ float phase_shift(float k_max, int size_x, int size_y, int size_z,
    int col_idx, int row_idx, int stk_idx, float step, float range, float x_dim, float y_dim){
        const double pi = acos(-1.0);
        float kx = float(col_idx) * k_max/float(size_x - 1) - k_max/2.;
        float ky = float(row_idx) * k_max/float(size_y - 1) - k_max/2.;
        float x_step = step;
        float y_step = step;
        float x_end = range * x_dim ;
        float y_end = range * y_dim ;
        if (x_step == 0.0 && y_step == 0.0)
        {
            return 0.0;
        }
        int ry_idx = rintf(floor(stk_idx * 1.f / sqrtf(size_z)));
        int rx_idx = stk_idx - ry_idx;
        float ry = ry_idx * y_step - y_end/2;
        float rx = rx_idx * sqrtf(size_z)) * x_step - x_end/2;
        //float ry = stk_idx * y_step - y_end/2;
        //float rx = stk_idx * x_step - x_end/2;
        float kr =  - kx * rx - ky * ry;
        return kr;
    }

    __global__ void k_grid(double *arr, double k_max_x, double k_max_y, int size_x, int size_y){
        int row_idx =  blockDim.y * blockIdx.y + threadIdx.y;
        int col_idx =  blockDim.x * blockIdx.x + threadIdx.x;
        int idx = row_idx * size_x  + col_idx;
        double k_rad = calc_krad(k_max_x, k_max_y, size_x, size_y, col_idx, row_idx);
        if (idx < size_x * size_y)
        {
            arr[idx] = k_rad;
        }
    }

    __global__ void real_complex_elementwisemult (pycuda::complex<float> *arr_cmpx, float *arr_real,
    int size_x, int size_y) {
        int row_idx =  blockDim.y * blockIdx.y + threadIdx.y;
        int col_idx =  blockDim.x * blockIdx.x + threadIdx.x;
        int idx = row_idx * size_x  + col_idx;
        if (row_idx < size_y && col_idx < size_x)
        {
            arr_cmpx[idx] *= arr_real[idx];
        }
    }

    __global__ void hard_aperture(float *arr, float k_max, float k_semi, int size_x, int size_y){
        int row_idx =  blockDim.y * blockIdx.y + threadIdx.y;
        int col_idx =  blockDim.x * blockIdx.x + threadIdx.x;
        int idx = row_idx * size_x  + col_idx;
        float k_rad = calc_krad(k_max, size_x, size_y, col_idx, row_idx);
        if (row_idx < size_x && col_idx < size_y)
        {
            if ( k_rad  < k_semi ){
                arr[idx] = 1.f;
            }
            else {
                arr[idx] = 0.f;
            }
        }
    }

    __global__ void soft_aperture(float *arr, float k_max, float k_semi, int size_x, int size_y){
        int row_idx =  blockDim.y * blockIdx.y + threadIdx.y;
        int col_idx =  blockDim.x * blockIdx.x + threadIdx.x;
        int idx = row_idx * size_x  + col_idx;
        float k_rad = calc_krad(k_max, size_x, size_y, col_idx, row_idx);
        if (row_idx < size_y && col_idx < size_x)
        {
            arr[idx] = 1.f / (1.f + expf(- 2.f * 80.f * (k_semi - k_rad)));
        }
    }

    __global__ void spherical_phase_error(pycuda::complex<float> *arr, float k_max, float Lambda,
    float C1, float C3, float C5, int size_x, int size_y) {
        const double pi = acos(-1.f);
        int row_idx =  blockDim.y * blockIdx.y + threadIdx.y;
        int col_idx =  blockDim.x * blockIdx.x + threadIdx.x;
        int idx = row_idx * size_x  + col_idx;
        double k_rad = calc_krad(k_max, size_x, size_y, col_idx, row_idx);
        double chi = 2. * pi / Lambda * (-1.f/2 * C1 * pow(k_rad * Lambda, 2) + 1.f/4 * C3 * pow(k_rad * Lambda, 4)
        + 1.f/6 * C5 * pow(k_rad * Lambda, 6));
        if (row_idx < size_y && col_idx < size_x)
        {
         arr[idx]._M_re =  cos(chi);
         arr[idx]._M_im = -sin(chi);
        }
    }


    __global__ void build_probes_stack(pycuda::complex<float> psi_pos[][{{y_sampling}}][{{x_sampling}}],
                        pycuda::complex<float> psi_k[{{y_sampling}}][{{x_sampling}}],
                        int z_size, float k_max, float x_dim, float y_dim, float step, float range){
        const double pi = acos(-1.0);
        unsigned col_idx = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned row_idx = blockIdx.y*blockDim.y + threadIdx.y;
        unsigned stk_idx = blockIdx.z*blockDim.z + threadIdx.z;
        if (col_idx < {{x_sampling}} && row_idx < {{y_sampling}} && stk_idx < z_size)
        {
            float kr = phase_shift(k_max, {{x_sampling}}, {{y_sampling}}, z_size,
            col_idx, row_idx, stk_idx, step, range, x_dim, y_dim);
            psi_pos[stk_idx][row_idx][col_idx]  = pycuda::complex<float>(cosf(2 * pi * kr), sinf(2 * pi * kr));
            psi_pos[stk_idx][row_idx][col_idx] *= psi_k[row_idx][col_idx];
        }
    }

    __global__ void fftshift_2d(pycuda::complex<float> *arr, int size_x, int size_y){
        // 2D Slice & 1D Line
        int sLine = size_x;
        int sSlice = size_x * size_y;

        // Transformations Equations
        int sEq1 = (sSlice + sLine) / 2;
        int sEq2 = (sSlice - sLine) / 2;

       int row_idx = blockDim.y * blockIdx.y + threadIdx.y;
       int col_idx = blockDim.x * blockIdx.x + threadIdx.x;
       int idx = row_idx * size_x  + col_idx;
       pycuda::complex<float> temp = pycuda::complex<float>(0.,0.);
       if (col_idx < size_x / 2)
        {
            if (row_idx < size_y / 2)
            {
                temp = arr[idx];

                // First Quad
                arr[idx] = arr[idx + sEq1];

                // Third Quad
                arr[idx + sEq1] = temp;
            }
        }
        else
        {
            if (row_idx < size_y / 2)
            {
                temp = arr[idx];

                // Second Quad
                arr[idx] = arr[idx + sEq2];

                // Fourth Quad
                arr[idx + sEq2] = temp;
            }
        }
    }

    __global__ void fftshift_2d_stack(pycuda::complex<float> arr[][{{y_sampling}} * {{x_sampling}}],
                                    int z_size, float k_max_x, float k_max_y, float x_dim, float y_dim, float step, float range){
        // 2D Slice & 1D Line
        int sLine = {{x_sampling}};
        int sSlice = {{x_sampling}} * {{y_sampling}};

        // Transformations Equations
        int sEq1 = (sSlice + sLine) / 2;
        int sEq2 = (sSlice - sLine) / 2;

       int row_idx = blockDim.y * blockIdx.y + threadIdx.y;
       int col_idx = blockDim.x * blockIdx.x + threadIdx.x;
       int stk_idx = blockIdx.z * blockDim.z + threadIdx.z;
       int idx = row_idx * {{x_sampling}}  + col_idx;
       pycuda::complex<float> temp = pycuda::complex<float>(0.,0.);
       if (col_idx < {{x_sampling}} / 2)
        {
            if (row_idx < {{y_sampling}} / 2)
            {
                temp = arr[stk_idx][idx];

                // First Quad
                arr[stk_idx][idx] = arr[stk_idx][idx + sEq1];

                // Third Quad
                arr[stk_idx][idx + sEq1] = temp;
            }
        }
        else
        {
            if (row_idx < {{y_sampling}} / 2)
            {
                temp = arr[stk_idx][idx];

                // Second Quad
                arr[stk_idx][idx] = arr[stk_idx][idx + sEq2];

                // Fourth Quad
                arr[stk_idx][idx + sEq2] = temp;
            }
        }
    }
""")

class ProbeKernels(object):
    def __init__(self, sampling=np.array([512, 512])):
        self.x_sampling= np.int(sampling[1])
        self.y_sampling= np.int(sampling[0])
        kernels = cuda_kerns.render(type= dtype_to_ctype(np.complex64), y_sampling=self.y_sampling, x_sampling=self.x_sampling)
        src = SourceModule(kernels)
        self.kernels['hard_aperture'] = src.get_function('hard_aperture')
        self.kernels['soft_aperture'] = src.get_function('soft_aperture')
        self.kernels['spherical_phase_error'] = src.get_function('spherical_phase_error')
        self.kernels['mult_wise_c2d_re2d'] = src.get_function('real_complex_elementwisemult')
        self.kernels['fftshift_2d'] = src.get_function('fftshift_2d')
        self.kernels['fftshift_2d_stack'] = src.get_function('fftshift_2d_stack')
        self.kernels['k_grid'] = src.get_function('k_grid')
        self.kernels['probes_stack'] = src.get_function('build_probes_stack')
