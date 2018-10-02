import pycuda.driver as cuda
from pycuda.tools import dtype_to_ctype
import pycuda.autoinit
from pycuda.compiler import SourceModule
from jinja2 import Template

cuda_kerns = Template("""
    #include <pycuda-complex.hpp>
    __device__ double calc_krad(float k_max_x, float k_max_y, int size_x, int size_y, int col_idx, int row_idx)
    {
        double kx = double(col_idx) * double(k_max_x)/double(size_x - 1) - double(k_max_x)/2. ; 
        double ky = double(row_idx) * double(k_max_y)/double(size_y - 1) - double(k_max_y)/2. ; 
        double k_rad = sqrt(kx * kx + ky * ky);
        return k_rad;
    }

    __global__ void k_grid(double *arr, double k_max_x, double k_max_y, int size_x, int size_y)
    {
        int row_idx = blockDim.y * blockIdx.y + threadIdx.y;
        int col_idx = blockDim.x * blockIdx.x + threadIdx.x;
        int idx = row_idx * size_x  + col_idx;
        double k_rad = calc_krad(k_max_x, k_max_y, size_x, size_y, col_idx, row_idx);
        if (idx < size_x * size_y)
        {
        arr[idx] = k_rad;
        } 
    }

    __global__ void real_complex_elementwisemult (pycuda::complex<float> *arr_cmpx, float *arr_real, 
    int size_x, int size_y)
    {
        int row_idx = blockDim.y * blockIdx.y + threadIdx.y;
        int col_idx = blockDim.x * blockIdx.x + threadIdx.x;
        int idx = row_idx * size_x  + col_idx;
        if (row_idx < size_y && col_idx < size_x)
        {
        arr_cmpx[idx] *= arr_real[idx];
        }
    }

    __global__ void hard_aperture(float *arr, float k_max_x, float k_max_y, float k_semi, int size_x, int size_y)
    {
        int row_idx = blockDim.y * blockIdx.y + threadIdx.y;
        int col_idx = blockDim.x * blockIdx.x + threadIdx.x;
        int idx = row_idx * size_x  + col_idx;
        float k_rad = calc_krad(k_max_x, k_max_y, size_x, size_y, col_idx, row_idx);
        if (row_idx < size_x && col_idx < size_y)
        {
            if ( k_rad  < k_semi ){
                arr[idx] = 1.0;
            } 
            else {
                arr[idx] = 0.0;
            }
        }
    }

    __global__ void soft_aperture(float *arr, float k_max_x, float k_max_y, float k_semi, int size_x, int size_y)
    {
        int row_idx = blockDim.y * blockIdx.y + threadIdx.y;
        int col_idx = blockDim.x * blockIdx.x + threadIdx.x;
        int idx = row_idx * size_x  + col_idx;
        float k_rad = calc_krad(k_max_x, k_max_y, size_x, size_y, col_idx, row_idx);
        if (row_idx < size_y && col_idx < size_x)
        {
         arr[idx] = 1.0 / (1.0 + expf(- 2.0 * 80. * (k_semi - k_rad)));
        }
    }

    __global__ void spherical_phase_error(pycuda::complex<float> *arr, float k_max_x, float k_max_y, 
    float Lambda, float C1, float C3, float C5, int size_x, int size_y)
    {
        const double pi = acos(-1.0);
        int row_idx = blockDim.y * blockIdx.y + threadIdx.y;
        int col_idx = blockDim.x * blockIdx.x + threadIdx.x;
        int idx = row_idx * size_x  + col_idx;
        double k_rad = calc_krad(k_max_x, k_max_y, size_x, size_y, col_idx, row_idx);
        double chi = 2. * pi / Lambda * (-1./2 * C1 * pow(k_rad * Lambda, 2.) + 1./4 * C3 * pow(k_rad * Lambda, 4.) 
        + 1./6 * C5 * pow(k_rad * Lambda, 6.));
        if (row_idx < size_y && col_idx < size_x)
        {
         arr[idx]._M_re =  cos(chi);
         arr[idx]._M_im = -sin(chi);
        }
    }

/*
 * Copyright 2011-2014,
 * Marwan Abdellah: <abdellah.marwan@gmail.com>
 *
 * This library (cufftShift) is free software; you can redistribute it
 * and/or modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 *
 * Adpated for namsa by Numan Laanait
 */
    __global__ void fftshift_2d(pycuda::complex<float> *arr, int size_x, int size_y)
{
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
""")
cuda_kerns = cuda_kerns.render(type=dtype_to_ctype(np.complex64))
# , size_x=size_x, size_y=size_y, radius=radius)
mod = SourceModule(cuda_kerns)
soft_apt = mod.get_function('soft_aperture')
spherical_phase_err = mod.get_function('spherical_phase_error')
multwise_cmpx_real = mod.get_function('real_complex_elementwisemult')
fft_shift = mod.get_function('fftshift_2d')
make_k_grid = mod.get_function('k_grid')