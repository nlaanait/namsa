#include <pycuda-complex.hpp>
#define FULL_MASK 0xffffffff
#include <stdio.h>

     __global__ void BuildScatteringPotential(pycuda::complex<float> slice[][{{y_sampling}}][{{x_sampling}}],
                                  float atom_pot_stack[][{{pot_shape_y}}][{{pot_shape_x}}],
                                  float sites[][{{sites_size}}],
                                  float sigma)

     {
         const int pot_size_y = {{pot_shape_y}}, pot_size_x = {{pot_shape_x}};
         const int slice_size_y = {{y_sampling}}, slice_size_x = {{x_sampling}};
         const int num_slices = {{num_slices}}, sites_size = {{sites_size}};
         int row_idx = blockDim.y * blockIdx.y + threadIdx.y;
         int col_idx = blockDim.x * blockIdx.x + threadIdx.x;
         int stk_idx = blockDim.z * blockIdx.z + threadIdx.z;
        // if (stk_idx == 0  && row_idx == 0  && col_idx == 0)
        // {
        // #pragma unroll
        for (int slice_num=0; slice_num<num_slices; slice_num++)
        {
             if (stk_idx == slice_num)
             {
                 for(int my_site=0; my_site<sites_size/3; my_site++)
                 {
                     const int Z = (int) sites[stk_idx][3 * my_site];
                     const int y_cen = (int) sites[stk_idx][3 * my_site + 1];
                     const int x_cen = (int) sites[stk_idx][3 * my_site + 2];
                     const int y_start = rintf(y_cen - pot_size_y * 1.f/2);
                     const int y_end =  rintf(y_cen + pot_size_y * 1.f /2);
                     const int x_start = rintf(x_cen - pot_size_x * 1.f/2);
                     const int x_end = rintf(x_cen + pot_size_x * 1.f/2);
                     //printf(" slice_num: %d, Z: %d, y_cen:%d, x_cen: %d\\n ", slice_num, Z, y_cen, x_cen);
                     if (row_idx >=0 && row_idx < y_end && row_idx < slice_size_y && row_idx >= y_start
                         && col_idx >=0 && col_idx < x_end && col_idx < slice_size_x && col_idx >= x_start)
                     {
                         const int pot_i = row_idx-y_start, pot_j = col_idx-x_start;
                         if (pot_i < pot_size_y && pot_j < pot_size_x && pot_i >= 0 && pot_j >= 0){
                            atomicAdd(&slice[stk_idx][row_idx][col_idx]._M_re, atom_pot_stack[Z][pot_i][pot_j]);
                         }
                     }
                 }
             }
         }
         __syncthreads();

        if (col_idx < slice_size_x && row_idx < slice_size_y && stk_idx < num_slices)
        {
             slice[stk_idx][row_idx][col_idx] = pycuda::complex<float>(cosf(slice[stk_idx][row_idx][col_idx]._M_re * sigma),
                                                                       sinf(slice[stk_idx][row_idx][col_idx]._M_re * sigma));
             if (isnan(slice[stk_idx][row_idx][col_idx]._M_re)){
                atomicExch(&slice[stk_idx][row_idx][col_idx]._M_re, 1.f);
             } 
             if (isnan(slice[stk_idx][row_idx][col_idx]._M_im)){
                atomicExch(&slice[stk_idx][row_idx][col_idx]._M_im, 1.f);
             }
        }

     }
