#include <pycuda-complex.hpp>
#define FULL_MASK 0xffffffff

__inline__ __device__ int warpReduceSumSync(int val, int mask){
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val += __shfl_down_sync( mask, val, offset);
  return val;
}

__device__ double calc_krad(float k_max, int size_x, int size_y, int col_idx, int row_idx){
    double kx = double(col_idx) * double(k_max)/double(size_x - 1) - double(k_max)/2. ;
    double ky = double(row_idx) * double(k_max)/double(size_y - 1) - double(k_max)/2. ;
    double k_rad = sqrt(kx * kx + ky * ky);
    return k_rad;
}

__device__ float phase_shift(float k_max, int size_x, int size_y, int col_idx, int row_idx, int stk_idx,
      int *grid_step, float *grid_range){
    const double pi = acos(-1.0);
    float kx = float(col_idx) * k_max/float(size_x - 1) - k_max/2.;
    float ky = float(row_idx) * k_max/float(size_y - 1) - k_max/2.;
    int grid_step_x = grid_step[0];
    int grid_step_y = grid_step[1];
    float grid_start_x = grid_range[0];
    float grid_end_x = grid_range[1];
    float grid_start_y = grid_range[2];
    float grid_end_y = grid_range[3];
    float ry_idx = rintf(floorf(stk_idx * 1.f  / grid_step_y));
    float rx_idx = stk_idx - ry_idx * grid_step_x;
    float ry = ry_idx * (grid_end_y - grid_start_y) / (grid_step_y - 1) + grid_start_y;
    float rx = rx_idx * (grid_end_x - grid_start_x) / (grid_step_x - 1) + grid_start_x;
    float kr = - kx * rx - ky * ry;
    return kr;
}


//TODO: 2d vectorize the indexing [idx]
__global__ void norm_const(pycuda::complex<float> arr[][{{x_sampling}} * {{y_sampling}}], float *norm, int size_z) {
  float sum = 0.f;
  int stk_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (stk_idx < size_z)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(idx;  idx < {{x_sampling}} * {{y_sampling}} ; idx += blockDim.x * gridDim.x)
      {
         sum += pycuda::norm(arr[stk_idx][idx]);
      }
    int mask = __ballot_sync(FULL_MASK, idx < {{x_sampling}} * {{y_sampling}});
    sum = warpReduceSumSync(sum, mask);
    if ((threadIdx.x & (warpSize - 1)) == 0)
     {
      atomicAdd(&norm[stk_idx], sum);
     }
  }
}

__global__ void normalize(pycuda::complex<float> arr[][{{y_sampling}}][{{x_sampling}}], float *norm, int size_z){
    int stk_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < {{y_sampling}} && col_idx < {{x_sampling}} && stk_idx < size_z)
    {
        arr[stk_idx][row_idx][col_idx] /= sqrtf(norm[stk_idx]);
    }
}

__global__ void k_grid(double *arr, double k_max, int size_x, int size_y){
    int row_idx =  blockDim.y * blockIdx.y + threadIdx.y;
    int col_idx =  blockDim.x * blockIdx.x + threadIdx.x;
    int idx = row_idx * size_x  + col_idx;
    double k_rad = calc_krad(k_max, size_x, size_y, col_idx, row_idx);
    if (idx < size_x * size_y)
    {
        arr[idx] = k_rad;
    }
}

__global__ void mult_wise_c2d_re2d (pycuda::complex<float> *arr_cmpx, float *arr_real, int size_x, int size_y) {
    int row_idx =  blockDim.y * blockIdx.y + threadIdx.y;
    int col_idx =  blockDim.x * blockIdx.x + threadIdx.x;
    int idx = row_idx * size_x  + col_idx;
    if (row_idx < size_y && col_idx < size_x)
    {
        arr_cmpx[idx] *= arr_real[idx];
    }
}

__global__ void mult_wise_c3d_c2d (pycuda::complex<float> arr_3d[][{{y_sampling}}][{{x_sampling}}],
    pycuda::complex<float> arr_2d[{{y_sampling}}][{{x_sampling}}], int z_size, float scale){
    int row_idx =  blockDim.y * blockIdx.y + threadIdx.y;
    int col_idx =  blockDim.x * blockIdx.x + threadIdx.x;
    int stk_idx =  blockDim.z * blockIdx.z + threadIdx.z;
    if (row_idx < {{y_sampling}} && col_idx < {{x_sampling}} && stk_idx < z_size)
    {
        arr_3d[stk_idx][row_idx][col_idx] *= arr_2d[row_idx][col_idx] * scale;
    }
}

__global__ void mod_square_stack(pycuda::complex<float> arr_3d[][{{y_sampling}}][{{x_sampling}}], int z_size){
    int row_idx = blockDim.y * blockIdx.y + threadIdx.y;
    int col_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stk_idx = blockDim.z * blockIdx.z + threadIdx.z;
    if (row_idx < {{y_sampling}} && col_idx < {{x_sampling}} && stk_idx < z_size)
    {
      arr_3d[stk_idx][row_idx][col_idx] = pycuda::norm(arr_3d[stk_idx][row_idx][col_idx]);
    }
}

__global__ void propagator(pycuda::complex<float> *arr, float k_max, float t_slice,float Lambda, int size_x, int size_y){
    const double pi = acos(-1.0);
    int row_idx =  blockDim.y * blockIdx.y + threadIdx.y;
    int col_idx =  blockDim.x * blockIdx.x + threadIdx.x;
    int idx = row_idx * size_x  + col_idx;
    if (row_idx < size_y && col_idx < size_x)
    {
        float k_rad = calc_krad(k_max, size_x, size_y, col_idx, row_idx);
        arr[idx]._M_re =  cosf(pi * k_rad * k_rad * Lambda * t_slice);
        arr[idx]._M_im =  - sinf(pi * k_rad * k_rad * Lambda * t_slice);
    }
}

__global__ void hard_aperture(float *arr, float k_max, float k_semi, int size_x, int size_y){
    int row_idx =  blockDim.y * blockIdx.y + threadIdx.y;
    int col_idx =  blockDim.x * blockIdx.x + threadIdx.x;
    int idx = row_idx * size_x  + col_idx;
    float k_rad = calc_krad(k_max, size_x, size_y, col_idx, row_idx);
    if (row_idx < size_y && col_idx < size_x)
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
                    int z_size, float k_max, int *grid_step, float *grid_range){
    const double pi = acos(-1.0);
    unsigned col_idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned row_idx = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned stk_idx = blockIdx.z*blockDim.z + threadIdx.z;
    if (col_idx < {{x_sampling}} && row_idx < {{y_sampling}} && stk_idx < z_size)
    {
        float kr = phase_shift(k_max, {{x_sampling}}, {{y_sampling}}, col_idx, row_idx, stk_idx, grid_step, grid_range);
        psi_pos[stk_idx][row_idx][col_idx]  = pycuda::complex<float>(cosf(2 * pi * kr), sinf(2 * pi * kr));
        psi_pos[stk_idx][row_idx][col_idx] *= psi_k[row_idx][col_idx];
    }
}


__global__ void fftshift_2d(pycuda::complex<float> arr[][{{x_sampling}}], int size_y){

   int row_idx = blockDim.y * blockIdx.y + threadIdx.y;
   int col_idx = blockDim.x * blockIdx.x + threadIdx.x;
   pycuda::complex<float> temp = pycuda::complex<float>(0.,0.);
   int bd_y = rintf(floorf(size_y * 1.f / 2));
   int bd_x = rintf(floorf({{x_sampling}} * 1.f  / 2));

   if (row_idx <= bd_y && col_idx < bd_x && row_idx + bd_y < size_y)
    {
      temp = arr[row_idx][col_idx];
      arr[row_idx][col_idx] = arr[row_idx + bd_y][col_idx + bd_x];
      arr[row_idx + bd_y][col_idx + bd_x] = temp;
    }

    if (row_idx <= bd_y && col_idx >= bd_x && col_idx < {{x_sampling}} && row_idx + bd_y < size_y)
    {
      temp = arr[row_idx][col_idx];
      arr[row_idx][col_idx] = arr[row_idx + bd_y][col_idx - bd_x];
      arr[row_idx + bd_y][col_idx - bd_x] = temp;
    }
}


__global__ void fftshift_2d_stack(pycuda::complex<float> arr[][{{y_sampling}}][{{x_sampling}}], int size_z){

   int row_idx = blockDim.y * blockIdx.y + threadIdx.y;
   int col_idx = blockDim.x * blockIdx.x + threadIdx.x;
   int stk_idx = blockIdx.z * blockDim.z + threadIdx.z;
   pycuda::complex<float> temp = pycuda::complex<float>(0.,0.);
   int bd_y = rintf(floorf({{y_sampling}} * 1.f / 2));
   int bd_x = rintf(floorf({{x_sampling}} * 1.f / 2));

   if (stk_idx < size_z){
      if (row_idx <= bd_y && col_idx < bd_x && row_idx + bd_y < {{y_sampling}})
      {
        temp = arr[stk_idx][row_idx][col_idx];
        arr[stk_idx][row_idx][col_idx] = arr[stk_idx][row_idx + bd_y][col_idx + bd_x];
        arr[stk_idx][row_idx + bd_y][col_idx + bd_x] = temp;
      }

      if (row_idx <= bd_y && col_idx >= bd_x && col_idx < {{x_sampling}} && row_idx + bd_y < {{y_sampling}})
      {
        temp = arr[stk_idx][row_idx][col_idx];
        arr[stk_idx][row_idx][col_idx] = arr[stk_idx][row_idx + bd_y][col_idx - bd_x];
        arr[stk_idx][row_idx + bd_y][col_idx - bd_x] = temp;
      }
   }
}
