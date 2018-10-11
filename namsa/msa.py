from .database import kirkland_params
from .optics import *
from .utils import *
from .probe_kernels import ProbeKernels
import numpy as np
from scipy.special import k0
import multiprocessing as mp
import ctypes
import sys
from warnings import warn
import os
import pyfftw
import pycuda
import pycuda.driver as cuda
import skcuda.fft as cufft
from time import time


XYZ_dtype = [('atomic_number', 'i'), ('x', 'f'), ('y', 'f'), ('z', 'f'), ('occ', 'f'), ('DW', 'f')]


def unwrap(args):
    (msa, params), kwargs = args
    method_name = kwargs.pop('method')
    if method_name == 'build_slices':
        return msa.make_slice(params)
    elif method_name == 'propagate_beams':
        return msa.propagate_beam(params)
    elif method_name == 'build_probes':
        return msa.build_probe(probe_position=params[0], probe_dict=params[1])


class MSA(object):
    def __init__(self, energy, semi_angle, supercell, sampling=np.array([512, 512]), max_angle=None, verbose=False,
                 debug=False, output_dir='', debye_waller=True):
        self.E = energy
        self.Lambda = voltage2Lambda(self.E*1e3)
        self.semi_ang = semi_angle
        self.verbose = verbose
        self.debug = debug
        self.output_dir = output_dir
        self.debye_waller = debye_waller
        # Load and set supercell properties
        if isinstance(supercell, np.ndarray):
            try:
                check = [supercell[0][field] for field in ['x', 'y', 'z', 'occ', 'DW', 'atomic_number']]
                supercell_arr = np.copy(supercell)
            except ValueError as err:
                warn('Supercell numpy array does not have required fields.')
                raise err
        elif isinstance(supercell, str):
            if os.path.exists(supercell):
                try:
                    supercell_arr = np.genfromtxt(supercell, skip_footer=1, skip_header=2, dtype=XYZ_dtype)
                except IOError as err:
                    warn('Unable to load supercell file into numpy array.')
                    raise err
            else:
                warn('Path %s was not found.' % supercell)
                sys.exit(0)
        else:
            warn('supercell format was not recognized.')
            sys.exit(0)
        self.supercell_xyz = np.column_stack([supercell_arr['x'], supercell_arr['y'], supercell_arr['z']]).astype(np.float32)
        self.supercell_Z = supercell_arr['atomic_number']
        self.supercell_dw = supercell_arr['DW']
        self.supercell_occ = supercell_arr['occ']

        # Adding random displacements due to thermal effects
        unique_Z = np.unique(self.supercell_Z)
        for Z in unique_Z:
            indx = np.where(self.supercell_Z == Z)[0]
            dw = self.supercell_dw[indx][0]
            displacements = float(self.debye_waller) * dw * np.random.standard_normal(size=self.supercell_xyz[indx].shape)
            self.supercell_xyz[indx] += displacements

        # Set simulation parameters
        self.dims = self.supercell_xyz.max(0) - self.supercell_xyz.min(0)
        if max_angle is None:
            self.sampling = sampling
            self.kmax = np.min(self.sampling/self.dims[:2])
            self.max_ang = self.kmax * self.Lambda
        else:
            self.max_ang = max_angle
            self.kmax = self.max_ang / self.Lambda
            self.sampling = np.floor(self.kmax * self.dims[:2]).astype(np.int)
        self.pix_size = self.dims[:2] / self.sampling
        self.kpix_size = self.kmax/self.sampling
        self.sigma = sigma_int(self.E*1e3)
        self.print_verbose('Simulation Parameters:\nSupercell dimensions xyz:%s (Å)\nReal, Reciprocal space pixel sizes:%s Å, %s 1/Å'
              '\nMax angle: %2.2f (rad)\nSampling in real and reciprocal space: %s pixels,\nThermal Effects: %s' %
              (format(np.round(self.dims, 2)), format(np.round(self.pix_size, 2)), format(np.round(self.kpix_size, 2)),
               self.max_ang, format(self.sampling), format(self.debye_waller)))

    def print_debug(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def print_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def calc_atomic_potentials(self, potential_range=8, oversample=2, kirkland=True):
        if kirkland:
            self.scattering_params = kirkland_params
            #TODO: Figure out how to calculate scattering potential using different methods
        start = potential_range / 2
        step_x, step_y = 1.j * np.floor(potential_range/self.pix_size).astype(np.int) * oversample
        grid_x, grid_y = np.mgrid[-start:start:step_x, -start:start:step_y]
        self.cached_pots = dict()
        for Z in np.unique(self.supercell_Z):
            self.cached_pots[Z] = self._get_potential(grid_x, grid_y, self.scattering_params[Z - 1], oversample)

    def build_potential_slices(self, slice_thickness):
        self.slice_t = slice_thickness
        num_slices = np.int(np.floor(self.dims[-1] / slice_thickness))
        tasks = [((self, slice_num), {'method': 'build_slices'}) for slice_num in range(num_slices)]
        processes = min(mp.cpu_count(), num_slices)
        chunk = np.int(np.floor(num_slices / processes))
        pool = mp.Pool(processes=processes, maxtasksperchild=1)
        jobs = pool.imap(unwrap, tasks, chunksize=chunk)
        potential_slices = np.array([j for j in jobs])
        pool.close()
        ##self.potential_slices = potential_slices.astype(np.float32)
        self.potential_slices = potential_slices
        self.print_verbose('Built %d potential slices with shape:%s pixels' % (self.potential_slices.shape[0],
                                                                  format(self.potential_slices.shape[1:])))

    def make_slice(self, args):
        slice_num = args
        mask = np.logical_and(self.supercell_xyz[:, -1] >= slice_num * self.slice_t,
                              self.supercell_xyz[:, -1] < (slice_num + 1) * self.slice_t)
        # TODO create empty slice that is byte aligned using fftw
        arr_slice = np.zeros(self.sampling, dtype=np.float32)
        for Z, xyz in zip(self.supercell_Z[mask], self.supercell_xyz[mask]):
            pot = self.cached_pots[Z]
            x_pix, y_pix = xyz[:2] * self.sampling / self.dims[:2]
            y_start, y_end = int(y_pix - pot.shape[0] / 2), int(y_pix + pot.shape[0] / 2)
            x_start, x_end = int(x_pix - pot.shape[1] / 2), int(x_pix + pot.shape[1] / 2)
            repl_y = slice(max(y_start, 0), y_end)
            repl_x = slice(max(x_start, 0), x_end)
            offset_y = abs(min(y_start, 0))
            offset_x = abs(min(x_start, 0))
            repl_shape = arr_slice[repl_y, repl_x].shape
            arr_slice[repl_y, repl_x] += pot[offset_y:repl_shape[0] + offset_y, offset_x:repl_shape[1] + offset_x]
        return arr_slice

    @staticmethod
    def _get_potential(grid_x, grid_y, params, oversample):
        eps = 0.1  # Å cutoff for Bessel function
        a_0 = 0.529  # Å Bohr radius
        e = 14.4  # V.Å electron charge
        r = np.sqrt(grid_x ** 2 + grid_y ** 2)
        prefac_1 = 4 * np.pi ** 2 * a_0 * e
        coeff = params
        a_coeff, b_coeff, c_coeff, d_coeff = coeff
        bessel_arg = 2 * np.pi * r[:, :, np.newaxis] * np.sqrt(b_coeff)
        bessel_eval = k0(bessel_arg + eps)
        sum_v = prefac_1 * a_coeff * bessel_eval + prefac_1 / 2. * c_coeff / d_coeff * np.exp(-np.pi ** 2 *
                                                                                     r[:, :, np.newaxis] ** 2 / d_coeff)
        v = sum_v.sum(-1)
        potential = bin_2d_array(v, dwnspl=oversample, mode='mean')
        potential -= potential.min()
        return potential.astype(np.float32)

    def build_probe(self, probe_position=np.array([0., 0.]), probe_dict={'smooth_apert': True, 'apert_smooth': 50, 'spherical_phase': True, 'aberration_dict':
                        {'C1': 0., 'C3': 0., 'C5': 0.},'scherzer': True}
                    ):
        self.probe_dict = probe_dict
        k_y, k_x = np.mgrid[-self.kmax/2: self.kmax/2: 1j*self.sampling[0],
                            -self.kmax/2: self.kmax/2: 1j * self.sampling[1]]
        k_rad = np.sqrt(k_x ** 2 + k_y ** 2)
        k_semi = self.semi_ang/self.Lambda

        # aperture function
        if probe_dict['smooth_apert']:
            aperture = 1 / (1 + np.exp(-2 * probe_dict['apert_smooth'] * (k_semi - k_rad)))
        else:
            aperture = np.heaviside(k_semi - k_rad, 0.5)

        # aberration
        if probe_dict['spherical_phase']:
            phase_error = spherical_phase_error(k_rad, self.Lambda, probe_dict['scherzer'],
                                                **probe_dict['aberration_dict'])

        else:
            pass
            #TODO: implement non-rotationally invariant phase error

        # probe wavefunction
        psi_k = aperture * phase_error
        psi_k = psi_k.astype(np.complex64)
        y, x = probe_position
        kr = k_x * x + k_y * y
        phase_shift = np.exp(2 * np.pi * 1.j * kr).astype(np.complex64)
        psi_x = pyfftw.interfaces.numpy_fft.ifft2(psi_k * phase_shift)
        psi_x = pyfftw.interfaces.numpy_fft.fftshift(psi_x)
        # TODO: need to make fft library choice optional
        # psi_x = np.fft.ifft2(psi_k * phase_shift, norm='ortho')
        # psi_x = np.fft.fftshift(psi_x)
        psi_x /= np.sqrt(np.sum(np.abs(psi_x) ** 2))
        return psi_x.astype(np.complex64), psi_k.astype(np.complex64), aperture.astype(np.float32)

    def build_propagator(self):
        k_y, k_x = np.mgrid[-self.kmax / 2: self.kmax / 2: 1j * self.sampling[0],
                   -self.kmax / 2: self.kmax / 2: 1j * self.sampling[1]]
        k_rad_sq = k_x ** 2 + k_y ** 2
        propag = np.exp(-np.pi * 1.j * self.Lambda * self.slice_t * k_rad_sq)
        return propag.astype(np.complex64)

    @staticmethod
    def bandwidth_limit_mask(arr_shape, radius=0.5):
        # assumes square image
        grid_x, grid_y = np.mgrid[-arr_shape[0] // 2:arr_shape[0] // 2, -arr_shape[1] // 2:arr_shape[1] // 2]
        r_grid = np.sqrt(grid_x ** 2 + grid_y ** 2)
        bl_mask = np.heaviside(max(arr_shape[0], arr_shape[1]) * radius - r_grid, 0)
        return bl_mask.astype(np.float32)

    def generate_probe_positions(self, probe_step=np.array([0.1, 0.1]), probe_range=np.array([[0., 1.0], [0., 1.0]])):
        grid_steps_x, grid_steps_y = np.floor(np.diff(probe_range).flatten() * self.dims[:2] / probe_step).astype(np.int)
        grid_range_x, grid_range_y = [(probe_range[i] - np.ones((2,)) * 0.5) * self.dims[i]
                                      for i in range(2)]
        y_pos, x_pos = np.mgrid[grid_range_y[0]: grid_range_y[1]: -1j * grid_steps_y,
                       grid_range_x[0]: grid_range_x[1]: -1j * grid_steps_x]
        probe_pos = np.array([[y, -x] for y, x in zip(y_pos.flatten()[::-1], x_pos.flatten())])
        self.probe_positions = probe_pos

    def multislice(self, probe_pos=np.array([0., 0.]), probe_grid=True, save_probes=True, bandwidth=1 / 3):
        # check for slices
        if isinstance(self.potential_slices, np.ndarray) is False:
            warn('Potential slices must be calculated first before calling multi_slice')
            return
        # check for probe
        if isinstance(self.probe_dict, dict) is False:
            warn('Probe wave function must be initialized first before calling multi_slice')
            return
        if probe_grid:
            # Define heuristics for python multiprocessing and FFTW multi-threading
            self.fftw_threads = int(self.sampling.max() // 512)
            processes = min(mp.cpu_count(), self.probe_positions.shape[0]) // self.fftw_threads
            chunk = int(np.floor(self.probe_positions.shape[0] / processes))


            # Put the potential slices in shared memory so all workers access it (asynchronously)
            global shared_slices
            shared_slices = mp.Array(ctypes.c_float, self.potential_slices.size, lock=False)
            temp = np.frombuffer(shared_slices, dtype=np.float32)
            for (i, pot) in enumerate(self.potential_slices):
                temp[i * pot.size:(i + 1) * pot.size] = pot.flatten().astype(np.float32)

            tasks = (((self, (probe_num, pos, save_probes, probe_grid, bandwidth)), {'method': 'propagate_beams'})
                     for (probe_num, pos) in enumerate(self.probe_positions))
            pool = mp.Pool(processes=processes, initargs=(shared_slices,))
            jobs = pool.map(unwrap, tasks, chunksize=chunk)
            trans_probes = np.array([j for j in jobs])
            pool.close()
            pool.join()
        else:
            trans_probes = self.propagate_beam([None, probe_pos, save_probes, probe_grid, bandwidth])

        self.print_verbose('Propagated %d probe wavefunctions' % trans_probes.shape[0])
        self.trans_probes = trans_probes
        return self.trans_probes

    def propagate_beam(self, args):
        probe_num, probe_pos, save_probes, probe_grid, bandwidth = args
        self.print_debug('received params.')
        propag = self.build_propagator()
        blim_mask = self.bandwidth_limit_mask(propag.shape, radius=bandwidth)
        probe, _, _ = self.build_probe(probe_pos, self.probe_dict)
        probes = []
        probe_last = probe
        if probe_grid:
            slices = np.frombuffer(shared_slices, dtype=np.float32)
            slices = slices.reshape(self.potential_slices.shape)
            self.print_debug('fetched data from shared memory.')
        else:
            slices = self.potential_slices
        slices = np.exp(1.j * self.sigma * slices).astype(np.complex64)

        for (i, trans) in enumerate(slices[::-1]):
            t_psi = pyfftw.byte_align(trans * probe_last)
            fft_fwd = pyfftw.builders.fft2(t_psi, threads=self.fftw_threads, avoid_copy=True)
            temp = pyfftw.byte_align(fft_fwd() * blim_mask * propag)
            fft_bwd = pyfftw.builders.ifft2(temp, threads=self.fftw_threads, avoid_copy=True)
            probe_last = fft_bwd()
            if save_probes:
                probes.append(probe_last)
        self.print_debug('finished beam propagation.')

        if save_probes:
            np.save(os.path.join(self.output_dir, 'probes_%d.npy') % probe_num, np.array(probes), allow_pickle=False)
            del probes
            self.print_verbose('finished with probe position: %s' % format(probe_pos))

        return probe_last

    def check_simulation(self):
        prob = np.sum([np.abs(probe)**2 for probe in self.trans_probes], axis=(1, 2))
        max_val = prob.max()
        min_val = prob.min()
        print('Max (Min) Integrated Intensity: %2.2f (%2.2f)' % (max_val, min_val))
        if max_val > 1.0 or min_val < 0.95:
            print('Significant deviation of the probability from unity is found.\n'
                  'Change the sampling and/or slice thickness!')
        return prob


class MSAHybrid(MSA):
    def setup_device(self, dev_num=0):
        # TODO: setup device or devices...
        pycuda.tools.make_default_context()
        # result = pycuda.cuInit(dev_num)
        # if result != 0:
        #     print("cuInit failed ")
        #     return

    def plan_simulation(self, num_probes=64):
        self.num_probes = num_probes
        free_mem, tot_mem = pycuda.driver.mem_get_info()
        self.free_mem = free_mem/1024e6  # in GB
        mem_alloc = num_probes * np.prod(self.sampling) * 8 / 1024e6 + self.potential_slices.nbytes / 1024e6
        print('mem_alloc:',mem_alloc)
        print('free_mem:',self.free_mem)
        while mem_alloc > 0.95 * self.free_mem:
            num_probes = num_probes // 2
            mem_alloc = num_probes * np.prod(self.sampling) * 8 / 1024e6 + self.potential_slices.nbytes / 1024e6
        self.batch = num_probes
        self.print_verbose('Simulation will propagate %d probes simultaneously' % self.num_probes)

    def multislice(self, save_probes=True, bandwidth=1/3):
        # not supporting a single probe!!
        # check for slices
        if isinstance(self.potential_slices, np.ndarray) is False:
            warn('Potential slices must be calculated first before calling multi_slice')
            return
        # check for probe
        if isinstance(self.probe_dict, dict) is False:
            warn('Probe wave function must be initialized first before calling multi_slice')
            return
        # check for probe positions
        if isinstance(self.probe_positions, np.ndarray) is False:
            warn('probe positions must be initialized first before calling multi_slice')
            return
        # Initialize needed data
        slices = np.exp(1.j * self.sigma * self.potential_slices).astype(np.complex64)
        propag = self.build_propagator()
        mask = self.bandwidth_limit_mask(propag.shape, radius=bandwidth)
        t = time()
        self.probes = self.build_probes_cpu()
        sim_t = time() - t
        self.print_verbose('Spent %2.4f s building %d probes on cpu' % (sim_t, self.batch))

        # Copy over to device
        trans_gpu = pycuda.gpuarray.to_gpu_async(slices)
        mask_propag_gpu = pycuda.gpuarray.to_gpu_async(mask * propag)
        probes_gpu = pycuda.gpuarray.to_gpu_async(self.probes)

        # Setup fft plans
        # TODO: tile multiple fft plans
        t = time()
        plan = cufft.Plan(propag.shape, np.complex64, np.complex64, batch=self.batch)
        for slice_num in range(trans_gpu.shape[0]):
            for z_slice in range(probes_gpu.shape[0]):
                probes_gpu[z_slice] *= trans_gpu[slice_num]
            cufft.fft(probes_gpu, probes_gpu, plan, True)
            for z_slice in range(probes_gpu.shape[0]):
                probes_gpu[z_slice] *= mask_propag_gpu
            cufft.ifft(probes_gpu, probes_gpu, plan, False)
        cufft.fft(probes_gpu, probes_gpu, plan, True)  #return probe wavefunctions in reciprocal space
        sim_t = time() - t
        self.print_verbose('Propagated %d probes in %2.4f s' % (self.batch, sim_t))
        self.probes = probes_gpu.get()

        # free up device memory
        trans_gpu.gpudata.free()
        mask_propag_gpu.gpudata.free()
        probes_gpu.gpudata.free()

        # destroy cufft object
        cufft.cufft.cufftDestroy(plan.handle)
        return self.probes

    def build_probes_cpu(self):
        processes = min(mp.cpu_count(), self.probe_positions.shape[0]) // 4
        chunk = int(np.floor(self.probe_positions.shape[0] / processes))
        tasks = (((self, (pos, self.probe_dict)), {'method': 'build_probes'})
                 for pos in self.probe_positions)
        pool = mp.Pool(processes=processes)
        jobs = pool.map(unwrap, tasks, chunksize=chunk)
        probes = np.array([j[0] for j in jobs])
        pool.close()
        pool.join()
        return probes


class MSAGPU(MSAHybrid):

    def build_potential_slices(self, slice_thickness):
        self.slice_t = slice_thickness
        self.num_slices = np.int32(np.floor(self.dims[-1] / slice_thickness))
        tasks = [((self, slice_num), {'method': 'build_slices'}) for slice_num in range(self.num_slices)]
        processes = min(mp.cpu_count(), self.num_slices)
        chunk = np.int(np.floor(self.num_slices / processes))
        pool = mp.Pool(processes=processes)
        jobs = pool.imap(unwrap, tasks, chunksize=chunk)
        self.potential_slices = np.array([j for j in jobs])
        pool.close()
        self.print_verbose('Built %d potential slices with shape:%s pixels' % (self.potential_slices.shape[0],
                                                                  format(self.potential_slices.shape[1:])))
    def make_slice(self, args):
        # overriding this method
        slice_num = args
        mask = np.logical_and(self.supercell_xyz[:, -1] >= slice_num * self.slice_t,
                              self.supercell_xyz[:, -1] < (slice_num + 1) * self.slice_t)
        # TODO create empty slice that is byte aligned using fftw
        arr_slice = np.zeros(self.sampling, dtype=np.float32)
        for Z, xyz in zip(self.supercell_Z[mask], self.supercell_xyz[mask]):
            pot = self.cached_pots[Z]
            x_pix, y_pix = xyz[:2] * self.sampling / self.dims[:2]
            y_start, y_end = int(y_pix - pot.shape[0] / 2), int(y_pix + pot.shape[0] / 2)
            x_start, x_end = int(x_pix - pot.shape[1] / 2), int(x_pix + pot.shape[1] / 2)
            repl_y = slice(max(y_start, 0), y_end)
            repl_x = slice(max(x_start, 0), x_end)
            offset_y = abs(min(y_start, 0))
            offset_x = abs(min(x_start, 0))
            repl_shape = arr_slice[repl_y, repl_x].shape
            arr_slice[repl_y, repl_x] += pot[offset_y:repl_shape[0] + offset_y, offset_x:repl_shape[1] + offset_x]
        return np.exp(1.j * self.sigma * arr_slice)

    def _load_kernels(self):
        try:
            probe_kernels = ProbeKernels(sampling=self.sampling)
            self.kernels = probe_kernels.kernels
            self.print_verbose('CUDA C/C++ Kernels compiled successfully')
        except cuda.CompileError:
            warn('CUDA C/C++ Kernels did not compile successfully')
            raise cuda.CompileError

    def build_probe(self, probe_dict={'smooth_apert': True, 'apert_smooth': 50, 'spherical_phase': True,
                                      'aberration_dict': {'C1': 0., 'C3': 0., 'C5': 0.},'scherzer': True}
                    ):

        # load kernels and prepare func args
        self._load_kernels()
        aber_dict = probe_dict['aberration_dict']
        k_semi = np.float32(self.semi_ang / self.Lambda)
        k_max, Lambda, C1, C3, C5 = [np.float32(itm) for itm in [self.kmax, self.Lambda, aber_dict['C1'],
                                                                 aber_dict['C3'], aber_dict['C5']]]
        if probe_dict['scherzer']:
            C1, _ = scherzer_params(self.Lambda, aber_dict['C3'])
            C1 = np.float32(C1)

        # grab appropriate kernels
        self.multwise_2d_func = self.kernels['mult_wise_c2d_re2d']
        fftshift_func = self.kernels['fftshift_2d']
        if probe_dict['smooth_apert']:
            apert_func = self.kernels['soft_aperture']
        else:
            apert_func = self.kernels['hard_aperture']
        if probe_dict['spherical_phase']:
            phase_func = self.kernels['spherical_phase_error']
        else:
            pass
            # TODO: implement non-rotationally invariant phase error kernel

        # define block/grid threads
        shape_x = np.int32(self.sampling[1])
        shape_y = np.int32(self.sampling[0])
        block, grid = self._get_blockgrid([shape_x, shape_y], mode='2D')

        # allocate memory
        self.apert = np.empty(self.sampling, dtype=np.float32)
        apert_d = cuda.mem_alloc(self.apert.nbytes)
        self.psi_k = np.empty(self.sampling, dtype=np.complex64)
        psi_k_d = cuda.mem_alloc(self.psi_k.nbytes)

        # build a probe in k-space
        apert_func(apert_d, k_max, k_semi, shape_x, shape_y, block=block, grid=grid, shared=0)
        #pycuda.autoinit.context.synchronize()
        phase_func(psi_k_d, k_max, Lambda, C1, C3, C5, shape_x, shape_y, block=block, grid=grid, shared=0)
        #pycuda.autoinit.context.synchronize()
        self.multwise_2d_func(psi_k_d, apert_d, shape_x, shape_y, block=block, grid=grid)
        #pycuda.autoinit.context.synchronize()
        cuda.memcpy_dtoh_async(self.psi_k, psi_k_d)
        cuda.memcpy_dtoh_async(self.apert, apert_d)

        # build probe in x-space
        psi_x = pycuda.gpuarray.GPUArray(self.psi_k.shape, self.psi_k.dtype)
        psi_k = pycuda.gpuarray.GPUArray(self.psi_k.shape, self.psi_k.dtype, allocator=psi_k_d, gpudata=psi_k_d)
        fft_plan = cufft.Plan(psi_k.shape, np.complex64, np.complex64, batch=1)
        cufft.ifft(psi_k, psi_x, fft_plan, False)
        #pycuda.autoinit.context.synchronize()
        fftshift_func(psi_x, shape_x, shape_y, block=block, grid=grid)
        #pycuda.autoinit.context.synchronize()
        self.psi = psi_x.get()
        self.psi /= np.sqrt(np.sum(np.abs(self.psi)**2))

        # free gpu mem
        psi_x.gpudata.free()
        psi_k_d.free()
        apert_d.free()
        cufft.cufft.cufftDestroy(fft_plan.handle)

    def generate_probe_positions(self, probe_step=np.array([0.1, 0.1]), probe_range=np.array([[0., 1.0], [0., 1.0]])):
        grid_steps_x, grid_steps_y = np.floor(np.diff(probe_range).flatten() * self.dims[:2] / probe_step).astype(np.int)
        grid_range_x, grid_range_y = [(probe_range[i] - np.ones((2,)) * 0.5) * self.dims[i]
                                      for i in range(2)]
        y_pos, x_pos = np.mgrid[grid_range_y[0]: grid_range_y[1]: -1j * grid_steps_y,
                       grid_range_x[0]: grid_range_x[1]: -1j * grid_steps_x]
        probe_pos = np.array([[y, -x] for y, x in zip(y_pos.flatten()[::-1], x_pos.flatten())])
        self.grid_steps = np.array([grid_steps_x, grid_steps_y])
        self.grid_range = np.array([grid_range_x, grid_range_y]).flatten()
        self.probe_positions = probe_pos

    @staticmethod
    def _get_blockgrid(shapes, mode='2D'):
        # define block/grid threads
        if mode == '3D':
            shape_x = shapes[0]
            shape_y = shapes[1]
            shape_z = shapes[2]
            blk_zsize = 1
            blk_xsize = 32
            blk_ysize = 32
            grid_xsize = int((shape_x + blk_xsize - 1) / blk_xsize)
            grid_ysize = int((shape_y + blk_ysize - 1) / blk_ysize)
            grid_zsize = int((shape_z + blk_zsize - 1) / blk_zsize)
            block_3d = (blk_xsize, blk_ysize, blk_zsize)
            grid_3d = (grid_xsize, grid_ysize, grid_zsize)
            return block_3d, grid_3d
        if mode == '2D':
            shape_x = shapes[0]
            shape_y = shapes[1]
            block_2d = (32, 32, 1)
            grid_2d = (int((shape_x + block_2d[0] - 1) / 32), int((shape_y + block_2d[1] - 1) / 32), 1)
            return block_2d, grid_2d

    def multislice(self, bandwidth=1/3):

        # checks
        if isinstance(self.potential_slices, np.ndarray) is False:
            warn('Potential slices must be calculated first before calling multi_slice\n. '
                 'Call method build_potential_slices().')
            return
        if isinstance(self.psi_k, np.ndarray) is False:
            warn('Probe in k-space must be initiated first before calling multi_slice\n. '
                 'Call method build_probe().')
            return
        if isinstance(self.probe_positions, np.ndarray) is False:
            warn('Probe positions must be calculated first before calling multislice\n. '
                 'Call method generate_probe_positions().')
            return

        # load cuda kernels and prepare arguments
        self._load_kernels()
        num_probes = np.int32(self.probe_positions.shape[0])
        shape_x = np.int32(self.sampling[1])
        shape_y = np.int32(self.sampling[0])
        self.plan_simulation(num_probes=num_probes)
        num_probes = self.batch

        # define block/grid threads
        block_3d, grid_3d = self._get_blockgrid([self.sampling[1], self.sampling[0], num_probes], mode='3D')
        block_2d, grid_2d = self._get_blockgrid([self.sampling[1], self.sampling[0], num_probes], mode='2D')
        print('block, grid:', block_3d, grid_3d)

        # setup fft plan
        fft_plan_probe = cufft.Plan(self.sampling, np.complex64, np.complex64, batch=num_probes)

        # allocate memory
        self.probes = np.empty((num_probes, shape_y, shape_x), dtype=np.complex64)
        psi_pos_d = cuda.mem_alloc(self.probes.nbytes)
        self.propag = np.empty(self.sampling, dtype=np.complex64)
        propag_d = cuda.to_device(self.propag)
        self.mask = np.empty(self.sampling, dtype=np.float32)
        mask_d = cuda.to_device(self.mask)
        atom_slices = pycuda.gpuarray.to_gpu_async(self.potential_slices)

        # grab needed kernels
        probe_stack_func = self.kernels['probes_stack']
        propag_func = self.kernels['propagator']
        mask_func = self.kernels['hard_aperture']
        multwise_stack_func = self.kernels['mult_wise_c3d_c2d']
        multwise_func = self.kernels['mult_wise_c2d_re2d']
        fftshift_func = self.kernels['fftshift_2d_stack']

        #1. build probes
        probe_stack_func(psi_pos_d, cuda.In(self.psi_k), num_probes, np.float32(self.kmax),
        cuda.In(self.grid_steps.astype(np.float32)), cuda.In(self.grid_range.astype(np.float32)),
                           block=block_3d, grid=grid_3d, shared=0)
        #pycuda.autoinit.context.synchronize()
        psi_x_pos = pycuda.gpuarray.GPUArray(self.probes.shape, self.probes.dtype, allocator=psi_pos_d, gpudata=psi_pos_d)
        cufft.ifft(psi_x_pos, psi_x_pos, fft_plan_probe, False)
        #pycuda.autoinit.context.synchronize()
        fftshift_func(psi_x_pos, block=block_3d, grid=grid_3d, shared=0)
        #pycuda.autoinit.context.synchronize()
        # TODO: normalization kernel


        # 2. Build propagator and bandwidth limiting mask
        self.slice_t = 1.25
        mask_func(mask_d, np.float32(self.kmax), np.float32(bandwidth * self.kmax), shape_x, shape_y, block=block_2d,
                  grid=grid_2d, shared=0)
        pycuda.autoinit.context.synchronize()
        propag_func(propag_d, np.float32(self.kmax), np.float32(1.0), np.float32(self.Lambda), shape_x, shape_y,
                    block=block_2d, grid=grid_2d, shared=0)
        pycuda.autoinit.context.synchronize()
        multwise_func(propag_d, mask_d, shape_x, shape_y, block=block_2d, grid=grid_2d, shared=0)
        cuda.memcpy_dtoh_async(self.propag, propag_d)
        cuda.memcpy_dtoh_async(self.mask, mask_d)

        # 3. Propagate probes through atomic potential
        t = time()
        pycuda.autoinit.context.synchronize()
        for i in range(self.num_slices):
            self.print_debug('Atomic potential slice #%d' % i)
            multwise_stack_func(psi_x_pos, atom_slices[i], num_probes, block=block_3d, grid=grid_3d, shared=0)
            #pycuda.autoinit.context.synchronize()
            cufft.fft(psi_x_pos, psi_x_pos, fft_plan_probe, True)
            #pycuda.autoinit.context.synchronize()
            multwise_stack_func(psi_x_pos, propag_d, num_probes, block=block_3d, grid=grid_3d, shared=0)
            #pycuda.autoinit.context.synchronize()
            cufft.ifft(psi_x_pos, psi_x_pos, fft_plan_probe, False)
            #pycuda.autoinit.context.synchronize()

        cufft.fft(psi_x_pos, psi_x_pos, fft_plan_probe, True)  #return probe wavefunctions in reciprocal space
        #pycuda.autoinit.context.synchronize()
        sim_t = time() - t
        self.print_verbose('Propagated %d probes in %2.4f s' % (np.prod(self.grid_steps), sim_t))
        cuda.memcpy_dtoh_async(self.probes,psi_pos_d)
        #pycuda.autoinit.context.synchronize()

        # Free memory
        psi_pos_d.free()
        atom_slices.gpudata.free()
        mask_d.free()
        propag_d.free()
        cufft.cufft.cufftDestroy(fft_plan_probe.handle)

        free_mem, tot_mem = pycuda.driver.mem_get_info()
        free_mem = free_mem / 1024e6  # in GB
        self.print_verbose('free mem:', free_mem)
