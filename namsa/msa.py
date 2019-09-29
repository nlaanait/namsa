from .database import kirkland_params
from .optics import *
from .utils import *
from .cuda_kernels import ProbeKernels, PotentialKernels
import numpy as np
from scipy.special import k0
from scipy import integrate
import multiprocessing as mp
import ctypes
import sys
from warnings import warn, catch_warnings, simplefilter
import os
try:
    import pyfftw
except:
    pyfftw=None
import pycuda.driver as cuda
from pycuda import gpuarray
import skcuda.fft as skfft
import skcuda.cufft as cufft
from time import time, sleep
import h5py
from mpi4py import MPI


XYZ_dtype = [('atomic_number', 'i'), ('x', 'f'), ('y', 'f'), ('z', 'f'), ('occ', 'f'), ('DW', 'f')]

def catch_warn():
    warn('complex_casting', RuntimeWarning)

def unwrap(args):
    (msa, params), kwargs = args
    method_name = kwargs.pop('method')
    if method_name == 'build_slices':
        return msa.make_slice(params)
    elif method_name == 'propagate_beams':
        return msa.propagate_beam(params)
    elif method_name == 'build_probes':
        return msa.build_probe(probe_position=params[0], probe_dict=params[1])


def setup_device(gpu_id=0):
    global ctx
    cuda.init()
    dev = cuda.Device(gpu_id)
    ctx = dev.make_context()

    import atexit
    def _clean_up():
        if ctx is not None:
            try:
                ctx.pop()
                ctx.detach()
            except Exception as e:
                warn(format(e))
        from pycuda.tools import clear_context_caches
        clear_context_caches()
    atexit.register(_clean_up)
    return ctx 

class MSA:
    '''
    Base Class Implementation of the Multi-Slice Algorithm. 
    Potential Construction and Beam Propagation are done in parallel (multiprocessing) and pyfftw (if available) is used for batched 2-D FFTs.
    '''
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
        self.max_angle = max_angle
        if self.max_angle is None:
            self.sampling = sampling
            self.kmax = np.min(self.sampling/self.dims[:2])
            self.max_ang = self.kmax * self.Lambda
        else:
            self.max_ang = self.max_angle
            self.kmax = self.max_ang / self.Lambda
            self.sampling = np.floor(self.kmax * self.dims[:2]).astype(np.int)
            self.sampling = self.sampling[::-1]
        self.pix_size = self.dims[:2][::-1] / self.sampling
        self.kpix_size = self.kmax/self.sampling
        self.sigma = sigma_int(self.E*1e3)
        self.print_verbose('Simulation Parameters:\nSupercell dimensions xyz:%s (Å)\nReal, Reciprocal space pixel sizes:%s Å, %s 1/Å'
              '\nMax angle: %2.2f (rad)\nSampling in real and reciprocal space: %s pixels,\nThermal Effects: %s' %
              (format(np.round(self.dims, 3)), format(np.round(self.pix_size, 3)), format(np.round(self.kpix_size, 3)),
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
        self.potential_slices = potential_slices
        self.print_verbose('Built %d potential slices with shape:%s pixels' % (self.potential_slices.shape[0],
                                                                  format(self.potential_slices.shape[1:])))

    def make_slice(self, args):
        slice_num = args
        mask = np.logical_and(self.supercell_xyz[:, -1] >= slice_num * self.slice_t,
                              self.supercell_xyz[:, -1] < (slice_num + 1) * self.slice_t)
        arr_slice = np.zeros(self.sampling, dtype=np.float32)
        for Z, xyz in zip(self.supercell_Z[mask], self.supercell_xyz[mask]):
            pot = self.cached_pots[Z]
            y_pix, x_pix = xyz[:2][::-1] / self.pix_size
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
        x, y = probe_position
        kr = -k_x * x - k_y * y
        phase_shift = np.exp(2 * np.pi * 1.j * kr).astype(np.complex64)
        if pyfftw is not None:
            psi_x = pyfftw.interfaces.numpy_fft.ifft2(psi_k * phase_shift)
            psi_x = pyfftw.interfaces.numpy_fft.fftshift(psi_x)
        else:
        # fall back on numpy 
            psi_x = np.fft.ifft2(psi_k * phase_shift, norm='ortho')
            psi_x = np.fft.fftshift(psi_x)
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

    def generate_probe_positions(self, probe_step=np.array([0.1, 0.1]), probe_range=np.array([[0., 1.0], [0., 1.0]]), 
            grid_steps=None, fraction=0.5, origin = np.array([0,0])):
        if grid_steps is not None:
            grid_range_start = (0.5 + origin[::-1] * self.dims[:2]  - self.dims[:2] * fraction)/2
            grid_range_stop = (0.5 + origin[::-1] * self.dims[:2] + self.dims[:2]  * fraction)/2
            # grid_range_start = (0.5 - self.dims[:2]/2)/2
            # grid_range_stop = (0.5 + self.dims[:2]/2)/2
            x_pos, y_pos = np.mgrid[grid_range_start[0]:grid_range_stop[0]:-1j*grid_steps[0], 
                                    grid_range_start[1]:grid_range_stop[1]:-1j*grid_steps[1]]
            grid_steps_x, grid_steps_y = grid_steps.astype(np.int) 
            grid_range_x = np.array([grid_range_start[0], grid_range_stop[0]])
            grid_range_y = np.array([grid_range_start[1], grid_range_stop[1]])
            probe_pos = np.array([[y, x] for y, x in zip(y_pos.flatten(), x_pos.flatten())])
            # probe_pos = np.array([[x, y] for y, x in zip(y_pos.flatten(), x_pos.flatten())])

        else:                            
            grid_steps_x, grid_steps_y = np.floor(np.diff(probe_range).flatten() * self.dims[:2] / probe_step).astype(np.int)
            grid_range_x, grid_range_y = [(probe_range[i] - np.ones((2,)) * 0.5) * self.dims[i]
                                        for i in range(2)]
            x_pos, y_pos = np.mgrid[grid_range_x[0]: grid_range_x[1]: -1j * grid_steps_x,
                        grid_range_y[0]: grid_range_y[1]: -1j * grid_steps_y]
            probe_pos = np.array([[y, x] for y, x in zip(y_pos.flatten(), x_pos.flatten())])
        self.grid_steps = np.array([grid_steps_x, grid_steps_y])
        self.grid_range = np.array([grid_range_x, grid_range_y]).flatten()
        self.probe_positions = probe_pos
        self.num_probes = np.int32(probe_pos.shape[0])
       
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
            if pyfftw is not None:
                # Define heuristics for python multiprocessing and FFTW multi-threading
                self.fftw_threads = int(self.sampling.max() // 512)
                processes = min(mp.cpu_count(), self.probe_positions.shape[0]) // self.fftw_threads
            else:
                processes = min(mp.cpu_count(), self.probe_positions.shape[0]) 
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
        self.probes = trans_probes
        return trans_probes

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

        if pyfftw is not None:
            for (i, trans) in enumerate(slices[::-1]):
                t_psi = pyfftw.byte_align(trans * probe_last)
                fft_fwd = pyfftw.builders.fft2(t_psi, threads=self.fftw_threads, avoid_copy=True)
                temp = pyfftw.byte_align(fft_fwd() * blim_mask * propag)
                fft_bwd = pyfftw.builders.ifft2(temp, threads=self.fftw_threads, avoid_copy=True)
                probe_last = fft_bwd()
                if save_probes:
                    probes.append(probe_last)
        else:
            for (i, trans) in enumerate(slices[::-1]):
                t_psi = trans * probe_last
                fft_fwd = np.fft.fft2(t_psi)
                temp = fft_fwd * blim_mask * propag
                fft_bwd = np.fft.ifft2(temp)
                probe_last = fft_bwd
                if save_probes:
                    probes.append(probe_last) 
        self.print_debug('finished beam propagation.')

        if save_probes:
            np.save(os.path.join(self.output_dir, 'probes_%d.npy') % probe_num, np.array(probes), allow_pickle=False)
            del probes
            self.print_verbose('finished with probe position: %s' % format(probe_pos))

        return probe_last

    def check_simulation(self):
        prob = np.sum([np.abs(probe)**2 for probe in self.probes], axis=(1, 2))
        max_val = prob.max()
        min_val = prob.min()
        print('Max (Min) Integrated Intensity: %2.2f (%2.2f)' % (max_val, min_val))
        if max_val > 1.0 or min_val < 0.95:
            print('Significant deviation of the probability from unity is found.\n'
                  'Change the sampling and/or slice thickness!')
        return prob

    def integrate_potential_slices(self, pot, grid, slice_thickness, output=False):
        '''
        Integrate a scattering potential, typically coming from all-electron ab initio simulations.
        args:
            pot: 3-d array, in units of e-/Angstrom**3. spatial order is [z,y,x]
            grid: (z,y,x), tuple of arrays with coordinates along each axis in units of Angstrom.
        '''
        
        self.slice_t = slice_thickness
        z_coord, y_coord, x_coord = grid
        z_sampling = z_coord[1] - z_coord[0]
        z_dim = z_coord[-1] - z_coord[0]
        num_slices = int(z_dim // self.slice_t)

        # Cropping to a square array- not necessary and should be removed later after cuda kernel tests 
        max_dim = min(x_coord.size, y_coord.size) 
        x_resamp = x_coord[:max_dim]
        y_resamp = y_coord[:max_dim]
        pot = pot[:, :max_dim, :max_dim]

        # Cropping potential array along z-dir to get int number of slices
        d_slice = int(pot.shape[0]//num_slices)
        if d_slice != int(np.ceil(pot.shape[0]/num_slices)):
            pot_resamp = pot[:d_slice * num_slices, :, :]
            z_resamp = z_coord[:d_slice * num_slices]
        else:
            pot_resamp = pot 
            z_resamp = z_coord

        # Integrating
        pot_slices = np.split(pot_resamp, num_slices, axis=0)
        zcoord_slices = np.split(z_resamp, num_slices, axis=0)
        pot_slices = np.array([integrate.trapz(pot_slice, x=z_coord, axis=0) for pot_slice, z_coord in zip(pot_slices, zcoord_slices)])
        self.potential_slices = - pot_slices

        # Update sim params
        ## TODO: below doesn't take into account user-specified max_ang
        self.num_slices = self.potential_slices.shape[0]
        self.dims = np.array([x_coord.max(), y_coord.max(), z_coord.max()])
        if self.max_angle is None:
            self.sampling = np.array(self.potential_slices.shape[1:]) 
            self.kmax = np.min(self.sampling/self.dims[:2])
            self.max_ang = self.kmax * self.Lambda
        else:
            self.max_ang = self.max_angle
            self.kmax = self.max_ang / self.Lambda
            self.sampling = np.floor(self.kmax * self.dims[:2]).astype(np.int)
            self.sampling = self.sampling[::-1]
        self.pix_size = self.dims[:2][::-1] / self.sampling
        self.kpix_size = self.kmax/self.sampling
        if output:
            return - pot_slices
    
    def integrate_cbed(self, detector_array=None, detector_params={'inner_angle':50e-3, 'outer_angle':100e-3}):
        if detector_array is None:
            assert detector_params['inner_angle'] < detector_params['outer_angle'] and detector_params['outer_angle'] < self.max_ang, \
                print('Detector angles exceed maximum scattering angle simulated and/or values are not consistent') 
            inner_radius = detector_params['inner_angle'] / (self.kpix_size * self.Lambda * self.sampling) / 2
            outer_radius = detector_params['outer_angle'] / (self.kpix_size * self.Lambda * self.sampling) / 2
            print(inner_radius, outer_radius) 
            inner_segment = self.bandwidth_limit_mask(self.sampling, radius=inner_radius[0]).astype(np.bool)
            outer_segment = self.bandwidth_limit_mask(self.sampling, radius=outer_radius[0]).astype(np.bool)
            detector_array = np.logical_not(outer_segment == inner_segment)
        new_shape = list(self.grid_steps) + list(self.sampling)
        self.probes = self.probes.reshape(new_shape)
        intgr_cbed = np.sum(self.probes * detector_array, axis=(2,3))
        return intgr_cbed, detector_array

class MSAHybrid(MSA):
    '''
    Class that performs potential building on CPU and beam propagation on GPU using scikit-cuda cufft interface. 
    '''
    def plan_simulation(self, num_probes=None):
        if num_probes is None:
            num_probes = self.num_probes
        self.print_verbose('Simulation requested %d probes simultaneously.' % self.num_probes)
        free_mem, tot_mem = cuda.mem_get_info()
        free_mem = free_mem/1024e6  # in GB
        mem_alloc = num_probes * np.prod(self.sampling) * 8 / 1024e6 + self.potential_slices.nbytes / 1024e6
        self.print_verbose('mem_alloc: %2.3f' % mem_alloc)
        self.print_verbose('free_mem: %2.3f' % free_mem)
        self.max_probes = num_probes
        mod = False
        while mem_alloc > 0.5 * free_mem:
            mod = True
            num_probes = int(num_probes * 0.75)
            mem_alloc = num_probes * np.prod(self.sampling) * 8 / 1024e6 + self.potential_slices.nbytes / 1024e6
            self.max_probes = num_probes
        if mod: self.print_verbose('Device can hold at most %d probes simultaneously.' % self.max_probes)

    def multislice(self, save_probes=True, bandwidth=1/3):
        # not supporting a single probe!!
        # check for potential slices
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
        self.print_verbose('Spent %2.4f s building %d probes on cpu' % (sim_t, self.max_probes))

        # Copy over to device
        trans_gpu = gpuarray.to_gpu_async(slices)
        mask_propag_gpu = gpuarray.to_gpu_async(mask * propag)
        probes_gpu = gpuarray.to_gpu_async(self.probes)

        # Setup fft plans
        # TODO: tile multiple fft plans
        t = time()
        plan = skfft.Plan(propag.shape, np.complex64, np.complex64, batch=self.max_probes)
        for slice_num in range(trans_gpu.shape[0]):
            for z_slice in range(probes_gpu.shape[0]):
                probes_gpu[z_slice] *= trans_gpu[slice_num]
            cufft.fft(probes_gpu, probes_gpu, plan, True)
            for z_slice in range(probes_gpu.shape[0]):
                probes_gpu[z_slice] *= mask_propag_gpu
            cufft.ifft(probes_gpu, probes_gpu, plan, False)
        cufft.fft(probes_gpu, probes_gpu, plan, True)  #return probe wavefunctions in reciprocal space
        sim_t = time() - t
        self.print_verbose('Propagated %d probes in %2.4f s' % (self.max_probes, sim_t))
        self.probes = probes_gpu.get()

        # free up device memory
        trans_gpu.gpudata.free()
        mask_propag_gpu.gpudata.free()
        probes_gpu.gpudata.free()

        # destroy cufft object
        cufft.cufftDestroy(plan.handle)
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
    '''
    Class that implements MSA on a single GPU. 
    Potential Building + Beam Propagation are done on the GPU using JIT-compiled CUDA-C Kernels.
    '''

    def setup_device(self, gpu_id=0):
        global ctx
        cuda.init()
        dev = cuda.Device(gpu_id)
        ctx = dev.make_context()

        import atexit
        def _clean_up():
            if ctx is not None:
                try:
                    ctx.pop()
                    ctx.detach()
                except Exception as e:
                    warn(format(e))
            from pycuda.tools import clear_context_caches
            clear_context_caches()
        atexit.register(_clean_up)
        self.ctx = ctx
        return ctx 

    @staticmethod
    def clean_up(ctx=None, vars=None):
        if vars is not None:
            for var in vars:
                if var is not None: 
                    var.free()
        if ctx is not None:
            ctx.pop()
            ctx.detach()
            ctx = None
            from pycuda.tools import clear_context_caches
            clear_context_caches()

    def integrate_potential_slices(self, pot, grid, slice_thickness, output=False):
        pot_slices = super(MSAGPU, self).integrate_potential_slices(pot, grid, slice_thickness, output=True)
        pot_slices_phase = np.exp(1.j * self.sigma * pot_slices).astype(np.complex64)
        self.potential_slices = cuda.register_host_memory(pot_slices_phase)
        potential_slices_d = cuda.to_device(self.potential_slices)
        # store device allocation ref for later use
        self.pot_dev_ptr = potential_slices_d
        self.vars = []
        self.vars.append(self.potential_slices.base)
        if output:
            return pot_slices

    def build_potential_slices(self, slice_thickness):
        # self.ctx = ctx
        # find number of slices and atomic sites per slice
        self.slice_t = slice_thickness
        self.num_slices = np.int32(np.floor(self.dims[-1] / slice_thickness))
        masks = [np.logical_and(self.supercell_xyz[:, -1] >= slice_num * self.slice_t,
                        self.supercell_xyz[:, -1] < (slice_num + 1) * self.slice_t)
                        for slice_num in range(self.num_slices)]
        # remap Z numbers to indices in cached atomic potential dictionary
        unique_Z = np.unique(self.supercell_Z)
        supercell_Z_idx = np.array([np.argmax(np.equal(unique_Z, Z_val)) for Z_val in self.supercell_Z])
        Z_arr = np.array([supercell_Z_idx[mask] for mask in masks])
        supercell_pix = self.supercell_xyz[:,:2]/self.pix_size[::-1]
        yx_arr = np.array([supercell_pix[mask] for mask in masks])

        # stack the input for all slices into a single 1-d array
        Zxy_input = np.array([np.column_stack([Z, yx[:,1], yx[:,0]]).astype(np.int32).flatten()
                                            for Z, yx in zip(Z_arr, yx_arr)])

        # pad sites array to have equal num of element per slice
        max_shape = np.max(np.array([itm.shape for itm in Zxy_input]))
        pads = [(0, max_shape - itm.shape[0]) for itm in Zxy_input]
        # padded = [np.pad(itm,pad,'constant',constant_values=-1) for pad, itm in zip(pads, Zxy_input)]
        Zxy_input = np.vstack([np.pad(itm, pad, 'constant', constant_values=-1) for pad, itm in zip(pads, Zxy_input)])
        Zxy_input = Zxy_input.astype(np.float32)
        
        # stack atomic potentials of unique elements
        atom_pot_stack = np.array([self.cached_pots[uq_Z] for uq_Z in unique_Z]).astype(np.float32)

        # compile/load cuda kernels
        self._load_potential_kernels([atom_pot_stack.shape[1], atom_pot_stack.shape[2]],
                                                Zxy_input.shape[-1])

        # allocate memory
        atom_pot_stack_d = cuda.to_device(atom_pot_stack)
        sites_d = cuda.to_device(Zxy_input)
        self.potential_slices = cuda.aligned_zeros((int(self.num_slices),
                    int(self.sampling[0]), int(self.sampling[1])), np.complex64)
        self.vars = []
        self.vars.append(self.potential_slices.base)
        self.potential_slices = cuda.register_host_memory(self.potential_slices)
        potential_slices_d = cuda.to_device(self.potential_slices)

        # store device allocation ref for later use
        self.pot_dev_ptr = potential_slices_d

        # get block, grid dimensions
        block, grid = self._get_blockgrid([self.sampling[1], self.sampling[0], self.num_slices],
                    mode='3D')
        build_potential = self.pot_kernels['build_potential']
        grid = (grid[0], grid[1], grid[2]*2)
        # print("block:%s, grid:%s" %(format(block), format(grid)))
        # print("max, row idx:%d, col idx:%d, stk idx:%d" %(block[0]*grid[0], block[1]*grid[1], block[2]*grid[2]))
        # print("sites: %s" %format(Zxy_input.shape))
        # build potential
        build_potential(potential_slices_d, atom_pot_stack_d, sites_d,
                        np.float32(self.sigma), block=block, grid=grid)
        self.ctx.synchronize()
        cuda.memcpy_dtoh_async(self.potential_slices, potential_slices_d, cuda.Stream())
        self.ctx.synchronize()
        # free gpu memory
        sites_d.free()
        atom_pot_stack_d.free()
        self.print_verbose('Built %d potential slices with shape:%s pixels' % (self.potential_slices.shape[0],
                                                                  format(self.potential_slices.shape[1:])))
        # unregister host memory
        #self.potential_slices.base.unregister()
        #self.potential_slices.base.free()

    def _load_kernels(self):
        try:
            probe_kernels = ProbeKernels(sampling=self.sampling)
            self.kernels = probe_kernels.kernels
            self.print_verbose('CUDA C/C++ Probe Kernels compiled successfully')
        except cuda.CompileError:
            warn('CUDA C/C++ Probe Kernels did not compile successfully')
            raise cuda.CompileError

    def _load_potential_kernels(self, potential_shape, sites_size ):
        try:
            potential_kernels = PotentialKernels(sampling=self.sampling,
            num_slices=self.num_slices, potential_shape=potential_shape,
            sites_size=sites_size)
            self.pot_kernels = potential_kernels.kernels
            self.print_verbose('CUDA C/C++ Potential Kernels compiled successfully')
        except cuda.CompileError:
            warn('CUDA C/C++ Potential Kernels did not compile successfully')
            raise cuda.CompileError

    def build_probe(self, probe_dict={'smooth_apert': True, 'apert_smooth': 50, 'spherical_phase': True,
                                      'aberration_dict': {'C1': 0., 'C3': 0., 'C5': 0.},'scherzer': True}):
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
        #self.apert = cuda.register_host_memory(self.apert)
        apert_d = cuda.mem_alloc(self.apert.nbytes)
        self.psi_k = np.empty(self.sampling, dtype=np.complex64)
        #self.psi_k = cuda.register_host_memory(self.psi_k)
        self.psi = np.empty_like(self.psi_k)
        #self.psi = cuda.register_host_memory(self.psi)
        psi_k_d = cuda.mem_alloc(self.psi_k.nbytes)
        psi_x_d = cuda.mem_alloc(self.psi_k.nbytes)

        # build a probe in k-space
        apert_func(apert_d, k_max, k_semi, shape_x, shape_y, block=block, grid=grid, shared=0, stream=cuda.Stream())
        #ctx.synchronize()
        phase_func(psi_k_d, k_max, Lambda, C1, C3, C5, shape_x, shape_y, block=block, grid=grid, shared=0, stream=cuda.Stream())
        #ctx.synchronize()
        self.multwise_2d_func(psi_k_d, apert_d, shape_x, shape_y, block=block, grid=grid)
        #ctx.synchronize()
        cuda.memcpy_dtoh_async(self.psi_k, psi_k_d, cuda.Stream())
        cuda.memcpy_dtoh_async(self.apert, apert_d, cuda.Stream())

        # build probe in x-space
        fft_plan = skfft.Plan(self.psi_k.shape, np.complex64, np.complex64, batch=1)
        cufft.cufftExecC2C(fft_plan.handle, int(psi_k_d), int(psi_x_d), cufft.CUFFT_INVERSE)
        self.ctx.synchronize()
        fftshift_func(psi_x_d, shape_y, block=block, grid=grid)
        #ctx.synchronize()
        cuda.memcpy_dtoh_async(self.psi, psi_x_d, cuda.Stream())
        self.normalization = np.sum(np.abs(self.psi, dtype=np.float64)**2)
        self.psi /= np.sqrt(self.normalization)
        cuda.memcpy_dtoh_async(self.psi_k, psi_k_d, cuda.Stream())

        # free gpu mem
        psi_x_d.free()
        psi_k_d.free()
        apert_d.free()
        cufft.cufftDestroy(fft_plan.handle)
        
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
            grid_xsize = int(round((shape_x + blk_xsize - 1) / blk_xsize))
            grid_ysize = int(round((shape_y + blk_ysize - 1) / blk_ysize))
            grid_zsize = int(round((shape_z + blk_zsize - 1) / blk_zsize))
            block_3d = (blk_xsize, blk_ysize, blk_zsize)
            grid_3d = (grid_xsize, grid_ysize, grid_zsize)
            return block_3d, grid_3d
        if mode == '2D':
            shape_x = shapes[0]
            shape_y = shapes[1]
            block_2d = (32, 32, 1)
            grid_2d = (int(round((shape_x + block_2d[0] - 1) / 32)), int(round((shape_y + block_2d[1] - 1) / 32)), 1)
            return block_2d, grid_2d
        if mode == '1D':
            shape_x = shapes[0]
            shape_y = shapes[1]
            shape_z = shapes[2]
            block_1d = (int(min(shape_x * shape_y, 1024)), 1, 1)
            grid_1d = (int((shape_x * shape_y) / block_1d[0]), int(shape_z), 1)
            return block_1d, grid_1d

    def multislice(self, bandwidth=1/3, unified_mem=False, batch_size=None, transmit=True):
        """

        :param bandwidth:
        :return:
        """
        ## TODO: when batch_size != self.num_probes, probe positions get scrambled!!!
        if batch_size == None:
            batch_size = self.num_probes 
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
        shape_x = np.int32(self.sampling[1])
        shape_y = np.int32(self.sampling[0])
        num_probes = self.max_probes
        k_semi = np.float32(self.semi_ang / self.Lambda)

        # define block/grid threads
        block_3d, grid_3d = self._get_blockgrid([self.sampling[1], self.sampling[0], num_probes], mode='3D')
        block_2d, grid_2d = self._get_blockgrid([self.sampling[1], self.sampling[0], num_probes], mode='2D')
        self.print_debug('block, grid:', block_3d, grid_3d)

        # allocate memory
        self.propag = cuda.aligned_zeros((int(self.sampling[0]), int(self.sampling[1])), np.complex64)
        self.vars.append(self.propag.base)
        self.propag = cuda.register_host_memory(self.propag)
        propag_d = cuda.to_device(self.propag)
        self.mask = cuda.aligned_zeros((int(self.sampling[0]), int(self.sampling[1])), np.float32)
        self.vars.append(self.mask.base)
        self.mask = cuda.register_host_memory(self.mask)
        mask_d = cuda.to_device(self.mask)
        grid_steps_d = cuda.to_device(self.grid_steps.astype(np.int32))
        grid_range_d = cuda.to_device(self.grid_range.astype(np.float32))
        psi_k_d = cuda.to_device(self.psi_k)
        if unified_mem:
            self.probes = cuda.managed_empty(shape=(int(self.num_probes), int(self.sampling[0]), int(self.sampling[1])),
                                        dtype=np.complex64, mem_flags=cuda.mem_attach_flags.GLOBAL)
        else:
            # pinned memory is default
            self.probes = cuda.aligned_zeros((int(self.num_probes), int(self.sampling[0]), int(self.sampling[1])), np.complex64)
            self.vars.append(self.probes.base)
            self.probes = cuda.register_host_memory(self.probes)
        ones = cuda.aligned_zeros((int(self.sampling[0]), int(self.sampling[1])), np.complex64) + 1
        self.vars.append(ones.base)
        ones = cuda.register_host_memory(ones)
        ones_d = cuda.mem_alloc(ones.nbytes)
        cuda.memcpy_htod_async(ones_d, ones, cuda.Stream())
        
        # grab needed kernels
        propag_func = self.kernels['propagator']
        mask_func = self.kernels['hard_aperture']
        multwise_func = self.kernels['mult_wise_c2d_re2d']

        # 1. Build propagator and bandwidth limiting mask
        mask_func(mask_d, np.float32(self.kmax), np.float32(bandwidth * self.kmax), shape_x, shape_y, block=block_2d,
                  grid=grid_2d, shared=0, stream=cuda.Stream())
        #ctx.synchronize()
        propag_func(propag_d, np.float32(self.kmax), np.float32(self.slice_t), np.float32(self.Lambda), shape_x, shape_y,
                    block=block_2d, grid=grid_2d, shared=0, stream=cuda.Stream())
        #ctx.synchronize()
        multwise_func(propag_d, mask_d, shape_x, shape_y, block=block_2d, grid=grid_2d, shared=0)
        cuda.memcpy_dtoh_async(self.propag, propag_d, cuda.Stream())
        cuda.memcpy_dtoh_async(self.mask, mask_d, cuda.Stream())

        # 1. Generate batch fft plans, create host/device pointers, and launch cuda streams
        num_phases = self.num_probes//self.max_probes
        phases = []
        self.results = []
        t = time()
        num_stream=0
        for i in range(num_phases + 1):
            phase = slice(i * self.max_probes, (i+1) * self.max_probes)
            if i == num_phases :
                phase = slice(i * self.max_probes, self.num_probes)
            phases.append(phase)
        self.print_debug('Simulation split into %d serial phases.' % len(phases))
        self.print_debug('Phases: %s ' %format(phases))
        for (i, phase) in enumerate(phases):
            t = time()
            if self.probes[phase].shape[0] == 0: break
            self.print_verbose('Starting simulation phase #%d' % i)
            num_batches = (phase.stop - phase.start + 1)//batch_size
            batches = []
            streams = []
            probes_d = []
            norm_consts = []
            plans = []
        # 2. Generate batch fft plans and create pinned memory pointers and cuda streams
            for batch_num in range(num_batches+1):
                if batch_num == num_batches:
                    slice_obj = slice(batch_num * batch_size, phase.stop - phase.start)
                else:
                    slice_obj = slice(batch_num * batch_size, (batch_num + 1) * batch_size)
                batches.append(slice_obj)
                stream = cuda.Stream()
                num_stream += 1
                self.print_debug(' Launch Stream # %d' % num_stream)
                streams.append(stream)
                num_probes = np.int32(self.probe_positions[phase][slice_obj].shape[0])
                if num_probes == 0: break
                probes_d.append(cuda.mem_alloc(int(num_probes*np.prod(self.sampling)*8)))
                plans.append(skfft.Plan(self.sampling, np.complex64, np.complex64, batch=num_probes, stream=stream))
                norm_consts.append(cuda.mem_alloc(np.empty(num_probes,dtype=np.float32).nbytes))
            self.print_debug('Batches: %s' % format(batches))
        # 3. Propagate Beams
            self.print_verbose("Simulating probes %d out of %d..." % (phase.stop, self.num_probes))
            for batch, stream, probe_d, plan,  norm_const in zip(batches, streams, probes_d, plans, norm_consts):
                num_probes = np.int32(self.probe_positions[phase][batch].shape[0])
                grid_range = self.probe_positions[phase][batch].astype(np.float32) 
                grid_range_d = cuda.mem_alloc(grid_range.nbytes)
                cuda.memcpy_htod_async(grid_range_d, grid_range, stream)
                self.print_debug('batch: %s, stream: %s' % (format(batch), format(stream)))
                self.__propagate_beams(num_probes, batch, probe_d, propag_d, psi_k_d, norm_const, grid_steps_d, grid_range_d,
                                     self.probes[phase][batch], plan, ones_d, stream, transmit=transmit)
            self.ctx.synchronize()
        # 4. clean-up
            for plan, probe_d, norm_const in zip(plans, probes_d, norm_consts):
               cufft.cufftDestroy(plan.handle)
               probe_d.free()
               norm_const.free()
               # ctx.synchronize()
               del probe_d, norm_const, plan
            self.ctx.synchronize()
            self.print_verbose('finished simulation phase #%d' % i)
            # self.probes /= self.normalization
            if transmit:
                self.probes[phase] /= self.normalization
            else:
                self.probes[phase] /= np.sqrt(self.normalization)
            sim_t = time()-t
            self.print_verbose('Propagated %d probes in %2.4f s' % (self.probe_positions[phase].shape[0], sim_t))
        
        if transmit:
            with catch_warnings():
                simplefilter('ignore')
                catch_warn()
                self.probes = self.probes.astype(np.float32) # discard imaginary
        # clean up device variables
        self.vars.append(self.pot_dev_ptr) # don't de-allocate here to allow for simulations with different probe/exp. params
        mask_d.free()
        propag_d.free()
        psi_k_d.free()
        grid_range_d.free()
        grid_steps_d.free()
        ones_d.free()
        return None

    def __propagate_beams(self, num_probes, batch, psi_pos_d, propag_d, psi_k_d,
                        norm_const_d, grid_steps_d, grid_range_d,
                        psi_x_pos_pin, fft_plan_probe, ones_d, stream, transmit=True):
        """
        :param batch:
        :param psi_pos_d:
        :param propag_d:
        :param psi_k_d:
        :param norm_const_d:
        :param grid_steps_d:
        :param grid_range_d:
        :param psi_x_pos_pin:
        :param fft_plan_probe:
        :param stream:
        :return:
        """
        # define block/grid threads
        block_3d, grid_3d = self._get_blockgrid([self.sampling[1], self.sampling[0], num_probes], mode='3D')
        block_1d, grid_1d = self._get_blockgrid([self.sampling[1], self.sampling[0], num_probes], mode='1D')
        self.print_debug('block, grid:', block_3d, grid_3d)

        # grab needed kernels
        probe_stack_func = self.kernels['probes_stack']
        multwise2d_stack_func = self.kernels['mult_wise_c3d_c2d']
        multwise_stack_func = self.kernels['mult_wise_c3d_c3d_ind']
        multwise_stack_func.prepare(("P", "P", np.int32, np.int32, np.float32))
        multwise2d_stack_func.prepare(("P", "P", np.int32, np.float32))
        fftshift_func = self.kernels['fftshift_2d_stack']
        # norm_const_func = self.kernels['norm_const']
        normalize_func = self.kernels['normalize']
        mod_square_func = self.kernels['mod_square_stack']

        #1. build probes
        probe_stack_func(psi_pos_d, psi_k_d, num_probes, np.float32(self.kmax), grid_steps_d, grid_range_d,
                         block=block_3d, grid=grid_3d, shared=0, stream=stream)
        #ctx.synchronize()
        cufft.cufftExecC2C(fft_plan_probe.handle, int(psi_pos_d), int(psi_pos_d), cufft.CUFFT_INVERSE)
        #ctx.synchronize()
        fftshift_func(psi_pos_d, num_probes, block=block_3d, grid=grid_3d, shared=0, stream=stream)
        #ctx.synchronize()
        ### TODO: Check why cuda normalization is off

        # 2. Propagate probes through atomic potential
        # ctx.synchronize()
        if transmit:
            for i in range(self.num_slices):
                # self.print_debug('Atomic potential slice #%d' % i)
                multwise_stack_func.prepared_async_call(grid_3d, block_3d, stream, psi_pos_d, self.pot_dev_ptr, num_probes, np.int32(i),
                                                        np.float32(1/np.prod(self.sampling)))
                #ctx.synchronize()
                cufft.cufftExecC2C(fft_plan_probe.handle, int(psi_pos_d), int(psi_pos_d), cufft.CUFFT_FORWARD)
                #ctx.synchronize()
                multwise2d_stack_func.prepared_async_call(grid_3d, block_3d, stream, psi_pos_d, propag_d, num_probes,
                                                        np.float32(1.0))
                #ctx.synchronize()
                cufft.cufftExecC2C(fft_plan_probe.handle, int(psi_pos_d), int(psi_pos_d), cufft.CUFFT_INVERSE)
                #ctx.synchronize()


            multwise2d_stack_func.prepared_async_call(grid_3d, block_3d, stream, psi_pos_d, ones_d, num_probes,
                                                    np.float32(1 / (np.prod(self.sampling))))
            cufft.cufftExecC2C(fft_plan_probe.handle, int(psi_pos_d), int(psi_pos_d), cufft.CUFFT_FORWARD)
            multwise2d_stack_func.prepared_async_call(grid_3d, block_3d, stream, psi_pos_d, ones_d, num_probes,
                                                    np.float32(np.sqrt(np.prod(self.sampling))))
            mod_square_func(psi_pos_d, num_probes, block=block_3d, grid=grid_3d, stream=stream)
        # ctx.synchronize()
        cuda.memcpy_dtoh_async(psi_x_pos_pin, psi_pos_d, stream=stream)

class MSAMPI(MSAGPU):
    '''
    Class that extends MSAGPU to multiple GPUs and/or nodes.
    The parallel distribution falls back on hdf5 writing of the results if the memory associated with the MPI root rank is exceeded.
    '''
    def __init__(self, *args, **kwargs):
        #global comm
        comm = MPI.COMM_WORLD
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        super(MSAMPI, self).__init__(*args, **kwargs)

    def print_rank(self, *args, **kwargs):
        if self.rank == 0:
            print(*args, **kwargs)

    def print_verbose(self, *args, **kwargs):
        if self.verbose:
            self.print_rank(*args, **kwargs)

    def generate_probe_positions(self, *args, **kwargs):
        super(MSAMPI, self).generate_probe_positions(*args, **kwargs)
        data_parts = []
        part = self.probe_positions.shape[0] // self.size
        self.total_num_probes = self.probe_positions.shape[0]
        for rank in range(self.size):
            slice_obj = slice(rank * part, (rank + 1)* part)
            if rank == self.size - 1:
                slice_obj = slice( rank * part, None)
            data_parts.append(slice_obj)
        self.data_parts = data_parts
        if self.rank == 0:
            data = [self.probe_positions[part] for part in data_parts]
            self.data_size = np.sum([dat.shape[0] for dat in data])
        else:
            data = None
        data = comm.scatter(data, root=0)
        self.probe_positions = data
        print("rank %d: my # of probe_positions %d" % (self.rank,
        self.probe_positions.shape[0]))
        self.num_probes = np.int32(self.probe_positions.shape[0])

    def multislice(self, *args, **kwargs):
        try:
            # h5_write= kwargs.pop('h5_write')
            h5_file = kwargs.pop('h5_write')
            h5_write = isinstance(h5_file, h5py.File)
        except KeyError:
            h5_write = False
        super(MSAMPI, self).multislice(*args, **kwargs)
        if h5_write:
            self.print_rank('writing to h5 file...')
            dset = h5_file.create_dataset('4D_CBED', (self.total_num_probes,
                                self.sampling[0], self.sampling[1]), dtype=self.probes.dtype)
            # Workaround the large parallel I/O limitation
            ### TODO: (low priority) test if doing below in chunks over indices < 2GB speeds up the writing.
            for i, p in enumerate(range(*self.data_parts[self.rank].indices(self.total_num_probes))):
                dset[p, :, :] = self.probes[i, :, :]
            self.print_rank('finished writing h5 file.')
        else:
            receive_buff = None
            self.print_verbose('rank %d: shape of calculated probes = %s' %(self.rank, format(self.probes.shape)))
            if self.rank == 0:
                buff_shape = (self.total_num_probes, self.sampling[0], self.sampling[1])
                self.print_rank('receive buffer shape: %s' %format(buff_shape))
                receive_buff = np.empty(buff_shape, dtype=self.probes.dtype)
            # Calculate buffer size for a vector gather op
            count = []
            for data_part in self.data_parts:
                start = data_part.start
                if start is None:
                    start = 0
                stop = data_part.stop
                if stop is None:
                    stop = self.total_num_probes
                count.append(stop - start)
            count = np.array(count)
            displ = np.cumsum(count)
            displ = np.append(displ[::-1],0)[::-1]
            xy_offset = np.prod(self.sampling)
            count *= xy_offset
            count = count.astype(np.int64)
            displ *= xy_offset
            displ = displ.astype(np.int64)
            self.print_verbose("rank %d: my # probes: %s %s, count: %d, displ: %d" %(self.rank, format(self.probes.shape), self.probes.dtype, count[self.rank], displ[self.rank]))
            # gather
            # comm.Gatherv(self.probes, [receive_buff, count, displ[:-1], MPI.C_FLOAT_COMPLEX], root=0)
            comm.Gatherv(self.probes, [receive_buff, count, displ[:-1], MPI.FLOAT], root=0)


            if self.rank == 0: self.print_rank('Gathered results of shape: %s' %format(receive_buff.shape))
