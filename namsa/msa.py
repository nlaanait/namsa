from .database import kirkland_params
from .optics import voltage2Lambda, sigma_int, spherical_phase_error
from .utils import *
import numpy as np
from scipy.special import k0
import multiprocessing as mp
import ctypes
import sys
from warnings import warn
import os
import pyfftw


XYZ_dtype = [('atomic_number', 'i'), ('x', 'f'), ('y', 'f'), ('z', 'f'), ('occ', 'f'), ('DW', 'f')]


def unwrap(args):
    (msa, params), kwargs = args
    method_name = kwargs.pop('method')
    if method_name == 'build_slices':
        return MSA.make_slice(msa, params)
    elif method_name == 'propagate_beams':
        return MSA.propagate_beam(msa, params)


class MSA(object):
    def __init__(self, energy, semi_angle, supercell, sampling=np.array([512, 512]), max_angle=None):
        self.E = energy
        self.Lambda = voltage2Lambda(self.E*1e3)
        self.semi_ang = semi_angle

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
        print('Simulation Parameters:\nSupercell dimensions xyz:%s (Å)\nReal, Reciprocal space pixel sizes:%s Å, %s 1/Å'
              '\nMax angle: %2.2f (rad)\nSampling in real and reciprocal space: %s pixels' %
              (format(np.round(self.dims, 2)), format(np.round(self.pix_size, 2)), format(np.round(self.kpix_size, 2)),
               self.max_ang, format(self.sampling)))

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
        tasks = [((self, slice_num), {'method':'build_slices'}) for slice_num in range(num_slices)]
        processes = min(mp.cpu_count(), num_slices)
        chunk = np.int(np.floor(num_slices / processes))
        pool = mp.Pool(processes=processes, maxtasksperchild=1)
        jobs = pool.imap(unwrap, tasks, chunksize=chunk)
        potential_slices = np.array([j for j in jobs])
        pool.close()
        self.potential_slices = potential_slices.astype(np.float32)
        print('Built potential slices with shape:%s' % format(self.potential_slices.shape))

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

    def build_probe(self, probe_position=np.array([0., 0.]), probe_dict={'smooth_apert': True, 'apert_smooth': 50,
                                                                        'spherical_phase': True, 'aberration_dict':
                                                                            {'C1': 0., 'C3': 0., 'C5': 0.},
                                                                        'scherzer': True}):
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

    def multi_slice(self, probe_pos=np.array([0.,0.]), probe_grid=True, return_probes=True, bandwidth=1/3):
        # check for slices
        if isinstance(self.potential_slices, np.ndarray) is False:
            warn('Potential slices must be calculated first before calling multi_slice')
            return
        # check for probe
        if isinstance(self.probe_dict, dict) is False:
            warn('Probe wave function must be initialized first before calling multi_slice')
            return
        if probe_grid:
            # Put the potential slices in shared memory so all workers access it
            global shared_slices
            shared_slices = mp.Array(ctypes.c_float, self.potential_slices.size, lock=False)
            temp = np.frombuffer(shared_slices, dtype=np.float32)
            for (i, pot) in enumerate(self.potential_slices):
                temp[i * pot.size:(i + 1) * pot.size] = pot.flatten().astype(np.float32)

            tasks = [((self, (pos, return_probes, probe_grid, bandwidth)), {'method': 'propagate_beams'})
                     for pos in self.probe_positions]
            processes = min(mp.cpu_count(), self.probe_positions.shape[0])
            chunk = np.floor(self.probe_positions.shape[0] / processes).astype(np.int)
            pool = mp.Pool(processes=processes, initargs=(shared_slices,))
            jobs = pool.imap(unwrap, tasks, chunksize=chunk)
            trans_probes = np.array([j for j in jobs])
            pool.close()
            return trans_probes
        else:
            trans_probes = self.propagate_beam([probe_pos, return_probes, probe_grid, bandwidth])
            return trans_probes

    def propagate_beam(self, args):
        probe_pos, return_probes, probe_grid, bandwidth = args
        propag = self.build_propagator()
        blim_mask = self.bandwidth_limit_mask(propag.shape, radius=bandwidth)
        probe, _, _ = self.build_probe(probe_pos, self.probe_dict)
        probes = []
        probe_last = probe
        if probe_grid:
            slices = np.frombuffer(shared_slices, dtype=np.float32)
            slices = slices.reshape(self.potential_slices.shape)
        else:
            slices = self.potential_slices
        slices = np.exp(1.j * self.sigma * slices).astype(np.complex64)
        pyfftw.interfaces.cache.enable()
        for (i, trans) in enumerate(slices[::-1]):
            t_psi = pyfftw.interfaces.numpy_fft.fft2(trans * probe_last, overwrite_input=True) * blim_mask
            probe_next = pyfftw.interfaces.numpy_fft.ifft2(propag * t_psi, overwrite_input=True)
            if return_probes:
                probes.append(probe_next)
            probe_last = probe_next
        #         if verbose and not bool(i%25):
        #             print('calculating slice %d/%d:'%(i,slices.shape[0]))
        #             print('integrated scattered intensity: %2.4f' %np.sum(np.abs(probe_next)**2))

        if return_probes:
            probes = np.array(probes)
            print('finished with probe position: %s' % format(probe_pos))
            return probes
        else:
            return probe_next

