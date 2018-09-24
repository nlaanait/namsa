from .database import kirkland_params
from .optics import voltage2Lambda, sigma_int, spherical_phase_error
from .utils import *
import numpy as np
from scipy.special import k0
import multiprocessing as mp
import sys
from warnings import warn
import os


XYZ_dtype = [('atomic_number', 'i'), ('x', 'f'), ('y', 'f'), ('z', 'f'), ('occ', 'f'), ('DW', 'f')]


def unwrap(args):
    (msa, params), kwargs = args
    method_name = kwargs.pop('method')
    if method_name == 'build_slices':
        return MSA.make_slice(msa, params)
    elif method_name == 'propagate_probes':
        return MSA.propagate_probes(msa, params, kwargs)


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
            self.sampling = np.floor(self.kmax * self.dims[:2])
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
        step_x, step_y = 1.j * (np.floor(potential_range/self.pix_size)) * oversample
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
        with mp.Pool(processes=processes, maxtasksperchild=1) as pool:
            jobs = pool.imap(unwrap, tasks, chunksize=chunk)
            potential_slices = np.array([j for j in jobs])

        self.potential_slices = potential_slices

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

    def build_probe(self, probe_position=np.array([0.,0.]), smooth_apert=True, apert_smooth=50, spherical_phase=True,
                    aberration_args={'C1':0., 'C3': 0., 'C5': 0.}, scherzer=True):

        k_y, k_x = np.mgrid[-self.kmax/2: self.kmax/2: 1.j*self.sampling[0],
                            -self.kmax/2: self.kmax/2: 1.j * self.sampling[1]]
        k_rad = np.sqrt(k_x ** 2 + k_y ** 2)
        k_semi = self.semi_ang/self.Lambda

        # aperture function
        aperture = np.heaviside(k_semi - k_rad, 0.5)
        if smooth_apert:
            aperture = 1 / (1 + np.exp(-2 * apert_smooth * (k_semi - k_rad)))

        # aberration
        if spherical_phase:
            phase_error = spherical_phase_error(k_rad, self.Lambda, scherzer, **aberration_args)
        else:
            pass
            #TODO: implement non-rotationally invariant phase error

        # probe wavefunction
        psi_k = aperture * phase_error
        y, x = probe_position
        kr = k_x * x + k_y * y
        phase_shift = np.exp(2 * np.pi * 1.j * kr)
        psi_x = np.fft.ifft2(psi_k * phase_shift, norm='ortho')
        psi_x = np.fft.fftshift(psi_x)
        psi_x /= np.sqrt(np.sum(np.abs(psi_x) ** 2))
        return psi_x, psi_k, aperture



