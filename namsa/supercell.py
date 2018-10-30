import numpy as np
from pymatgen.core import Element
from pymatgen.io.cif import CifParser
from itertools import chain
from warnings import warn


class Centerings(object):
    def __init__(self, centering='P', primitive=False):
        self.centering = centering
        conventional_transformations = {'I': np.array([[-1, 1, 1],
                                           [1, -1, 1],
                                           [1, 1, -1]]),
                           'A': np.array([[1, 0, 0],
                                           [0, 1, -1],
                                           [0, 1, 1]]),
                           'C': np.array([[1, 1, 0],
                                           [-1, 1, 0],
                                           [0, 0, 1]]),
                           'R': np.array([[2, -1, -1],
                                           [1, 1, -2],
                                           [1, 1, 1]]),
                           'F': np.array([[0, 1, 1],
                                           [1, 0, 1],
                                           [1, 1, 0]]),
                           'P': np.array(np.identity(3)),
                           }
        primitive_transformations = {'I': 0.5 * np.array([[-1, 1, 1],
                                              [1, -1, 1],
                                              [1, 1, -1]]),
                        'A': 0.5 * np.array([[1, 0, 0],
                                              [0, 1, -1],
                                              [0, 1, 1]]),
                        'C': 0.5 * np.array([[1, 1, 0],
                                              [-1, 1, 0],
                                              [0, 0, 1]]),
                        'R': 1. / 3 * np.array([[2, -1, -1],
                                                 [1, 1, -2],
                                                 [1, 1, 1]]),
                        'F': 0.5 * np.array([[0, 1, 1],
                                              [1, 0, 1],
                                              [1, 1, 0]]),
                        'P': np.array(np.identity(3)),
                        }

        self.transformations = conventional_transformations

        if primitive:
            self.transformations = primitive_transformations

    @property
    def trans_mat(self):
        return self.transformations[self.centering]


class SupercellBuilder(object):
    """
    Build supercell out of cif file
    """
    def __init__(self, cif_file, primitive=False, debug=False, verbose=False):
        """

        :param cif_file: string, path to cif file
        :param primitive: bool, if primitive unit cell should be used
        """

        # housekeeping
        self.debug = debug
        self.verbose = verbose
        # dtypes for numpy structured arrays
        self.sites_dtype = [('atom_type', '|U16'), ('atomic_number', 'i'), ('frac_x', 'f'), ('frac_y', 'f'),
                            ('frac_z', 'f'), ('occ', 'f'), ('DW', 'f')]
        self.supercell_sites_dtype = [('atom_type', '|U16'), ('atomic_number', 'i'), ('x', 'f'), ('y', 'f'), ('z', 'f'),
                                      ('occ', 'f'), ('DW', 'f')]
        self.xyz_dtype = [('atom_type', '|U16'), ('x', 'f'), ('y', 'f'), ('z', 'f')]
        self.XYZ_dtype = [('atomic_number', 'i'), ('x', 'f'), ('y', 'f'), ('z', 'f'), ('occ', 'f'), ('DW', 'f')]
        # initializing matrices
        self.p_mat = np.identity(3)
        self.P_mat = np.identity(4)
        self.Q_mat = np.identity(4)

        # parse cif
        cifparser = CifParser(cif_file)

        # get lattice and symmetry operations matrices
        self.structure = cifparser.get_structures(primitive=primitive)[0]
        self.basis = np.column_stack([self.structure.lattice.matrix[0], self.structure.lattice.matrix[1],
                                      self.structure.lattice.matrix[2]])
        self.lattice_const = np.array([self.structure.lattice.a, self.structure.lattice.b, self.structure.lattice.c])
        self.lattice_angles = np.array([self.structure.lattice.alpha, self.structure.lattice.beta,
                                        self.structure.lattice.gamma])
        ops = cifparser.symmetry_operations
        self.sym_ops = [op.affine_matrix for op in ops]

        # Get sites
        dic = cifparser.as_dict()
        data = [dic[key] for key in dic.keys()][0]
        frac_x = np.array(data['_atom_site_fract_x'], dtype=np.float)
        frac_y = np.array(data['_atom_site_fract_y'], dtype=np.float)
        frac_z = np.array(data['_atom_site_fract_z'], dtype=np.float)
        frac_coord = np.column_stack((frac_x, frac_y, frac_z))
        site_labels = np.array(data['_atom_site_label'])
        site_symbols = np.array(data['_atom_site_type_symbol'])
        site_occps = np.array(data['_atom_site_occupancy']).astype(np.float)
        try:
            DWs = np.array(data['_atom_site_B_iso_or_equiv'] / (8 * np.pi ** 2))
            DWs = np.array(data['_atom_site_U_iso_or_equiv'])
        except:
            DWs = np.array([0.07 for _ in range(site_labels.size)])
        self.sites = np.array([(atom_type, Element(atom_type).Z, x, y, z, occ, dw) for atom_type, (x, y, z), occ, dw in
                               zip(site_symbols, frac_coord, site_occps, DWs)], dtype=self.sites_dtype)
        self.print_verbose(' Lattice vectors:\n', np.round(self.basis, 4))
        self.print_verbose(' Lattice Constants (Å): \n', np.round(self.lattice_const, 4))
        self.print_verbose(' Lattice Angles (deg.): \n', np.round(self.lattice_angles, 4))
        self.print_verbose(' Volume (Å**3): \n', np.round(self.structure.volume, 4))
        self.print_verbose(' Chemical Formula: \n', self.structure.formula)

    # @property
    # def p_mat(self):
    #     return self.__p_mat
    #
    # @p_mat.setter
    # def p_mat(self, value):
    #     self.__p_mat = value
    #
    # @property
    # def P_mat(self):
    #     return
    #
    # @P_mat.setter
    # def P_mat(self, value):
    #     self.__P_mat = value
    #
    # @property
    # def Q_mat(self):
    #     return
    #
    # @Q_mat.setter
    # def Q_mat(self, value):
    #     self.__Q_mat = value
    #
    # @property
    # def lattice_volume(self):
    #     return
    #
    # @lattice_volume.setter
    # def lattice_volume(self, value):
    #     self.__lattice_volume = value

    def print_debug(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def print_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def transform_unit_cell(self, centering_type='P', primitive=False, scaling=np.array([1, 1, 1]), rotation=None,
                            origin_shift=np.array([0, 0, 0]), max_iter=4):
        """

        :param centering_type:
        :param primitive:
        :param scaling:
        :param rotation:
        :param origin_shift:
        :param max_iter:
        :return:
        """

        # Build transformation matrix (P_mat) and its inverse (Q_mat)
        centering = Centerings(centering_type, primitive=primitive)

        if rotation is not None:
            rotation = rotation
        else:
            rotation = centering.trans_mat

        self.p_mat = np.array(rotation)
        self.p_mat[:, 0] = self.p_mat[:, 0] * scaling[0]
        self.p_mat[:, 1] = self.p_mat[:, 1] * scaling[1]
        self.p_mat[:, 2] = self.p_mat[:, 2] * scaling[2]
        self.P_mat = np.identity(4)
        self.P_mat[:3][:, :3] = self.p_mat
        self.P_mat[:3][:, 3] = origin_shift
        self.Q_mat = np.linalg.inv(self.P_mat)

        # Transform lattice
        self.basis = np.dot(self.basis, self.p_mat)
        self.lattice_const = np.linalg.norm(self.basis, axis=0)
        permut = [[1, 2], [2, 0], [0, 1]]
        self.lattice_angles = np.array([np.arccos(np.dot(self.basis.T[ind_1], self.basis.T[ind_2])
                                                  / (self.lattice_const[ind_1] * self.lattice_const[ind_2]))
                                        for ind_1, ind_2 in permut])
        self.lattice_angles = np.rad2deg(self.lattice_angles)
        self.lattice_volume = np.dot(self.basis[0], np.cross(self.basis[1], self.basis[2]))
        self.print_verbose('Transformed Lattice vectors:\n', np.round(self.basis, 5))
        self.print_verbose('Transformed Lattice Constants (Å): \n', self.lattice_const)
        self.print_verbose('Transformed Lattice Angles (deg.): \n', self.lattice_angles)
        self.print_verbose('Transformed Volume (Å**3): \n', self.lattice_volume)

        # Transform symmetry operations
        self.sym_ops = [np.dot(self.Q_mat, mat).dot(self.P_mat) for mat in self.sym_ops]

        # Find new atomic sites in the new unit-cell
        if centering_type != 'P':
            total_old_sites = 0
            num_iter = 0
            while True:
                n_x, n_y, n_z = scaling.astype(np.int)
                new_sites_l = []
                for site in self.sites:
                    x_range = np.linspace(site['frac_x'], site['frac_x'] + n_x, n_x + 1)
                    y_range = np.linspace(site['frac_y'], site['frac_y'] + n_y, n_y + 1)
                    z_range = np.linspace(site['frac_z'], site['frac_z'] + n_z, n_z + 1)
                    X, Y, Z = np.meshgrid(x_range, y_range, z_range)
                    X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
                    new_pos = np.column_stack((X, Y, Z, np.ones_like(Z)))
                    new_frac_pos = np.dot(self.Q_mat, new_pos.T)[:3, :].T
                    new_frac_pos -= new_frac_pos // 1
                    _, un_ind = np.unique(np.round(new_frac_pos, 2), axis=0, return_index=True)
                    new_frac_pos = np.round(new_frac_pos[un_ind], 6)
                    mask = new_frac_pos < 1.0
                    mask = mask.all(axis=1)
                    new_frac_pos = new_frac_pos[mask]
                    new_sites_arr = np.array([(site['atom_type'], site['atomic_number'],
                                               frac_x, frac_y, frac_z, site['occ'], site['DW'])
                                              for (frac_x, frac_y, frac_z) in new_frac_pos], dtype=self.sites_dtype)
                    new_sites_l.append(new_sites_arr)

                new_sites = np.concatenate(new_sites_l)
                total_new_sites = new_sites.size
                self.print_debug('Iteration #%d' % num_iter)
                self.print_debug('Found %d atomic sites' % total_new_sites)
                self.print_debug('Previous %d atomic sites' % total_old_sites)
                if total_new_sites == total_old_sites or num_iter >= max_iter:
                    self.sites = new_sites
                    break
                else:
                    total_old_sites = total_new_sites
                    num_iter += 1
                    scaling += 1

            self.print_verbose('Total number of atomic sites: %d' % self.sites.size)

    def build_unit_cell(self, xyz=None):
        cell_positions = []
        xyz_cell_positions = []
        atom_types = []
        site_occps = []
        DWs = []
        atomic_numbers = []

        for site in self.sites:
            site_pos = np.array([site['frac_x'], site['frac_y'], site['frac_z'], 1])
            sym_pos = np.array([ np.dot(sym_op, site_pos)[:3] for sym_op in self.sym_ops ])
            sym_pos -= sym_pos // 1
            _, uniq_indx = np.unique(np.round(sym_pos, 2),return_index=True, axis=0)
            uc_positions = sym_pos[uniq_indx]
            xyz_positions = np.dot(self.basis, uc_positions.T).T
            cell_positions.append(uc_positions)
            xyz_cell_positions.append(xyz_positions)
            atom_types.append([site['atom_type'] for _ in range(len(xyz_positions))])
            atomic_numbers.append([site['atomic_number'] for _ in range(len(xyz_positions))])
            site_occps.append([site['occ'] for _ in range(len(xyz_positions))])
            DWs.append([site['DW'] for _ in range(len(xyz_positions))])

        cell_positions = list(chain.from_iterable(cell_positions))
        xyz_cell_positions = list(chain.from_iterable(xyz_cell_positions))
        atom_types = list(chain.from_iterable(atom_types))
        DWs = list(chain.from_iterable(DWs))
        atomic_numbers = list(chain.from_iterable(atomic_numbers))
        site_occps = list(chain.from_iterable(site_occps))

        self.unit_cell_positions = np.array([(atom_type, atomic_number, frac_x, frac_y, frac_z, occ, DW)
                                             for atom_type, atomic_number, (frac_x, frac_y, frac_z), occ, DW
                                             in zip(atom_types, atomic_numbers, cell_positions, site_occps, DWs)],
                                            dtype=self.sites_dtype)

        self.xyz_positions = np.array([(atom_type, frac_x, frac_y, frac_z) for atom_type, (frac_x, frac_y, frac_z)
                                       in zip(atom_types, xyz_cell_positions)], dtype=self.xyz_dtype)
        if xyz is not None:
            self.to_xyz(xyz, supercell=False)


    def to_xyz(self, filepath, supercell=False):
        if supercell:
            cell = self.supercell_xyz_positions
        else:
            cell = self.unit_cell_positions
        xyz_arr = np.array([(atom_type, x, y, z) for atom_type, x, y, z in zip(cell['atom_type'], cell['x'], cell['y'],
                                                                               cell['z'])], dtype=self.xyz_dtype)
        try:
            np.savetxt(filepath, xyz_arr, fmt='%s    %2.4f    %2.4f    %2.4f', header='%s\n' % str(xyz_arr.shape[0]),
                       comments='')
            return True
        except FileNotFoundError:
            warn('File error!')
            return False


    def to_XYZ(self, filepath, supercell=False):
        if supercell:
            cell = self.supercell_sites
        else:
            cell = self.unit_cell_positions

        XYZ_arr = np.array([(atomic_number, x, y, z, occ, dw) for atomic_number, x, y, z, occ, dw in
                            zip(cell['atomic_number'], cell['x'], cell['y'], cell['z'], cell['occ'], cell['DW'])],
                           dtype=self.XYZ_dtype)
        cell_dims = [XYZ_arr['x'].max(0), XYZ_arr['x'].max(0), XYZ_arr['z'].max(0)]
        try:
            np.savetxt(filepath, XYZ_arr, fmt='  %d  %2.4f  %2.4f  %2.4f  %2.4f  %2.4f',
                       header='#\n      %2.4f %2.4f %2.4f' % (cell_dims[0], cell_dims[1], cell_dims[2]), footer='-1',
                       comments='')
            return True
        except FileNotFoundError:
            warn('File error!')
            return False

    def CartesianBasis(self):
        # Construct Orthogonal Lattice basis
        a, b, c = self.lattice_const
        alpha, beta, gamma = self.lattice_angles * np.pi / 180
        a_x = a
        b_x = b * np.cos(gamma)
        b_y = b * np.sin(gamma)
        c_x = c * np.cos(beta)
        c_y = c * np.cos(alpha) * np.sin(gamma)  # c(np.dot(B.C)*by/b)
        c_z = np.sqrt(np.abs(c ** 2 - c_x ** 2 - c_y ** 2))
        cart_basis = np.array([[a_x, b_x, c_x],
                               [0, b_y, c_y],
                               [0, 0, c_z]])
        return cart_basis

    def OrientationMatrix(self, hkl=np.array([0, 0, 1]), uvw=np.array([0, 1, 0]), mode='uvw'):
        '''
        Orient Crystal
        hkl: hkl-plane.
        uvw: direct lattice vector.
        mode: projection mode, 'uvw', 'hkl': Projection is along G_hkl (taken as z-axis), 'none.
        '''

        # Setup Orientation Matrix
        a1, a2, a3 = self.basis.T
        UVW = uvw[0] * a1 + uvw[1] * a2 + uvw[2] * a3
        b1, b2, b3 = [np.cross(a2, a3), np.cross(a3, a1), np.cross(a1, a2)] / self.lattice_volume
        HKL = hkl[0] * b1 + hkl[1] * b2 + hkl[2] * b3
        if mode == 'hkl':
            b_vec = np.cross(UVW, HKL)
            b_vec_norm = b_vec / np.linalg.norm(b_vec)  # "x"
            n_vec_norm = HKL / np.linalg.norm(HKL)  # "z"
            t_vec_norm = np.cross(n_vec_norm, b_vec_norm) / np.linalg.norm(np.cross(n_vec_norm, b_vec_norm))  # "y"
            self.R = np.hstack((b_vec_norm, t_vec_norm, n_vec_norm))
            self.R = self.R.reshape(-1, 3)
        elif mode == 'uvw':
            b_vec = np.cross(HKL, UVW)
            b_vec_norm = b_vec / np.linalg.norm(b_vec)  # "x"
            n_vec_norm = UVW / np.linalg.norm(UVW)  # "z"
            t_vec_norm = np.cross(n_vec_norm, b_vec_norm) / np.linalg.norm(np.cross(n_vec_norm, b_vec_norm))  # "y"
            self.R = np.hstack((b_vec_norm, t_vec_norm, n_vec_norm))
            self.R = self.R.reshape(-1, 3)
        elif mode == 'none':
            self.R = np.identity(3)
        self.print_verbose('Orientation Matrix:')
        self.print_verbose(np.round(self.R, 4))

    def make_orthogonal_supercell(self, supercell_size=np.array([20., 20., 20.]), projec_1=np.array([1, 0, 0]),
                                  projec_2=np.array([0, 0, 1]), xyz=None, XYZ=None):

        # Construct orthonormal lattice basis
        basis_orthogonal = self.CartesianBasis()
        basis_orthonormal = basis_orthogonal / np.linalg.norm(basis_orthogonal, axis=0)
        self.print_debug('orthonormal basis:\n', np.round(basis_orthogonal.T, 4))

        # Project orthonormal lattice basis on the projection directions
        b_out_orthogonal = np.array([projec_1 * vec for vec in basis_orthogonal.T)]).sum(0)
        b_out_orthonormal = b_out_orthogonal / np.linalg.norm(b_out_orthogonal)
        c_out_orthogonal = np.array([projec_2 * vec for vec in basis_orthogonal.T)]).sum(0)
        c_out_orthonormal = c_out_orthogonal / np.linalg.norm(c_out_orthogonal)
        a_out_orthonormal = np.cross(b_out_orthonormal, c_out_orthonormal)
        self.print_debug('projected basis:\n%s' %format(np.round(np.array([a_out_orthonormal,b_out_orthonormal,
                                                                           c_out_orthonormal]),4)))
        # Orthogonality Check
        check = np.round(np.dot(b_out_orthonormal, c_out_orthonormal), 5)
        if check != 0.0:
            self.print_debug('New b and c vectors are not orthonormal, dot(b,c)= %2.2f' % check)
            self.print_debug('Replace b by cross(a,c):')
            b_out_orthonormal = np.cross(a_out_orthonormal, c_out_orthonormal)
            self.print_debug('new b:',b_out_orthonormal)
            check = np.round(np.dot(b_out_orthonormal, c_out_orthonormal), 5)
            if check != 0.0:
                self.print_debug('Can not find orthonormal basis')

        self.print_debug('new orthonormal basis (a,b,c):\n',a_out_orthonormal,b_out_orthonormal,c_out_orthonormal)
        new_basis_orthonormal = np.column_stack([a_out_orthonormal, b_out_orthonormal, c_out_orthonormal])

        # Transformation Matrix
        trans_mat = np.dot(basis_orthonormal.T, new_basis_orthonormal).T
        trans_inv_mat = np.linalg.inv(trans_mat)
        self.print_debug('Transformation Matrix:\n', np.round(trans_mat,5))
        self.print_debug('Inverse Transformation Matrix:\n', np.round(trans_inv_mat,5))

        # Find supercell dimensions in old orthogonal basis
        supercell_corners = []
        supercell_size = np.array(supercell_size).astype(np.float)
        sp_a, sp_b, sp_c = supercell_size
        boundaries = [[sp_a, 0, 0], [0, sp_b, 0], [0, 0, sp_c],
                      [sp_a, sp_b, 0], [sp_a, 0, sp_c], [0, sp_b, sp_c],
                      [sp_a, sp_b, sp_c], [0, 0, 0]]
        for bd in boundaries:
            supercell_corners.append(bd)
        supercell_corners = np.unique(supercell_corners, axis=0)

        supercell_corners_org = np.dot(trans_inv_mat, supercell_corners.T).T

        max_range = np.ceil(supercell_corners_org.max(0) / self.lattice_const).astype(np.int)
        min_range = np.floor(supercell_corners_org.min(0) / self.lattice_const).astype(np.int)
        repeats = max_range - min_range
        self.print_debug('Supercell %s contains unit cell repetition (a,b,c): %s' %(format(supercell_size),format(repeats)))

        # Find new translation vectors
        t_vectors_org = np.array([[self.lattice_const[0], 0, 0],
                                  [0, self.lattice_const[1], 0],
                                  [0, 0, self.lattice_const[2]]])
        t_a, t_b, t_c = np.dot(trans_mat, t_vectors_org).T
        self.print_debug('New Translation Vectors:\n%s' %format(np.round(np.array([t_a,t_b,t_c]),4)))

        # Build supercell
        t_vec = np.column_stack([t_a, t_b, t_c])
        bnorm = np.linalg.norm(basis_orthogonal, axis=0)
        supercell_sites = []
        for site in self.unit_cell_positions:
            old_pos = np.array([site['frac_x'], site['frac_y'], site['frac_z']])
            new_pos = np.dot(trans_mat, old_pos * bnorm)
            col_x = np.linspace(min_range[0], max_range[0], num=repeats[0] + 1)
            col_y = np.linspace(min_range[1], max_range[1], num=repeats[1] + 1)
            col_z = np.linspace(min_range[2], max_range[2], num=repeats[2] + 1)
            col_X, col_Y, col_Z = np.meshgrid(col_x, col_y, col_z)
            col_X, col_Y, col_Z = col_X.flatten(), col_Y.flatten(), col_Z.flatten()
            col_pos = np.column_stack([col_X, col_Y, col_Z])
            pos = np.dot(col_pos, t_vec.T)
            pos += new_pos
            mask = np.logical_and(pos >= 0, pos <= supercell_size).all(axis=1)
            xyz_pos_arr = pos[mask]
            supercell_site = np.array([(site['atom_type'], site['atomic_number'], x, y, z, site['occ'], site['DW'])
                                       for (x, y, z) in xyz_pos_arr], dtype=self.supercell_sites_dtype)

            supercell_sites.append(supercell_site)
        self.supercell_sites = np.sort(np.concatenate(supercell_sites), axis=0)
        self.supercell_xyz_positions = np.array([(atom_type, x, y, z)
                                                 for atom_type, x, y, z in zip(self.supercell_sites['atom_type'],
                                                                                 self.supercell_sites['x'],
                                                                                 self.supercell_sites['y'],
                                                                                 self.supercell_sites['z'])],
                                                dtype=self.xyz_dtype)


        if xyz is not None:
            self.to_xyz(xyz, supercell=True)
            self.print_verbose('saving xyz file %s' % xyz)
        if XYZ is not None:
            self.to_XYZ(XYZ, supercell=True)
            self.print_verbose('saving XYZ file %s' % XYZ)
