import json
import os
from .optics import *


def make_label_sim_json(structure, space_group, unit_cell, supercell_size, output_name, direc, num_uc=2,
                        probe_sampling=8):
    """
    Make label with space and (fractional) chemical composition.
    """

    a, b, c = np.round(np.array([structure.lattice.a, structure.lattice.b, structure.lattice.c]), 4)
    alpha, beta, gamma = np.round(np.array([structure.lattice.alpha, structure.lattice.beta, structure.lattice.gamma]),
                                  4)
    label_dic = dict()

    # Get fractional chemical composition
    unique_elem = np.unique(unit_cell['atomic_number'])
    num_atoms = unit_cell['occ'].sum(0)
    for elem in unique_elem:
        ind = np.where(unit_cell['atomic_number'] == elem)[0]
        total_num_elem = unit_cell['occ'][ind].sum()
        label_dic[str(elem)] = np.round(np.round(total_num_elem / num_atoms, 4).astype(np.float), 4)

    # Add space group
    label_dic['space_group'] = int(space_group.split('_')[0])

    # Add lattice constants
    label_dic['a'] = a
    label_dic['b'] = b
    label_dic['c'] = c

    # Add lattice angles
    label_dic['alpha'] = alpha
    label_dic['beta'] = beta
    label_dic['gamma'] = gamma

    # Set simulation params
    sim_dic = dict()
    sim_dic['c_x'] = supercell_size[0]
    sim_dic['c_y'] = supercell_size[1]
    sim_dic['c_z'] = supercell_size[2]
    #     tile_x, tile_y, pix_x, pix_y, x_dim, y_dim = match_tile_pix(a,b, verbose=True)
    #     assert x == y and x == 1024, 'x-dim and y-dim do not match 1024'
    #     tile_x, tile_y, pix_x, pix_y, x_dim, y_dim = find_tile_pix(a,b,verbose=False)
    tile_x, tile_y, pix_x, pix_y = [1, 1, 0.1, 0.1]
    sim_dic['tile.uc.x'] = int(tile_x)
    sim_dic['tile.uc.y'] = int(tile_y)
    sim_dic['px'] = pix_x
    sim_dic['py'] = pix_y
    #     sim_dic['wx_start'], sim_dic['wy_start'], sim_dic['wx_end'], sim_dic['wy_end'] = get_scan_range(num_uc,tile_x, tile_y)
    sim_dic['wx_start'], sim_dic['wy_start'], sim_dic['wx_end'], sim_dic['wy_end'] = [0.5, 0.5, 0.6, 0.6]
    #     sim_dic['r'] = np.round(min(a,b)/8.,4)
    #     sim_dic['slice.thickness'] = min(np.round(c/4,4), 1.25)
    sim_dic['r'] = 0.25
    sim_dic['slice.thickness'] = 1.0
    semi_angle, _, _ = overlap_params(1, structure.lattice.d_hkl([2, 2, 0]), voltage2Lambda(100e3))
    sim_dic['semi_angle'] = np.round(semi_angle, 2)

    # Combine
    out_dic = {'sim': sim_dic, 'label': label_dic}

    # Write label
    with open(os.path.join(direc, output_name + '.json'), mode='w+') as f:
        json.dump(out_dic, f, indent=4)


def pop_DS(lst):
    for (i, itm) in enumerate(lst):
        if '.DS_Store' in itm:
            lst.pop(i)


def get_random_cif(root_path):
    # get space group folders
    space_group_dirs = os.listdir(root_path)
    pop_DS(space_group_dirs)
    idx = np.random.randint(0, len(space_group_dirs))
    # get crystals folder
    space_group_dir = space_group_dirs[idx]
    crystal_files = os.listdir(os.path.join(root_path, space_group_dir))
    pop_DS(crystal_files)
    # get crysal cif file
    idx = np.random.randint(0, len(crystal_files))
    cif_name = crystal_files[idx]
    cif_path = os.path.join(os.path.join(root_path, space_group_dir), cif_name)
    return cif_name.split('.')[0], cif_path, space_group_dir


def dhkl_spacing(uc_volume, basis, hkl=[0, 0, 1]):
    a1, a2, a3 = basis
    b1, b2, b3 = np.array([np.cross(a2, a3), np.cross(a3, a1), np.cross(a1, a2)]) / uc_volume
    h, k, l = hkl
    d_hkl_rep_vec = h * b1 + k * b2 + l * b3
    d_hkl_rep = np.sqrt(np.dot(d_hkl_rep_vec, d_hkl_rep_vec))
    d_hkl = 1. / d_hkl_rep
    return d_hkl
