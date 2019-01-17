from namsa import SupercellBuilder, MSAGPU
from namsa.scattering import get_kinematic_reflection, get_cell_orientation, overlap_params 
from namsa.optics import voltage2Lambda
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import numpy as np
from time import time
import sys, os, re
import h5py
from mpi4py import MPI
from itertools import chain

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()

def pop_DS(lst):
    for (i,itm) in enumerate(lst):
        if '.DS_Store' in itm:
            lst.pop(i)

def get_cif_paths(root_path):
    space_group_dirs = os.listdir(root_path)
    pop_DS(space_group_dirs)
    cifpath_list = []
    for spg_dir in space_group_dirs:
        cif_list = os.listdir(os.path.join(root_path,spg_dir))
        pop_DS(cif_list)
        cif_paths = [os.path.join(os.path.join(root_path,spg_dir),cif_name) for cif_name in cif_list]
        cifpath_list.append(cif_paths)
    cifpath_list = list(chain.from_iterable(cifpath_list))
    return cifpath_list 

def parse_cif_path(cif_path):
    spgroup, matname = cif_path.split(os.sep)[-2:]
    matname = matname.split('.')[0]
    spgroup_num = re.findall('\d+',spgroup)[0]
    return spgroup_num, matname

def write_h5(h5group, cbed, potential, params):
    # try:
    num_itms = len(h5group.items())
    g = h5group.create_group('sample_%d' % num_itms)
    dset_cbed = g.create_dataset('CBED',  data=cbed)
    dset_pot = g.create_dataset('potential', data=potential)
    # need to figure out how to assign attributes to each dset and the parent group.
    # now dumping everything into group attribute
    for key in params.keys():
        if key == 'probe_params':
            for probe_key in params['probe_params']:
                if probe_key is 'aberration_dict':
                    aberr = params['probe_params']['aberration_dict']
                    g.attrs['aberrations']= np.array([aberr['C1'], aberr['C3'], aberr['C5']])
                else:
                    g.attrs[probe_key] = params['probe_params'][probe_key]
        else:
            g.attrs[key] = params[key]
    return

def get_sim_params(sp_cell, slab_t= 200, sampling=np.array([512,512]), d_cutoff=4, grid_steps=np.array([32, 32]),
                   cell_dim = 100, energy=100e3, orientation_num=3, beam_overlap=1):
    """
    return a dict object to set params of simulation and write to h5.
    """
    sim_params= dict()
    
    # scattering params
    hkls, dhkls = get_kinematic_reflection(sp_cell.structure, top=orientation_num)
    if hkls[0].size > 3: # hexagonal systems    
        hkls = np.array([[itm[0], itm[1], itm[-1]] for itm in hkls])
    cutoff = dhkls < 5 # not considering less than 5 ang. d-spacing
    if dhkls[cutoff].size > 1:
        hkls, dhkls = hkls[cutoff], dhkls[cutoff]
    y_dirs = np.array([get_cell_orientation(z_dir) for z_dir in hkls])
    semi_angles, _, _ = overlap_params(1, dhkls, voltage2Lambda(energy))
    sim_params['y_dirs'] = y_dirs
    sim_params['z_dirs'] = hkls
    sim_params['semi_angles'] = semi_angles * 1e-3 # mrad
    sim_params['d_hkl'] = dhkls
    sim_params['cell_dim'] = cell_dim # ang.
    sim_params['energy'] = energy * 1e-3 # keV
    sim_params['sampling'] = sampling
    sim_params['slab_t'] = slab_t # ang.
    sim_params['grid_steps'] = grid_steps
    
    # optics params
    sim_params['probe_params'] = {'smooth_apert': True, 'scherzer': False, 'apert_smooth': 30, 
                'aberration_dict':{'C1':0., 'C3':0 , 'C5':0.}, 'spherical_phase': True}
    return sim_params

def update_sim_params(sim_params, msa_cls=None, sp_cls=None):
    # msa params
    if msa_cls is not None:
        msa_params = ['max_ang', 'kmax', 'debye_waller', 'dims', 'kpix_size', 'pix_size', 'sigma']
        for key in msa_params:
            try:
                val = msa_cls.__getattribute__(key)
                sim_params[key] = val
            except Exception as e:
                sim_params[key] = str(e)
    if sp_cls is not None:
        try:
            sim_params['formula'] = sp_cls.structure.formula
            sim_params['abc'] = sp_cls.structure.lattice.abc
            sim_params['angles'] = sp_cls.structure.lattice.angles
        except Exception as e:
            sim_params['formula'] = str(e)
            sim_params['abc'] = str(e)
            sim_params['angles'] = str(e)
    return sim_params

def process_potential(pot_slices, mask=None, sampling=None):
    proj_potential = np.imag(pot_slices).sum(0)
    if mask is None:
        mask = np.ones((sampling, sampling), dtype=np.bool)
        snapshot = slice(int(proj_potential.shape[0]// 4), int(3 * proj_potential.shape[1]//4))
        mask[snapshot, snapshot] = False
    else:
        mask = np.logical_not(mask)
    proj_potential[mask] = 0
    return proj_potential
    
def simulate(h5g, cif_path, gpu_id=0, clean_up=False):
    # load cif and get sim params
    index = 0 
    sp = SupercellBuilder(cif_path, verbose=False, debug=False)
    sim_params = get_sim_params(sp)
    z_dir = sim_params['z_dirs'][index]
    y_dir = sim_params['y_dirs'][index]
    cell_dim = sim_params['cell_dim']
    slab_t = sim_params['slab_t']
    
    # build supercell
    sp.build_unit_cell()
    sp.make_orthogonal_supercell(supercell_size=np.array([cell_dim,cell_dim,slab_t]),
                             projec_1=y_dir, projec_2=z_dir)
    
    # set simulation params
    slice_thickness = sim_params['d_hkl'][index]
    energy = sim_params['energy']
    semi_angle= sim_params['semi_angles'][index]
    probe_params = sim_params['probe_params']
    sampling = sim_params['sampling']
    grid_steps = sim_params['grid_steps']
    
    # simulate
    msa = MSAGPU(energy, semi_angle, sp.supercell_sites, sampling=sampling,
                 verbose=False, debug=False)
    ctx = msa.setup_device(gpu_rank=gpu_id)
    msa.calc_atomic_potentials()
    msa.build_potential_slices(slice_thickness)
    msa.build_probe(probe_dict=probe_params)
    msa.generate_probe_positions(grid_steps=grid_steps) 
    msa.plan_simulation()
    msa.multislice()
    
    # process cbed and potential
    mask = msa.bandwidth_limit_mask(sampling, radius=1./3).astype(np.bool)
    proj_potential = process_potential(msa.potential_slices, mask=mask)
    
    # update sim_params dict
    sim_params = update_sim_params(sim_params, msa_cls=msa, sp_cls=sp)
    
    # write to h5
    write_h5(h5g, msa.probes, proj_potential, sim_params)
    print('rank=%d, simulation=%s' % (comm_rank, cif_path))
    
    # clean-up context and/or allocated memory
    if clean_up and ctx is not None:
        msa.clean_up(ctx=ctx, vars=msa.vars)
    else:
        msa.clean_up(ctx=None, vars=msa.vars)

def main(cifdir_path, h5dir_path):
    cifpath_list = get_cif_paths(cifdir_path)
    h5path = os.path.join(h5dir_path, 'batch_%d.h5'% comm_rank)
    if os.path.exists(h5path):
        mode ='r+'
    else:
        mode ='w'
    with h5py.File(h5path, mode=mode) as f:
        for idx in range(comm_rank, 10, comm_size):
            cif_path = cifpath_list[idx]
            manual = idx < (10 - comm_size) 
            spgroup_num, matname = parse_cif_path(cif_path)
            try:
                h5g = f.create_group(matname)
            except Exception as e:
                print("rank=%d" % comm_rank, e, "group=%s exists" % matname)
                h5g = f[matname]
            if comm_rank == 0:
                print('current idx: %d' %idx)
            simulate(h5g, cif_path, gpu_id=int(np.mod(comm_rank, 6)), clean_up=manual)

def main_test(cifdir_path):
    cifpath_list = get_cif_paths(cifdir_path)
    idx = np.random.randint(0, len(cifpath_list))
    for _ in range(1000):
        cif_path = cifpath_list[idx]
        spgroup_num, matname = parse_cif_path(cif_path)
        sp = SupercellBuilder(cif_path, verbose=False, debug=False)
        sim_params = set_sim_params(sp, energy=100e3, orientation_num=3, beam_overlap=2)
        y_dir, z_dir = sim_params['y_dirs'][0], sim_params['z_dirs'][0]
        sp.build_unit_cell()
        sp.make_orthogonal_supercell(supercell_size=np.array([2*34.6,2*34.6,198.0]),
                             projec_1=y_dir, projec_2=z_dir)
        
        print("rank=%d, spgroup= %s, material=%s, z_dir=%s, y_dir=%s, d_hkl=%2.2f, semi_angle=%2.2f" 
              % (comm_rank, spgroup_num, matname, z_dir, y_dir, sim_params['d_hkl'][0], sim_params['semi_angles'][0]))
        if comm_rank == 0:
            print('current idx: %d' %idx)
                
if __name__ == "__main__":
    if len(sys.argv) > 2:
        cifdir_path, h5dir_path = sys.argv[-2:]
        main(cifdir_path, h5dir_path)
    elif len(sys.argv) == 2:
        cifdir_path = sys.argv[-1]
        main_test(cifdir_path)
    else:
        print("Pass directory paths for sim input files and h5 output files")
