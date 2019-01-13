from namsa import SupercellBuilder, MSAGPU
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
    # for key in json_labels
#    for key, itm in json_labels['sim'].items():
#        dset_cbed.attrs[key] = itm
#    for key, itm in json_labels['label'].items():
#        dset_pot.attrs[key] = itm
#    return potential_data.min(), potential_data.max(), potential_data.mean()
    return

def set_sim_params(unit_cell):
    """
    return a dict object to set params of simulation and write to h5.
    """
    pass
    
def simulate(h5g, cif_path, gpu_rank=0, clean_up=False):
    # build supercell
    sp = SupercellBuilder(cif_path, verbose=False, debug=False)
    sim_params = set_sim_params(sp)
    z_dir = np.array([0,0,1])
    y_dir = np.array([1,0,0])
    sp.build_unit_cell()
    sp.make_orthogonal_supercell(supercell_size=np.array([2*34.6,2*34.6,198.0]),
                             projec_1=y_dir, projec_2=z_dir)
    # set simulation params
    slice_thickness = 0.5 # Angstroms
    en = 100 # keV
    semi_angle= 10e-3 # radians
    max_ang = 150e-3 # radians
    step = 2.1 # Angstroms
    aberration_params = {'C1':500., 'C3': 3.3e7, 'C5':44e7}
    probe_params = {'smooth_apert': True, 'scherzer': False, 'apert_smooth': 60, 
                'aberration_dict':aberration_params, 'spherical_phase': True}
    
    # simulate
    msa = MSAGPU(en, semi_angle, sp.supercell_sites, sampling=np.array([256,256]),
                 verbose=False, debug=False)
    ctx = msa.setup_device(gpu_rank=gpu_rank)
    msa.calc_atomic_potentials()
    msa.build_potential_slices(slice_thickness)
    msa.build_probe(probe_dict=probe_params)
    msa.generate_probe_positions(probe_step=np.array([step,step]), 
                             probe_range=np.array([[0.25,0.75],[0.25,0.75]]))
    msa.plan_simulation()
    msa.multislice()
    
    # write to h5
    write_h5(h5g, msa.probes, msa.potential_slices.sum(0), None)
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
        for idx in range(comm_rank, len(cifpath_list), comm_size):
            cif_path = cifpath_list[idx]
            manual = idx < (len(cifpath_list) - comm_size) 
            spgroup_num, matname = parse_cif_path(cif_path)
            try:
                h5g = f.create_group(matname)
            except Exception as e:
                print("rank=%d" % comm_rank, e, "group=%s exists" % matname)
                h5g = f[matname]
            if comm_rank == 0:
                print('current idx: %d' %idx)
            simulate(h5g, cif_path, gpu_rank=int(np.mod(comm_rank, 6)), clean_up=manual)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        cifdir_path, h5dir_path = sys.argv[-2:]
        main(cifdir_path, h5dir_path)
    else:
        print("Pass directory paths for sim input files and h5 output files")
