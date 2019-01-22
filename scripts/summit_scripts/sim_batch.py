from namsa import SupercellBuilder, MSAGPU
from utils import *
import numpy as np
from time import time
import sys, os, re
import h5py
from mpi4py import MPI
from itertools import chain
import tensorflow as tf

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()



def simulate(filehandle, cif_path, gpu_id=0, clean_up=False):
    # load cif and get sim params
    spgroup_num, matname = parse_cif_path(cif_path)
    index = 0 
    sp = SupercellBuilder(cif_path, verbose=False, debug=False)
    sim_params = get_sim_params(sp)
    z_dir = sim_params['z_dirs'][index]
    y_dir = sim_params['y_dirs'][index]
    cell_dim = sim_params['cell_dim']
    slab_t = sim_params['slab_t']
    sim_params['space_group']= spgroup_num
    sim_params['material'] = matname
    
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
    
    # write to h5 / tfrecords
    if isinstance(filehandle, h5py.Group):
         write_h5(filehandle, msa.probes, proj_potential, sim_params)
    else:
         write_tfrecord(filehandle, msa.probes, proj_potential, sim_params)

    print('rank=%d, simulation=%s' % (comm_rank, cif_path))
    
    # clean-up context and/or allocated memory
    if clean_up and ctx is not None:
        msa.clean_up(ctx=ctx, vars=msa.vars)
    else:
        msa.clean_up(ctx=None, vars=msa.vars)


def main(cifdir_path, outdir_path, save_mode="h5"):
    t = time()
    cifpath_list = get_cif_paths(cifdir_path)
    batch_num, _ = np.divmod(comm_rank, 6)
    if save_mode == "h5": 
        h5path = os.path.join(outdir_path, 'batch_%d.h5'% comm_rank)
        if os.path.exists(h5path):
            mode ='r+'
        else:
            mode ='w'
        with h5py.File(h5path, mode=mode) as f:
            for idx in range(comm_rank, len(cifpath_list), comm_size):
                cif_path = cifpath_list[idx]
                manual = idx < ( len(cifpath_list) - comm_size) 
                spgroup_num, matname = parse_cif_path(cif_path)
                try:
                    h5g = f.create_group(matname)
                except Exception as e:
                    print("rank=%d" % comm_rank, e, "group=%s exists" % matname)
                    h5g = f[matname]
                if comm_rank == 0 and bool(idx % 500):
                    print('time=%3.2f, idx= %d' %(time() - t, idx))
                simulate(h5g, cif_path, gpu_id=int(np.mod(comm_rank, 6)), clean_up=manual)
    else:
        tfrecpath = os.path.join(outdir_path, 'batch_%d.tfrecords'% comm_rank)   
        with tf.python_io.TFRecordWriter(tfrecpath) as tfrec:
            for idx in range(comm_rank, len(cifpath_list), comm_size):
                cif_path = cifpath_list[idx]
                manual = idx < ( len(cifpath_list) - comm_size) 
                spgroup_num, matname = parse_cif_path(cif_path)
                if comm_rank == 0 and bool(idx % 500):
                    print('time=%3.2f, idx= %d' %(time() - t, idx))
                simulate(tfrec, cif_path, gpu_id=int(np.mod(comm_rank, 6)), clean_up=manual)
    
    sim_t = time() - t
    if comm_rank == 0:
        print("took %3.3f seconds" % sim_t)    

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
        cifdir_path, outdir_path, save_mode = sys.argv[-3:]
        if save_mode not in ["h5","tfrecord"]:
            print("specify saving format")
            sys.exit()
        main(cifdir_path, outdir_path, save_mode)
    elif len(sys.argv) == 2:
        cifdir_path = sys.argv[-1]
        main_test(cifdir_path)
    else:
        print("Pass directory paths for sim input files and h5 output files")
