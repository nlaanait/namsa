from namsa import SupercellBuilder, MSAGPU
from utils import *
import numpy as np
from time import time
import sys, os, re
import h5py
from mpi4py import MPI
from itertools import chain, product
import tensorflow as tf
import lmdb

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()



def simulate(filehandle, cif_path, idx= None, gpu_id=0, clean_up=False):
    # load cif and get sim params
    spgroup_num, matname = parse_cif_path(cif_path)
    sp = SupercellBuilder(cif_path, verbose=False, debug=False)
    sim_params = get_sim_params(sp, grid_steps=np.array([2,2]), orientation_num=3)
    z_dirs = sim_params['z_dirs']
    y_dirs = sim_params['y_dirs']
    cell_dim = sim_params['cell_dim']
    slab_t = sim_params['slab_t']
    sim_params['space_group']= spgroup_num
    sim_params['material'] = matname
    energies = [100, 125, 150, 175, 200]
    for (sample_idx, (y_dir, (z_idx, z_dir), energy)) in enumerate(product(y_dirs, enumerate(z_dirs), energies)):
        try:
            t = time()
            # build supercell
            sp.build_unit_cell()
            sp.make_orthogonal_supercell(supercell_size=np.array([cell_dim,cell_dim,slab_t]),
                             projec_1=y_dir, projec_2=z_dir)
    
            # set simulation params
            slice_thickness = sim_params['d_hkl'][z_idx]
            #energy = sim_params['energy']
            semi_angle= sim_params['semi_angles'][z_idx]
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
            proj_potential = process_potential(msa.potential_slices, mask=mask, normalize=True, fp16=True)
            cbed = process_cbed(msa.probes, normalize=True, fp16=True)
            
            # update sim_params dict
            sim_params = update_sim_params(sim_params, msa_cls=msa, sp_cls=sp)
   
            #
            #del sp

            # 
            # clean-up context and/or allocated memory
            if clean_up and ctx is not None:
                msa.clean_up(ctx=ctx, vars=msa.vars)
                del msa
            else:
                msa.clean_up(ctx=None, vars=msa.vars)
                del msa
            
            # check data integrity
            has_nan = np.all(np.isnan(cbed)) or np.all(np.isnan(proj_potential))
            wrong_shape = cbed.shape != (4, 512, 512) or proj_potential.shape != (1, 512, 512)
            if has_nan or wrong_shape:
                print('rank=%d, skipped simulation=%s, index=%d, error=NaN' % (comm_rank, cif_path, sample_idx))
            else:
                # write to h5 / tfrecords / lmdb
                g = filehandle.create_group('sample_%d_%d' % (idx, sample_idx))
                g.attrs['space_group'] = np.string_(sim_params['space_group'])
                g.attrs['material'] = np.string_(sim_params['material'])
                g.attrs['cif_path'] = np.string_(cif_path)
                g.attrs['energy'] = energy
                dset = g.create_dataset('z_dir',  data=z_dir)
                dset = g.create_dataset('y_dir',  data=y_dir)
                g.attrs['time'] = time() - t
                g.attrs['semi_angle'] = sim_params['semi_angles'][z_idx]
                g.attrs['d_hkl'] = sim_params['d_hkl'][z_idx]
                filehandle.flush() 
                print('rank=%d, finished simulation=%s, index=%d' % (comm_rank, cif_path, sample_idx))
        except Exception as e:
            print("rank=%d, skipped simulation=%s, error=%s" % (comm_rank, cif_path, format(e)))
        finally:
            try:
                if clean_up and ctx is not None:
                    msa.clean_up(ctx=ctx, vars=msa.vars)
                    del msa
                else:
                    msa.clean_up(ctx=None, vars=msa.vars)
                    del msa
            except:
                pass
        

def main(cifdir_path, outdir_path):
    t = time()
    cifpaths = get_cif_paths(cifdir_path)
    num_sims = cifpaths.size
    #num_sims = comm_size * 2 
    # HDF5
    h5path = os.path.join(outdir_path, 'params_batch_%d.h5'% comm_rank)
    mode = 'w'
    with h5py.File(h5path, mode=mode) as f:
        for (idx, cif_path) in enumerate(cifpaths[comm_rank:num_sims:comm_size]):
            manual = idx < ( num_sims - comm_size) 
            spgroup_num, matname = parse_cif_path(cif_path)
            if comm_rank == 0 and bool(idx % 100):
                print('time=%3.2f, num_sims= %d' %(time() - t, idx * comm_size))
            simulate(f, cif_path, idx=idx, gpu_id=int(np.mod(comm_rank, 6)), clean_up=manual)

def main_test(cifdir_path):
    cifpaths_train, cifpaths_test= get_cif_paths(cifdir_path, ratio=0.2)
    print("train", cifpaths_train[:10])
    print("test", cifpaths_test[:10])
                
if __name__ == "__main__":
    if len(sys.argv) > 2:
        cifdir_path, outdir_path = sys.argv[-2:]
        main(cifdir_path, outdir_path)
    else:
        print("Pass directory paths for sim input files and h5 output files")
