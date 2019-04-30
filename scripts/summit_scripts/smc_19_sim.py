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
import pycuda.driver as cuda
import pycuda 

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()

def setup_device(gpu_rank=0):
    global ctx
    cuda.init()
    dev = cuda.Device(gpu_rank)
    ctx = dev.make_context()
    import atexit
    def _clean_up():
        global ctx
        if ctx is not None:
            try:#global ctx
                #ctx.push()
                ctx.pop()
                ctx.detach()
                #ctx = None
            except:
                pass
        from pycuda.tools import clear_context_caches
        clear_context_caches()
    atexit.register(_clean_up)
    return ctx 

def simulate(filehandle, cif_path, idx= None, gpu_ctx=None, clean_up=False):
    
    # load cif and get sim params
    spgroup_num, matname = parse_cif_path(cif_path)
    sp = SupercellBuilder(cif_path, verbose=False, debug=False)
    latts = np.array(sp.structure.lattice.abc)
    if np.any(latts >= 10.):
        print('rank=%d, skipped simulation=%s, latt. const. too large=%s' % (comm_rank, cif_path, format(latts)))
        return    
         
    sim_params = get_sim_params(sp, grid_steps=np.array([8,8]), orientation_num=3)
    z_dirs = sim_params['z_dirs']
    y_dirs = sim_params['y_dirs']
    cell_dim = sim_params['cell_dim']
    slab_t = sim_params['slab_t']
    sim_params['space_group']= spgroup_num
    sim_params['material'] = matname
    energies = np.arange(100,200,10)
    
#     ctx = msa.setup_device(gpu_rank=gpu_id)
    for (sample_idx, energy) in enumerate(energies):
        try:
            cbed_stack = []
            for y_dir, (z_idx, z_dir) in zip(y_dirs, enumerate(z_dirs)):
                t = time()
                # build supercell
                sp.build_unit_cell()
                sp.make_orthogonal_supercell(supercell_size=np.array([cell_dim,cell_dim,slab_t]),
                                 projec_1=y_dir, projec_2=z_dir)

                # set simulation params
                slice_thickness = sim_params['d_hkl'][z_idx]
                semi_angle= sim_params['semi_angles'][z_idx]
                probe_params = sim_params['probe_params']
                sampling = sim_params['sampling']
                grid_steps = sim_params['grid_steps']

                # simulate
                msa = MSAGPU(energy, semi_angle, sp.supercell_sites, sampling=sampling,
                     verbose=False, debug=False)
#                 ctx = msa.setup_device(gpu_rank=gpu_id)
                msa.calc_atomic_potentials()
                msa.build_potential_slices(gpu_ctx, slice_thickness)
                msa.build_probe(probe_dict=probe_params)
                msa.generate_probe_positions(grid_steps=grid_steps) 
                msa.plan_simulation()
                msa.multislice(bandwidth=1./2)

                # process cbed 
                cbed = msa.probes.mean(0) 
                pot_isnan = np.isnan(msa.potential_slices)
                # update sim_params dict
                sim_params = update_sim_params(sim_params, msa_cls=msa, sp_cls=sp)

                # 
                # clean-up context and/or allocated memory
                msa.clean_up(ctx=None, vars=msa.vars)
#                 if clean_up and ctx is not None:
#                     msa.clean_up(ctx=ctx, vars=msa.vars)
# #                     del msa
#                 else:
#                     msa.clean_up(ctx=None, vars=msa.vars)
# #                     del msa

                # check data integrity
#                 isnan = np.isnan(cbed)
#                 print('cbed elements with nan', np.where(isnan == True)[0].size)
                has_nan = np.all(np.isnan(cbed)) 
#                 pot_isnan = np.isnan(msa.potential_slices)
#                 print('pot elements with nan', np.where(pot_isnan == True)[0].size)
                
#                 if has_nan:
# #                     print(z_dirs,y_dirs,sim_params['semi_angles'],sim_params['abc'], energy)
#                     print('')
                wrong_shape = cbed.shape != (512, 512) 
                if has_nan: 
                    print('rank=%d, skipped simulation=%s, index=%d, error=NaN, abc=%s' % (comm_rank, cif_path, sample_idx, format(sim_params['abc'])))
                    pass
                elif wrong_shape:
                    print('rank=%d, skipped simulation=%s, index=%d, error=wrong cbed shape' % (comm_rank, cif_path, sample_idx))
                    pass
                else:
                    cbed_stack.append(cbed)


            # write to h5 / tfrecords / lmdb
            if len(cbed_stack) != 3:
                pass
            else:
                cbed_stack = np.stack(cbed_stack)
                g = filehandle.create_group('sample_%d_%d' % (idx, sample_idx))
                g.attrs['space_group'] = np.string_(sim_params['space_group'])
                g.attrs['material'] = np.string_(sim_params['material'])
                g.attrs['energy_keV'] = energy
                dset = g.create_dataset('cbed_stack', data=cbed_stack)
                g.attrs['z_dirs'] = np.dstack(z_dirs)
                g.attrs['y_dirs'] = np.dstack(y_dirs)
                g.attrs['semi_angles_rad'] = sim_params['semi_angles']
                g.attrs['d_hkls_angstrom'] = sim_params['d_hkl']
                g.attrs['abc_angstrom'] = sim_params['abc']
                g.attrs['angles_degree'] = sim_params['angles']
                g.attrs['formula'] = sim_params['formula'] 
                filehandle.flush() 
                print('rank=%d, finished simulation=%s, index=%d' % (comm_rank, cif_path, sample_idx))
        except Exception as e:
            print("rank=%d, skipped simulation=%s, error=%s" % (comm_rank, cif_path, format(e)))
        finally:
            try:
                if clean_up and gpu_ctx is not None:
                    msa.clean_up(ctx=gpu_ctx, vars=msa.vars)
                    del msa
                else:
                    msa.clean_up(ctx=None, vars=msa.vars)
                    del msa
            except:
                pass
        

def generate_data(samples, outdir_path, mode='train', runtime=2000, gpu_ctx=None):
    t = time()
    num_sims = samples.size
    h5path = os.path.join(outdir_path, 'batch_%s_%d.h5'% (mode, comm_rank))
    mode = 'w'
    f = h5py.File(h5path, mode=mode)
    for (idx, cif_path) in enumerate(samples[comm_rank:num_sims:comm_size]):
        manual = idx < ( num_sims - comm_size) 
        spgroup_num, matname = parse_cif_path(cif_path)
        if comm_rank == 0 and bool(idx % 100):
            print('time=%3.2f, num_sims= %d' %(time() - t, idx * comm_size))
        if (time() - t_elaps) < runtime:
            simulate(f, cif_path, idx=idx, gpu_ctx=gpu_ctx, clean_up=False)
        else:
            f.flush()
            f.close()
            return
    f.flush()
    f.close()
    return
            
def get_samples(cif_paths, ratio=0.9):
    samples = cif_paths
    if comm_rank == 0:
        indices = np.arange(0,samples.size, dtype=np.int64) 
        np.random.shuffle(indices)
    else:
        indices = np.empty(samples.size, dtype=np.int64)
    comm.Bcast(indices, root=0)
    samples = samples[indices]
    if ratio is not None:
        train_size = int(samples.size * ratio)
        remain = (1 - ratio)/2
        samples_train = samples[:train_size]
        samples_dev = samples[train_size:train_size + int(remain * train_size)]
        samples_test = samples[train_size + int(remain * train_size):]
        if comm_rank == 0:
            print('samples sizes (train, dev, test): %d, %d, %d' %(samples_train.size, samples_dev.size, samples_test.size))
        return samples_train, samples_dev, samples_test
    return samples
         
def main(cifdir_path, outdir_path, runtime=1800):
    global t_elaps
    t_elaps = time()
    cif_paths = get_cif_paths(cifdir_path)
    ctx = setup_device(gpu_rank=int(np.mod(comm_rank, 6)))
    samples_train, samples_dev, samples_test = get_samples(cif_paths, ratio=0.9)
    generate_data(samples_train, outdir_path, mode='train', runtime=runtime*0.8, gpu_ctx=ctx)
    print('rank=%d, finished simulating training data' % comm_rank)
    generate_data(samples_dev, outdir_path, mode='dev', runtime=runtime*0.9, gpu_ctx=ctx)
    print('rank=%d, finished simulating dev data' % comm_rank)
    generate_data(samples_test, outdir_path, mode='test', runtime=runtime, gpu_ctx=ctx)
    print('rank=%d, finished simulating test data' % comm_rank)
    return
            
def main_test(cifdir_path):
    cifpaths_train, cifpaths_test= get_cif_paths(cifdir_path, ratio=0.2)
    print("train", cifpaths_train[:10])
    print("test", cifpaths_test[:10])
                
if __name__ == "__main__":
    if len(sys.argv) > 3:
        cifdir_path, outdir_path, runtime = sys.argv[-3:]
        main(cifdir_path, outdir_path, runtime=int(runtime))
    else:
        print("Pass directory paths for sim input files and h5 output files")
