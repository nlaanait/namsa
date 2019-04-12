from namsa import SupercellBuilder, MSAGPU
from utils import *
import numpy as np
from time import time
import sys, os, re, subprocess, shlex
import h5py
from mpi4py import MPI
from itertools import chain
import tensorflow as tf
import lmdb
import numpy as np

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()

def swap_out(lmdb_path):
    # delete current lmdb dir
    rm_args = "rm -r %s" % lmdb_path
    rm_args = shlex.split(rm_args)
    try:
        subprocess.run(rm_args, check=True)
    except subprocess.SubprocessError as e:
        print("rank %d: %s" % (comm_rank, format(e)))

    # replace with lmdb from repo
    user = os.environ.get('USER')
    lmdb_repo = "/gpfs/alpine/lrn001/proj-shared/nl/sims/data/lmdb_fix"
    lmdb_repo_list = os.listdir(lmdb_repo)
    index = np.random.randint(0, len(lmdb_repo_list))
    lmdb_path_src = os.path.join(lmdb_repo, lmdb_repo_list[index])
    if not os.path.exists(lmdb_path_src):
        print('replacement file %s not found' % lmdb_path_src)
        return
    src = lmdb_path_src 
    trg = lmdb_path 
    cp_args = "cp -r %s %s" %(src, trg)
    cp_args = shlex.split(cp_args)
    if not os.path.exists(trg):
        try:
            subprocess.run(cp_args, check=True)
        except subprocess.SubprocessError as e:
            print("rank %d: %s" % (comm_rank, format(e)))


def simulate(filehandle, cif_path, idx= None, gpu_id=0, clean_up=False):
    # load cif and get sim params
    spgroup_num, matname = parse_cif_path(cif_path)
    index = 1 
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
    proj_potential = process_potential(msa.potential_slices, mask=mask, normalize=True, fp16=True)
    cbed = process_cbed(msa.probes, normalize=True, fp16=True)
    
    # update sim_params dict
    sim_params = update_sim_params(sim_params, msa_cls=msa, sp_cls=sp)
   
    has_nan = np.all(np.isnan(cbed)) or np.all(np.isnan(proj_potential))
    wrong_shape = cbed.shape != (1024, 512, 512) or proj_potential.shape != (1, 512, 512)
    if has_nan or wrong_shape:
        # clean-up context and/or allocated memory
        #print('rank=%d, found this many %d nan in cbed' %(comm_rank, np.where(np.isnan(cbed)==True)[0].size))
        #print('rank=%d, found this many %d nan in proj_pot' %(comm_rank, np.where(np.isnan(proj_potential)==True)[0].size))
        #print('rank=%d, found this many %d nan in raw cbed' %(comm_rank, np.where(np.isnan(msa.probes)==True)[0].size))
        if clean_up and ctx is not None:
            msa.clean_up(ctx=ctx, vars=msa.vars)
        else:
            msa.clean_up(ctx=None, vars=msa.vars)
        return False
    else:
        # write to h5 / tfrecords / lmdb
        if isinstance(filehandle, h5py.Group):
            write_h5(filehandle, cbed, proj_potential, sim_params)
        elif isinstance(filehandle, lmdb.Transaction):
            #write_lmdb(filehandle, idx + index, cbed, proj_potential, sim_params)
            write_lmdb(filehandle, idx , cbed, proj_potential, sim_params)
        elif isinstance(filehandle, tf.python_io.TFRecordWriter):
            write_tfrecord(filehandle, cbed, proj_potential, sim_params)
        
        # clean-up context and/or allocated memory
        if clean_up and ctx is not None:
            msa.clean_up(ctx=ctx, vars=msa.vars)
        else:
            msa.clean_up(ctx=None, vars=msa.vars)
        return True

def generate_eval_data(cifpaths, outdir_path, save_mode="h5", runtime=1800*0.9):
    t = time()
    batch_num, _ = np.divmod(comm_rank, 6)
    num_sims = cifpaths.size
    #num_sims = comm_size * 5
    if comm_rank == 0:
        print('simulating evaluation data')
    if save_mode == "h5": 
    # HDF5
        h5path = os.path.join(outdir_path, 'batch_%d.h5'% comm_rank)
        if os.path.exists(h5path):
            mode ='r+'
        else:
            mode ='w'
        with h5py.File(h5path, mode=mode) as f:
            for (idx, cif_path) in enumerate(cifpaths[comm_rank:num_sims:comm_size]):
                manual = idx < ( num_sims - comm_size) 
                spgroup_num, matname = parse_cif_path(cif_path)
                try:
                    h5g = f.create_group(matname)
                except Exception as e:
                    print("rank=%d" % comm_rank, e, "group=%s exists" % matname)
                    h5g = f[matname]
                if comm_rank == 0 and bool(idx % 500):
                    print('time=%3.2f, num_sims= %d' %(time() - t, idx * comm_size))
                
                try: 
                    simulate(h5g, cif_path, gpu_id=int(np.mod(comm_rank, 6)), clean_up=manual)
                    print('rank=%d, finished simulation=%s' % (comm_rank, cif_path))
                except Exception as e:
                    print("rank=%d, skipped simulation=%s, error=%s" % (comm_rank, cif_path, format(e)))
    # TFRECORDS 
    elif save_mode == "tfrecord":
        tfrecpath = os.path.join(outdir_path, 'batch_%d.tfrecords'% comm_rank)   
        with tf.python_io.TFRecordWriter(tfrecpath) as tfrec:
            for (idx, cif_path) in enumerate(cifpaths[comm_rank:num_sims:comm_size]):
                manual = idx < ( num_sims - comm_size) 
                spgroup_num, matname = parse_cif_path(cif_path)
                if comm_rank == 0 and bool(idx % 500):
                    print('time=%3.2f, num_sims= %d' %(time() - t, idx * comm_size))
                try: 
                    status = simulate(tfrec, cif_path, gpu_id=int(np.mod(comm_rank, 6)), clean_up=manual)
                    if status:
                        print('rank=%d, finished simulation=%s' % (comm_rank, cif_path))
                    else:
                        print("rank=%d, skipped simulation=%s, error=NaN" % (comm_rank, cif_path))
                except Exception as e:
                    print("rank=%d, skipped simulation=%s, error=%s" % (comm_rank, cif_path, format(e)))

    # LMDB            
    elif save_mode == "lmdb":
        lmdbpath = os.path.join(outdir_path, 'batch_eval_%d.db' % comm_rank)
        env = lmdb.open(lmdbpath, map_size=int(100e9), map_async=True, writemap=True, create=True) # max of 100 GB
        with env.begin(write=True) as txn:
            fail = 0
            success = 0
            for (idx, cif_path) in enumerate(cifpaths[comm_rank:num_sims:comm_size]):
                manual = idx < ( num_sims - comm_size) 
                spgroup_num, matname = parse_cif_path(cif_path)
                if comm_rank == 0 and bool(idx % 500):
                    print('time=%3.2f, num_sims= %d' %(time() - t, idx * comm_size))
                if (time() - t_elaps) < runtime:
                    try:
                        status = simulate(txn, cif_path, idx=idx-fail, gpu_id=int(np.mod(comm_rank, 6)), clean_up=manual)
                        if status:
                            print('rank=%d, finished simulation=%s' % (comm_rank, cif_path))
                            env.sync()
                            success += 1
                        else:
                            print("rank=%d, skipped simulation=%s, error=NaN" % (comm_rank, cif_path))
                            fail += 1
                    except Exception as e:
                        print("rank=%d, skipped simulation=%s, error=%s" % (comm_rank, cif_path, format(e)))
                        fail += 1
                else:
                    break
            # write lmdb headers
            headers = {b"input_dtype": bytes('float16', "ascii"),
                       b"input_shape": np.array([1024,512,512]).tostring(),
                       b"output_shape": np.array([1,512,512]).tostring(),
                       b"output_dtype": bytes('float16', "ascii")}
            for key, val in headers.items():
                txn.put(key, val)
            env.sync()
        if success < 4:
            swap_out(lmdbpath)

    #comm.Barrier()            
    # time the simulation run        
    sim_t = time() - t
    if comm_rank == 0:
        print("took %3.3f seconds" % sim_t)    

def generate_training_data(cifpaths, outdir_path, save_mode="h5", runtime=1800*0.7):
    t = time()
    batch_num, _ = np.divmod(comm_rank, 6)
    num_sims = cifpaths.size
    #num_sims = comm_size * 5 
    if comm_rank == 0:
        print('simulating training data')
    if save_mode == "h5": 
    # HDF5
        h5path = os.path.join(outdir_path, 'batch_%d.h5'% comm_rank)
        if os.path.exists(h5path):
            mode ='r+'
        else:
            mode ='w'
        with h5py.File(h5path, mode=mode) as f:
            for (idx, cif_path) in enumerate(cifpaths[comm_rank:num_sims:comm_size]):
                manual = idx < ( num_sims - comm_size) 
                spgroup_num, matname = parse_cif_path(cif_path)
                try:
                    h5g = f.create_group(matname)
                except Exception as e:
                    print("rank=%d" % comm_rank, e, "group=%s exists" % matname)
                    h5g = f[matname]
                if comm_rank == 0 and bool(idx % 500):
                    print('time=%3.2f, num_sims= %d' %(time() - t, idx * comm_size))
                
                try: 
                    simulate(h5g, cif_path, gpu_id=int(np.mod(comm_rank, 6)), clean_up=manual)
                    print('rank=%d, finished simulation=%s' % (comm_rank, cif_path))
                except Exception as e:
                    print("rank=%d, skipped simulation=%s, error=%s" % (comm_rank, cif_path, format(e)))
    # TFRECORDS 
    elif save_mode == "tfrecord":
        tfrecpath = os.path.join(outdir_path, 'batch_%d.tfrecords'% comm_rank)   
        with tf.python_io.TFRecordWriter(tfrecpath) as tfrec:
            for (idx, cif_path) in enumerate(cifpaths[comm_rank:num_sims:comm_size]):
                manual = idx < ( num_sims - comm_size) 
                spgroup_num, matname = parse_cif_path(cif_path)
                if comm_rank == 0 and bool(idx % 500):
                    print('time=%3.2f, num_sims= %d' %(time() - t, idx * comm_size))
                try: 
                    status = simulate(tfrec, cif_path, gpu_id=int(np.mod(comm_rank, 6)), clean_up=manual)
                    if status:
                        print('rank=%d, finished simulation=%s' % (comm_rank, cif_path))
                    else:
                        print("rank=%d, skipped simulation=%s, error=NaN" % (comm_rank, cif_path))
                except Exception as e:
                    print("rank=%d, skipped simulation=%s, error=%s" % (comm_rank, cif_path, format(e)))

    # LMDB            
    elif save_mode == "lmdb":
        lmdbpath = os.path.join(outdir_path, 'batch_train_%d.db' % comm_rank)
        env = lmdb.open(lmdbpath, map_size=int(100e9), map_async=True, writemap=True, create=True) # max of 100 GB
        with env.begin(write=True) as txn:
            fail = 0
            success = 0
            for (idx, cif_path) in enumerate(cifpaths[comm_rank:num_sims:comm_size]):
                manual = idx < ( num_sims - comm_size) 
                spgroup_num, matname = parse_cif_path(cif_path)
                if comm_rank == 0 and bool(idx % 500):
                    print('time=%3.2f, num_sims= %d' %(time() - t, idx * comm_size))

                if (time() - t_elaps) < runtime:
                    try:
                        status = simulate(txn, cif_path, idx=idx-fail, gpu_id=int(np.mod(comm_rank, 6)), clean_up=manual)
                        if status:
                            print('rank=%d, finished simulation=%s' % (comm_rank, cif_path))
                            env.sync()
                            success += 1
                        else:
                            print("rank=%d, skipped simulation=%s, error=NaN" % (comm_rank, cif_path))
                            fail += 1
                    except Exception as e:
                        print("rank=%d, skipped simulation=%s, error=%s" % (comm_rank, cif_path, format(e)))
                        fail += 1
                else: 
                    break
            # write lmdb headers
            headers = {b"input_dtype": bytes('float16', "ascii"),
                       b"input_shape": np.array([1024,512,512]).tostring(),
                       b"output_shape": np.array([1,512,512]).tostring(),
                       b"output_dtype": bytes('float16', "ascii")}
            for key, val in headers.items():
                txn.put(key, val)
            env.sync()
        if success < 4:
            swap_out(lmdbpath)

    #comm.Barrier()            
    # time the simulation run        
    sim_t = time() - t
    if comm_rank == 0:
        print("took %3.3f seconds" % sim_t)    

def main(cifdir_path, outdir_path, save_mode, runtime=1800):
    global t_elaps
    t_elaps = time()
    cifpaths_train, cifpaths_eval= get_cif_paths(cifdir_path, ratio=0.2)
    generate_training_data(cifpaths_train, outdir_path, save_mode=save_mode, runtime=runtime*0.7)
    generate_eval_data(cifpaths_eval, outdir_path, save_mode=save_mode, runtime=runtime*0.9)
    return

if __name__ == "__main__":
    start_time = time()
    if len(sys.argv) > 2:
        cifdir_path, outdir_path, save_mode, runtime  = sys.argv[-4:]
        if save_mode not in ["h5", "tfrecord", "lmdb"]:
            print("saving format not of h5, tfrecord, lmdb")
            sys.exit()
        main(cifdir_path, outdir_path, save_mode, runtime=int(runtime))
        #comm.Barrier()
        if comm_rank == 0:
            print('Spent %2.4f s in simulation' %(time() - start_time))
        sys.exit()
    elif len(sys.argv) == 2:
        cifdir_path = sys.argv[-1]
        main_test(cifdir_path)
    else:
        print("Pass directory paths for sim input files and h5 output files")
