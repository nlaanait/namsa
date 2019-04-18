from namsa import SupercellBuilder, MSAGPU
from utils import *
import numpy as np
from time import time
import sys, os, re, subprocess, shlex, shutil
import h5py
from mpi4py import MPI
from itertools import chain
import tensorflow as tf
import lmdb
import numpy as np

#print(os.environ)

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
    lmdb_repo = "/gpfs/alpine/lrn001/proj-shared/nl/sims/data/lmdb_bank_64_256_256"
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


def simulate(filehandle, h5g, idx= None, gpu_id=0, record_names=["2d_potential_", "cbed_"], clean_up=False):
    try:
        # load cif and get sim params
        cif_path = h5g.attrs['cif_path'].decode('ascii')
        z_dir = [0,0,1]
        y_dir = np.array([[1,0,0],[0,1,0]])[np.random.randint(2)]
        #z_dir = h5g['z_dir'][()]
        #y_dir = h5g['y_dir'][()]
        slice_thickness = h5g.attrs['d_hkl']
        #semi_angle = h5g.attrs['semi_angle']
        semi_angle = 0.01
        sampling = np.array([256,256])
        cell_dim = 50 
        slab_t = 200
        energy = 100e3
        grid_steps = np.array([8,8])
        probe_params = {'smooth_apert': True, 'scherzer': False, 'apert_smooth': 30, 
                'aberration_dict':{'C1':0., 'C3':0 , 'C5':0.}, 'spherical_phase': True}
        semi_angle = 0.01
        energy = np.random.randint(60,140)
        probe_params['aberration_dict']['C3'] = np.round(10**(np.random.rand()*7))
        # build supercell
        sp = SupercellBuilder(cif_path, verbose=False, debug=False)
        slice_thickness = max(1.0, min(5.0, get_slice_thickness(sp, direc=np.array([0,0,1]))))
        # filter out 
        angles = np.array(sp.structure.lattice.angles)
        angles = np.round(angles).astype(np.int)
        cutoff = np.array([90,90,90])
        tol = 3
        cubic_cond = np.logical_not(np.logical_and(angles > cutoff - tol, angles < cutoff + tol)).any()
        hexag_cond_1 = np.logical_and(angles[:2] > cutoff[:2] - tol, angles[:2] < cutoff[:2] + tol).any()
        hexag_cond_2 = np.logical_and(angles[-1] > 120 - tol, angles[-1] < 120 + tol)
        hexag_cond = np.logical_not(hexag_cond_1 and hexag_cond_2)
        if cubic_cond:
            if hexag_cond:
                return False
            else:
                pass
        
        sp.build_unit_cell()
        sp.make_orthogonal_supercell(supercell_size=np.array([cell_dim,cell_dim,slab_t]),
                             projec_1=y_dir, projec_2=z_dir)
    
        # simulate
        msa = MSAGPU(energy, semi_angle, sp.supercell_sites, sampling=sampling,
                 verbose=False, debug=False)
        ctx = msa.setup_device(gpu_rank=gpu_id)
        msa.calc_atomic_potentials()
        msa.build_potential_slices(slice_thickness)
        msa.build_probe(probe_dict=probe_params)
        msa.generate_probe_positions(grid_steps=grid_steps) 
        msa.plan_simulation()
        msa.multislice(bandwidth=1.)
    
        # process cbed and potential
        #mask = msa.bandwidth_limit_mask(sampling, radius=1./3).astype(np.bool)
        proj_potential = process_potential(msa.potential_slices, sampling=sampling, mask=None, normalize=True, fp16=True)
        cbed = process_cbed(msa.probes, normalize=True, fp16=True)
        
        # 
        del sp

        # clean-up context and/or allocated memory
        if clean_up and ctx is not None:
            msa.clean_up(ctx=ctx, vars=msa.vars)
            del msa
        else:
            msa.clean_up(ctx=None, vars=msa.vars)
            del msa

        # check data integrity
        has_nan = np.all(np.isnan(cbed)) or np.all(np.isnan(proj_potential))
        true_cbed_shape = (np.prod(grid_steps),) + tuple(sampling)
        true_pot_shape = (1,) + tuple(sampling)
        wrong_shape = cbed.shape != true_cbed_shape or proj_potential.shape != true_pot_shape 
        if has_nan or wrong_shape:
            print("rank=%d, skipped simulation=%s, error=NaN" % (comm_rank, cif_path))
            return False
        else:
            # write to h5 / tfrecords / lmdb
            if isinstance(filehandle, lmdb.Transaction):
                write_lmdb(filehandle, idx , cbed, proj_potential, record_names=record_names)
            print('rank=%d, finished simulation=%s' % (comm_rank, cif_path))
            return True
    except Exception as e:
        print("rank=%d, skipped simulation=%s, error=%s" % (comm_rank, cif_path, format(e)))
        return False
    finally:
        # clean-up context and/or allocated memory
        try:
            if clean_up and ctx is not None:
                msa.clean_up(ctx=ctx, vars=msa.vars)
                del msa
            else:
                msa.clean_up(ctx=None, vars=msa.vars)
                del msa
        except:
            pass


def generate_eval_data(samples, h5_params, outdir_path, save_mode="h5", runtime=1800*0.9):
    t = time()
    batch_num, _ = np.divmod(comm_rank, 6)
    num_sims = samples.size
    #num_sims = comm_size * 5
    print('rank_%d:simulating eval data' %comm_rank)
    # LMDB            
    if save_mode == "lmdb":
        lmdbpath = os.path.join(outdir_path, 'batch_eval_%d.db' % comm_rank)
        env = lmdb.open(lmdbpath, map_size=int(100e9), map_async=True, writemap=True, create=True) # max of 100 GB
        with env.begin(write=True) as txn:
            # write lmdb headers
            record_names = ["2d_potential_", "cbed_"]
            headers = {b"input_dtype": bytes('float16', "ascii"),
                       b"input_shape": np.array([64,256,256]).tostring(),
                       b"output_shape": np.array([1,256,256]).tostring(),
                       b"output_dtype": bytes('float16', "ascii"),
                       b"output_name": bytes(record_names[0], "ascii"),
                       b"input_name": bytes(record_names[1], "ascii")}
            for key, val in headers.items():
                txn.put(key, val)
            env.sync()
            # start simulation
            fail = 0
            success = 0
            for (idx, sample) in enumerate(samples[comm_rank:num_sims:comm_size]):
                h5g = h5_params[sample]
                manual = idx < ( num_sims - comm_size) 
                if comm_rank == 0 and bool(idx % 500):
                    print('time=%3.2f, num_sims= %d' %(time() - t, idx * comm_size))
                if (time() - t_elaps) < runtime: 
                    #try:
                    status = simulate(txn, h5g, idx=idx-fail, gpu_id=int(np.mod(comm_rank, 6)), clean_up=manual,
                            record_names=record_names)
                    if status:
                        env.sync()
                        success += 1
                    else:
                        fail += 1
                else: 
                    break
        if success < 2:
            swap_out(lmdbpath)
    #comm.Barrier()            
    # time the simulation run        
    sim_t = time() - t
    if comm_rank == 0:
        print("took %3.3f seconds" % sim_t)    
    return

def generate_training_data(samples, h5_params, outdir_path, save_mode="h5", runtime=1800*0.7):
    t = time()
    batch_num, _ = np.divmod(comm_rank, 6)
    num_sims = samples.size
    #num_sims = comm_size * 5 
    print('rank_%d:simulating training data' %comm_rank)
    # LMDB            
    if save_mode == "lmdb":
        lmdbpath = os.path.join(outdir_path, 'batch_train_%d.db' % comm_rank)
        env = lmdb.open(lmdbpath, map_size=int(100e9), map_async=True, writemap=True, create=True) # max of 100 GB
        with env.begin(write=True) as txn:
            # write lmdb headers
            record_names = ["2d_potential_", "cbed_"]
            headers = {b"input_dtype": bytes('float16', "ascii"),
                       b"input_shape": np.array([64,256,256]).tostring(),
                       b"output_shape": np.array([1,256,256]).tostring(),
                       b"output_dtype": bytes('float16', "ascii"),
                       b"output_name": bytes(record_names[0], "ascii"),
                       b"input_name": bytes(record_names[1], "ascii")}
            for key, val in headers.items():
                txn.put(key, val)
            env.sync()

            # simulate
            fail = 0
            success = 0
            for (idx, sample) in enumerate(samples[comm_rank:num_sims:comm_size]):
                manual = idx < ( num_sims - comm_size) 
                h5g = h5_params[sample]
                #print(list(h5g.items()))
                if comm_rank == 0 and bool(idx % 500):
                    print('time=%3.2f, num_sims= %d' %(time() - t, idx * comm_size))

                if (time() - t_elaps) < runtime:
                    status = simulate(txn, h5g, idx=idx-fail, gpu_id=int(np.mod(comm_rank, 6)), clean_up=manual, 
                            record_names=record_names)
                    if status:
                        env.sync()
                        success += 1
                    else:
                        fail += 1
                else:
                    break
        if success < 4:
            swap_out(lmdbpath)
    #comm.Barrier()            
    # time the simulation run        
    sim_t = time() - t
    if comm_rank == 0:
        print("took %3.3f seconds" % sim_t)    
    return

def get_samples(h5_params, ratio=0.8, time_std=2):
    samples = np.array(list(h5_params.keys()))
    np.random.shuffle(samples)
    times = np.array([h5_params[sample].attrs['time'] for sample in samples])
    mean_time = times.mean()
    std_time = times.std()
    cutoff_time = times.mean() + time_std * times.std()
    samples = samples[times < cutoff_time] 
    if ratio is not None:
        train_size = int(samples.size * ratio)
        samples_train = samples[:train_size]
        samples_test = samples[train_size:]
        return samples_train, samples_test
    return samples

def main(h5_params_file, outdir_path, save_mode, runtime=1800):
    global t_elaps
    t_elaps = time()
    with h5py.File(h5_params_file, mode='r') as h5_params:
        samples_train, samples_test = get_samples(h5_params, ratio=0.9, time_std=3)
        generate_training_data(samples_train, h5_params, outdir_path, save_mode=save_mode, runtime=runtime*0.9)
        generate_eval_data(samples_test, h5_params, outdir_path, save_mode=save_mode, runtime=runtime)
    return

if __name__ == "__main__":
    start_time = time()
    if len(sys.argv) > 2:
        h5_params_path, outdir_path, save_mode, runtime  = sys.argv[-4:]
        if save_mode not in ["h5", "tfrecord", "lmdb"]:
            print("saving format not of h5, tfrecord, lmdb")
            sys.exit()
        main(h5_params_path, outdir_path, save_mode, runtime=int(runtime))
        #comm.Barrier()
        print('Spent %2.4f s in simulation' %(time() - start_time))
        comm.Barrier()
        if int(np.mod(comm_rank, 6)) == 0:
            user = os.environ.get('USER')
            nvme_dir = '/mnt/bb/%s' % user
            usage = shutil.disk_usage(nvme_dir).used // 1024e6
            mpi_host = MPI.Get_processor_name()
            print('nvme on node: %s, disk used: %2.3f GB, contents:%s'% (mpi_host, usage, os.listdir(nvme_dir)))
            print('DONE...')
        #if comm_rank == 0:
        #    print('Spent %2.4f s in simulation' %(time() - start_time))
        #sys.exit()
    elif len(sys.argv) == 2:
        cifdir_path = sys.argv[-1]
        main_test(cifdir_path)
    else:
        print("Pass directory paths for sim input files and h5 output files")
