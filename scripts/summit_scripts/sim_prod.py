from namsa import SupercellBuilder, MSAGPU, setup_device
from namsa.utils import dhkl_spacing
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
import shutil

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()

global counter
counter = 0

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


def simulate(filehandle, cif_path, idx= None, gpu_ctx=None, clean_up=False, record_names=["2d_potential_", "cbed_"]):
    # load cif and get sim params
    spgroup_num, matname = parse_cif_path(cif_path)
    index = 0 
    sp = SupercellBuilder(cif_path, verbose=False, debug=False)
    latts = np.array(sp.structure.lattice.abc)
    if np.any(latts >= 10.) or np.all(latts >= 7):
        print('rank=%d, skipped simulation=%s, latt. const. too large=%s' % (comm_rank, cif_path, format(latts)))
        return False, None   
    angles = np.array(sp.structure.lattice.angles)
    angles = np.round(angles).astype(np.int)
    cutoff = np.array([90,90,90])
    tol = 2
    cubic_cond = np.logical_not(np.logical_and(angles > cutoff - tol, angles < cutoff + tol)).any()
    hexag_cond_1 = np.logical_and(angles[:2] > cutoff[:2] - tol, angles[:2] < cutoff[:2] + tol).any()
    hexag_cond_2 = np.logical_and(angles[-1] > 120 - tol, angles[-1] < 120 + tol)
    hexag_cond = np.logical_not(hexag_cond_1 and hexag_cond_2)
    if cubic_cond:
        if hexag_cond:
            print("rank=%d, skipped simulation=%s, msg=not cubic/hexagonal" % (comm_rank, cif_path.split('/')[-2:]))
            return False, None
        else:
            pass
        
    sim_params = get_sim_params(sp, slab_t= 100, cell_dim = 50, grid_steps=np.array([8,8]), orientation_num=5, 
                                sampling=np.array([256,256]))
    cell_dim = sim_params['cell_dim']
    slab_t = sim_params['slab_t']
    sim_params['space_group']= spgroup_num
    sim_params['material'] = matname
    energies = np.linspace(100,200,4)
    np.random.shuffle(energies)
    counter = 0
    for energy in energies:
        for index in range(len(sim_params['z_dirs'])): 
            # build supercell
            z_dir = sim_params['z_dirs'][index]
            y_dir = sim_params['y_dirs'][index]
            sp.build_unit_cell()
            sp.make_orthogonal_supercell(supercell_size=np.array([cell_dim,cell_dim,slab_t]),
                                     projec_1=y_dir, projec_2=z_dir)

            # set simulation params
            slice_thickness = sim_params['d_hkl'][index]
#             energy = sim_params['energy']
            semi_angle= sim_params['semi_angles'][index]
            probe_params = sim_params['probe_params']
            sampling = sim_params['sampling']
            grid_steps = sim_params['grid_steps']

            ####### mods ######
        #     slice_thickness = max(1.0, min(5.0, get_slice_thickness(sp, direc=np.array([0,0,1]))))
        #     semi_angle = 0.01
        #     energy = np.random.randint(60,140)
            probe_params['aberration_dict']['C3'] = np.round(10**(np.random.rand()*7))
            ##################

            # simulate
            msa = MSAGPU(energy, semi_angle, sp.supercell_sites, sampling=sampling,
                         verbose=False, debug=False)
            msa.calc_atomic_potentials()
            msa.build_potential_slices(gpu_ctx, slice_thickness)
            msa.build_probe(probe_dict=probe_params)
            msa.generate_probe_positions(grid_steps=grid_steps) 
            msa.plan_simulation()
            msa.multislice(bandwidth=2./3)

            # process cbed and potential
            mask = msa.bandwidth_limit_mask(sampling, radius=1./3).astype(np.bool)
            proj_potential = process_potential(msa.potential_slices, mask=mask, normalize=True, fp16=True)
            cbed = process_cbed(msa.probes, normalize=True, fp16=True, new_shape=(64,256,256))

            # update sim_params dict
            sim_params = update_sim_params(sim_params, msa_cls=msa, sp_cls=sp)

            has_nan = np.all(np.isnan(cbed)) or np.all(np.isnan(proj_potential))
            wrong_shape = cbed.shape != (64, 256, 256) or proj_potential.shape != (1, 128, 128)
            if has_nan or wrong_shape:
                # clean-up context and/or allocated memory
                print('rank=%d, found this many %d nan in cbed' %(comm_rank, np.where(np.isnan(cbed)==True)[0].size))
                print('rank=%d, found this many %d nan in proj_pot' %(comm_rank, np.where(np.isnan(proj_potential)==True)[0].size))
                pass
            else:
                # write to h5 / tfrecords / lmdb
                if isinstance(filehandle, h5py.Group):
                    write_h5(filehandle, cbed, proj_potential, sim_params)
                elif isinstance(filehandle, lmdb.Transaction):
                    write_lmdb(filehandle, idx + counter , cbed, proj_potential, record_names=record_names)
                    print('rank=%d, wrote sim_index=%d' % (comm_rank, idx+counter))
                    counter += 1
                elif isinstance(filehandle, tf.python_io.TFRecordWriter):
                    write_tfrecord(filehandle, cbed, proj_potential, sim_params)

            # free-up gpu memory
            msa.clean_up(ctx=None, vars=msa.vars)
    return True, counter
    
def generate_data(cifpaths, outdir_path, save_mode="h5", data_mode="train", gpu_ctx=None, runtime=1800*0.9):
    t = time()
    num_sims = cifpaths.size
    if comm_rank == 0:
        print('simulating evaluation data')
    if save_mode == "h5": 
    # HDF5
        h5path = os.path.join(outdir_path, 'batch_%s_%d.h5'% (data_mode, comm_rank))
        if os.path.exists(h5path):
            mode ='r+'
        else:
            mode ='w'
        with h5py.File(h5path, mode='w') as f:
            for (idx, cif_path) in enumerate(cifpaths[comm_rank:num_sims:comm_size]):
                manual = idx < ( num_sims - comm_size) 
                spgroup_num, matname = parse_cif_path(cif_path)
                if (time() - t_elaps) < runtime:
                    try:
                        h5g = f.create_group(matname)
                    except Exception as e:
                        print("rank=%d" % comm_rank, e, "group=%s exists" % matname)
                        h5g = f[matname]
                    if comm_rank == 0 and bool(idx % 500):
                        print('time=%3.2f, num_sims= %d' %(time() - t, idx * comm_size))

                    try: 
                        status, _ = simulate(h5g, cif_path, gpu_ctx=gpu_ctx, clean_up=manual)
                        if status:
                            print('rank=%d, finished simulation=%s' % (comm_rank, cif_path.split('/')[-2:]))
                            f.flush()
                        else:
                            pass
                    except Exception as e:
                        print("rank=%d, skipped simulation=%s, error=%s" % (comm_rank, cif_path.split('/')[-2:], format(e)))
                else:
                    f.flush()
                    break
    # TFRECORDS 
    elif save_mode == "tfrecord":
        tfrecpath = os.path.join(outdir_path, 'batch_%s_%d.tfrecords'% (data_mode, comm_rank))   
        with tf.python_io.TFRecordWriter(tfrecpath) as tfrec:
            for (idx, cif_path) in enumerate(cifpaths[comm_rank:num_sims:comm_size]):
                manual = idx < ( num_sims - comm_size) 
                spgroup_num, matname = parse_cif_path(cif_path)
                if comm_rank == 0 and bool(idx % 500):
                    print('time=%3.2f, num_sims= %d' %(time() - t, idx * comm_size))
                try: 
                    status, _ = simulate(tfrec, cif_path, gpu_ctx=gpu_ctx, clean_up=manual)
                    if status:
                        print('rank=%d, finished simulation=%s' % (comm_rank, cif_path.split('/')[-2:]))
                    else:
                        pass
                        print("rank=%d, skipped simulation=%s, error=NaN" % (comm_rank, cif_path.split('/')[-2:]))
                except Exception as e:
                    print("rank=%d, skipped simulation=%s, error=%s" % (comm_rank, cif_path.split('/')[-2:], format(e)))

    # LMDB            
    elif save_mode == "lmdb":
        lmdbpath = os.path.join(outdir_path, 'batch_%s_%d.db' % (data_mode, comm_rank))
        env = lmdb.open(lmdbpath, map_size=int(10e12), map_async=True, writemap=True, create=True) # max of 100 GB
        with env.begin(write=True) as txn:
            # write lmdb headers
            record_names = ["2d_potential_", "cbed_"]
            headers = {b"input_dtype": bytes('float16', "ascii"),
                       b"input_shape": np.array([64,256,256]).tostring(),
                       b"output_shape": np.array([1,128,128]).tostring(),
                       b"output_dtype": bytes('float16', "ascii"),
                       b"output_name": bytes(record_names[0], "ascii"),
                       b"input_name": bytes(record_names[1], "ascii")}
            for key, val in headers.items():
                txn.put(key, val)
            txn.put(b"header_entries", bytes(len(list(headers.items()))))
            env.sync()
            
            # start simulation
            fail = 0
            success = 0
            counter = 0
            for (idx, cif_path) in enumerate(cifpaths[comm_rank:num_sims:comm_size]):
                manual = idx < ( num_sims - comm_size) 
                spgroup_num, matname = parse_cif_path(cif_path)
                if comm_rank == 0 and bool(idx % 500):
                    print('time=%3.2f, num_sims= %d' %(time() - t, idx * comm_size))
                if (time() - t_elaps) < runtime:
                    try:
                        status, write_counter = simulate(txn, cif_path, idx=counter, gpu_ctx=gpu_ctx, clean_up=manual)
                        if status:
                            print('rank=%d, finished simulation=%s' % (comm_rank, cif_path.split('/')[-2:]))
#                             print('rank=%d, counter=%s' % (comm_rank, counter))
                            env.sync()
                            success += 1
                            counter += write_counter
                        else:
                            fail += 1
#                             print('rank=%d, counter=%s' % (comm_rank, counter))
                    except Exception as e:
                        print("rank=%d, skipped simulation=%s, error=%s" % (comm_rank, cif_path.split('/')[-2:], format(e)))
#                         print('rank=%d, counter=%s' % (comm_rank, counter))
                        fail += 1
                else:
                    env.sync()
                    break

    #comm.Barrier()            
    # time the simulation run        
    sim_t = time() - t
    if comm_rank == 0:
        print("took %3.3f seconds" % sim_t)    
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

def main(cifdir_path, outdir_path, save_mode, runtime=1800):
    global t_elaps
    t_elaps = time()
    cif_paths = get_cif_paths(cifdir_path)
    samples_train, samples_dev, samples_test = get_samples(cif_paths, ratio=0.95)
    ctx = setup_device(gpu_id=int(np.mod(comm_rank, 6)))
    generate_data(samples_train, outdir_path, save_mode=save_mode, data_mode='train', runtime=runtime*0.95, gpu_ctx=ctx)
    print('rank=%d, finished simulating training data' % comm_rank)
#     generate_data(samples_dev, outdir_path, save_mode=save_mode, data_mode='dev', runtime=runtime*0.9, gpu_ctx=ctx)
#     print('rank=%d, finished simulating dev data' % comm_rank)
    generate_data(samples_test, outdir_path, save_mode=save_mode, data_mode='test', runtime=runtime, gpu_ctx=ctx)
    print('rank=%d, finished simulating test data' % comm_rank)
    return

if __name__ == "__main__":
    start_time = time()
    if len(sys.argv) > 2:
        cifdir_path, outdir_path, save_mode, runtime  = sys.argv[-4:]
        if save_mode not in ["h5", "tfrecord", "lmdb"]:
            print("saving format not of h5, tfrecord, lmdb")
            sys.exit()
        main(cifdir_path, outdir_path, save_mode, runtime=int(runtime))
        comm.Barrier()
        if int(np.mod(comm_rank, 6)) == 0:
            usage = shutil.disk_usage(outdir_path).used // 1024e6
            mpi_host = MPI.Get_processor_name()
            print('nvme on node: %s, disk used: %2.3f GB, contents:%s'% (mpi_host, usage, os.listdir(outdir_path)))
            print('DONE...')
    elif len(sys.argv) == 2:
        cifdir_path = sys.argv[-1]
        main_test(cifdir_path)
    else:
        print("Pass directory paths for sim input files and h5 output files")
