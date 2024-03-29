from namsa import SupercellBuilder, MSAMPI
from namsa.utils import imageTile
import numpy as np
from time import time
import sys
import h5py
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()

#job_pid = os.environ.get('LS_JOBPID')
#job_id = os.environ.get('LSB_JOBID')
#hosts = np.array(os.environ.get('LSB_HOSTS').split(' '))
#job_hosts = np.unique(hosts)


if comm_rank == 0:
    try: 
        job_pid = os.environ.get('LS_JOBPID')
        job_id = os.environ.get('LSB_JOBID')
        hosts = np.array(os.environ.get('LSB_HOSTS').split(' '))
        job_hosts = np.unique(hosts)
        print('JOB_ID:%s, JOB_PID:%s, HOSTS:%s' %(job_id, job_pid, format(job_hosts)))
    except:
        print('NO JOB VARIABLES!!!')

cif_path = os.environ.get('CIF')
h5_path = os.environ.get('H5F')

def run(h5_file= None, step=2.5, gpu_rank=0):
    
    sp = SupercellBuilder(cif_path, verbose=False, debug=False)
    z_dir = np.array([1,1,1])
    y_dir = np.array([1,0,0])
    #sp.transform_unit_cell()
    sp.build_unit_cell()
    sp.make_orthogonal_supercell(supercell_size=np.array([4*34.6,4*34.6,198.0]),
                             projec_1=y_dir, projec_2=z_dir)
    en = 100 # keV
    semi_angle= 4e-3 # radians
    max_ang = 200e-3
    msa = MSAMPI(en, semi_angle, sp.supercell_sites, sampling=np.array([512,512]),
                 verbose=True, debug=False)
    msa.setup_device(gpu_rank=gpu_rank)
    t = time()
    msa.calc_atomic_potentials()
    slice_thickness = 3.135 #
    msa.build_potential_slices(slice_thickness)
    print('rank %d: time to build atomic potential: %2.2f' % (comm_rank, time() - t))
    aberration_params = {'C1':500., 'C3': 3.3e7, 'C5':44e7}
    probe_params = {'smooth_apert': True, 'scherzer': False, 'apert_smooth': 60, 
                'aberration_dict':aberration_params, 'spherical_phase': True}
    msa.build_probe(probe_dict=probe_params)
    t = time()
    msa.generate_probe_positions(probe_step=np.array([step,step]), 
                             probe_range=np.array([[0.25,0.75],[0.25,0.75]]))
    msa.plan_simulation()
    if h5_file is not None:
        msa.multislice(h5_write=h5_file)
    else:
        msa.multislice()
    print('rank %02d: time to propagate probes:%2.2f' % (comm_rank, time() - t))
    if comm_rank == 0: print('Simulation Finished Successfully!')
    return
    
if __name__ == '__main__':
    gpu_rank = int(np.mod(comm_rank,6))
    if len(sys.argv) == 2:
        step = float(sys.argv[-1])
    if len(sys.argv) == 3:
        step = float(sys.argv[-2])
        write = bool(int(sys.argv[-1]))
        if write:
            with h5py.File(h5_path, driver='mpio', mode='w', comm=MPI.COMM_WORLD, libver='latest') as f:
                f.atomic = False
                run(h5_file=f, step=step, gpu_rank=gpu_rank)
        else:
            run(step=step, gpu_rank=gpu_rank)
    else:
        run(step=2.5, gpu_rank=gpu_rank)

