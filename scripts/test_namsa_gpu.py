#!/home/nl7/anaconda3/bin/python
from namsa import SupercellBuilder, MSAGPU
from namsa.utils import imageTile
import numpy as np
from time import time
import sys

def run(step=2.5, gpu_rank=0):
    
    sp = SupercellBuilder('XYZ_files/Si.cif')
    z_dir = np.array([1,1,1])
    y_dir = np.array([1,0,0])
    #sp.transform_unit_cell()
    sp.build_unit_cell()
    sp.make_orthogonal_supercell(supercell_size=np.array([4*34.6,4*34.6,489.0]),
                             projec_1=y_dir, projec_2=z_dir)
    en = 100 # keV
    semi_angle= 4e-3 # radians
    max_ang = 200e-3
    msa = MSAGPU(en, semi_angle, sp.supercell_sites, sampling=np.array([512,512])) 
          #verbose=True, debug=False)
    msa.setup_device(gpu_rank=gpu_rank)
    t = time()
    msa.calc_atomic_potentials()
    slice_thickness = 2.0 #
    msa.build_potential_slices(slice_thickness)
    print('time to build atomic potential: %2.2f' % (time() - t))
    aberration_params = {'C1':500., 'C3': 3.3e7, 'C5':44e7}
    probe_params = {'smooth_apert': True, 'scherzer': False, 'apert_smooth': 60, 
                'aberration_dict':aberration_params, 'spherical_phase': True}
    msa.build_probe(probe_dict=probe_params)
    t = time()
    msa.generate_probe_positions(probe_step=np.array([step,step]), 
                             probe_range=np.array([[0.25,0.75],[0.25,0.75]]))
    msa.plan_simulation()
    msa.multislice()
    print('time to propagate probes:%2.2f' % (time() - t)) 
    
if __name__ == '__main__':
    if len(sys.argv) == 2:
        step = float(sys.argv[-1])
        run(step=step)
    elif len(sys.argv) == 3:
        gpu_rank = int(sys.argv[-1])  
        step = float(sys.argv[-2])
        #print('step %2.3f, rank %d'%(step, gpu_rank))
        run(step=step, gpu_rank=gpu_rank)
    elif len(sys.argv) == 1:
        run(step=2.5)
