import numpy as np
import sys, os, re
from itertools import chain
from namsa.scattering import get_kinematic_reflection, get_cell_orientation, overlap_params 
from namsa.optics import voltage2Lambda
import tensorflow as tf
import h5py

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _feature_fetch(cbed, potential, params):
    '''
    Returns a tf.train.Example that is written by TFRecordwriter
    '''
    example = tf.train.Example(features=tf.train.Features(feature={

                        #labels
                        'material':_bytes_feature(np.array(params['material']).tostring()),
                        'space_group': _bytes_feature(np.array(params['space_group']).tostring()),
                        'abc': _bytes_feature(np.array(params['abc']).tostring()),
                        'angles': _bytes_feature(np.array(params['angles']).tostring()),
                        'formula': _bytes_feature(np.array(params['formula']).tostring()),

                        #potential
                        '2d_potential': _bytes_feature(potential),
          
                        #image
                        'cbed': _bytes_feature(cbed)}))
    return example 

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

def write_tfrecord(tfrecord_writer, cbed, potential, params):
    #serialize image
    cbed = cbed.tostring()
    potential = potential.tostring()
    example = _feature_fetch(cbed, potential, params)
    tfrecord_writer.write(example.SerializeToString()) 
    return

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

