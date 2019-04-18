import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator

def get_kinematic_reflection(unit_cell, top=3):
    """
    Find top hkl and d_hkl with highest intensity.
    """
    xrd = XRDCalculator().get_pattern(unit_cell)
    hkls = np.array([list(itm.keys())[0] for itm in xrd.hkls])
    intens = xrd.y
    if top > intens.size:
        top = intens.size 
    top_ind = np.argsort(intens)[::-1][:top]
    hkl_vecs = hkls[top_ind]
    d_hkls = np.array(xrd.d_hkls)[top_ind]
    return hkl_vecs, d_hkls

def get_cell_orientation(z_dir):
    vecs = np.array([[1,0,0], [0,1,0], [0,0,1]])
    ortho_check = np.logical_not(np.dot(vecs,z_dir).astype('bool'))
    if vecs[ortho_check].shape[0] > 1:
        y_dir = vecs[ortho_check][0].flatten()
    elif vecs[ortho_check].shape[0] > 0:
        y_dir = vecs[ortho_check].flatten()
    elif vecs[ortho_check].shape[0] == 0:
        y_dir = vecs[0]
    return y_dir

def overlap_params(overlap, d_hkl, Lambda):
    """
    Calculate objective aperture, C_3 and defocus assuming Scherzer focus condition and three-beam CBED overlap.
    Params:
    d_hkl: d-spacing in Å.
    Lambda: wavelength in Å.
    Returns:
    objective aperture (mrad), C_3 (mm), defocus (Å).
    """
    theta_bragg = Lambda / (2 * d_hkl)
    theta_c = overlap * theta_bragg
    C_3 = (theta_c / 1.51) ** (-4) * Lambda
    defocus = -1.15 * C_3 ** (-1. / 4) * Lambda ** (-3. / 4)
    return theta_c * 1e3, C_3 * 1.e-6, defocus
