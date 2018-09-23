import numpy as np
from scipy.constants import speed_of_light, electron_mass, elementary_charge

def voltage2Lambda(voltage):
    """
    Energy to Wavelength conversion.
    Params:
    voltage: accelerating voltage in V.
    Returns:
    wavelength in Å.
    """
    return 12.2643 / np.sqrt(voltage + 0.97845e-6 * voltage ** 2)


def transverse_width(Lambda, theta_s):
    """
    Transverse Coherence Width (Zernike-Van Cittert Theorem).
    Params:
    Lambda: electron wavelength in Å.
    theta_s: convergence semi-angle of source (or image of source) in radians.
    Returns:
    transverse coherence width in Å.
    """

    return Lambda / (2 * np.pi * theta_s)


def no_overlap_dhkl(Lambda, C_3=1):
    """
    Largest d-spacing producing no CBED order overlap if the probe size is diffraction limited.
    Params:
    Lambda: electron wavelength in Å
    C_3: spherical aberration constant in mm, default 1.
    Returns:
    largest d_hkl without CBED order overlap.
    """
    C_3 *= 1.e7
    return 0.7874 * C_3 ** (1. / 4) * Lambda ** (3. / 4)


def semi_angle_limit(Lambda, C_3=1):
    """
    Convergence semi-angle if the probe size is diffraction limited.
    Params:
    Lambda: electron wavelength in Å
    C_3: spherical aberration constant in mm, default 1.
    Returns:
    Convergence semi-angle in mrad.
    """
    C_3 *= 1.e7
    return 1.51 * C_3 ** (-1. / 4) * Lambda ** (1. / 4) * 1e3


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

def scherzer_params(Lambda, Cs):
    defocus = np.sqrt(1.5) * np.sqrt(Cs * Lambda)
    aperture = (6 * Lambda / Cs) ** 1/4
    return defocus, aperture

def spherical_phase_error(k_rad, Lambda, scherzer, **kwargs):
    """
    Return phase error due to rotation invariant aberration function.
    :param k_rad: radial K-vector.
    :param Lambda: wavelength
    :param scherzer: bool, True: use Scherzer defocus.
    :param kwargs: keys: 'C1': |defocus|; 'C3': spherical aberration, 3rd Cs; 'C5': 5th-order Cs. If not specified, no
    aberrations.
    :return: exp(-i * chi), chi is aberration function.
    """

    # C1 = kwargs.get('C1', 0.)
    C1 = kwargs['C1']
    C3 = kwargs['C3']
    C5 = kwargs['C5']
    if scherzer:
        if C3 > 1.0:
            C1, _ = scherzer_params(Lambda, C3)
        else:
            print('Spherical Aberration is too small!! Not using Scherzer condition.')
    chi = 2 * np.pi / Lambda * (-1/2 * C1 * (k_rad * Lambda) ** 2 + 1/4 * C3 * (k_rad * Lambda) ** 4 + 1/6 * C5 *
                                (k_rad * Lambda) ** 6)
    return np.exp(-1.j * chi)


def sigma_int(V):
    """
    Interaction Parameter (units of radians/eV/Å)
    Params:
    V: accelerating voltage in Volts
    """
    sigma = 2*np.pi*(electron_mass*speed_of_light**2 + elementary_charge*V)/(2*electron_mass*speed_of_light**2 +
                                                                             elementary_charge*V)/(voltage2Lambda(V)*V)
    return sigma
