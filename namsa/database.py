import numpy as np
import os
import namsa

#TODO setup config file for custom scattering_database location

kirkland_path = os.path.join(namsa.__path__[0], 'scattering_database/kirkland_params.npy')
kirkland_params = np.load(kirkland_path)
