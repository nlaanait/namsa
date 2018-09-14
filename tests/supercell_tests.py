from namsa.supercell import SupercellBuilder
import unittest
import numpy as np


class TestSupercellBuilder(unittest.TestCase):

    def test_unitcell_cif_parse(self):
        cifpath = 'cif_files/mvc-9418.cif'
        spbuild = SupercellBuilder(cifpath, debug=False)
        spbuild.build_unit_cell()
        test_xyz = spbuild.xyz_positions
        true_xyz = np.loadtxt('cif_files/mvc-9418_P_1_1_1.xyz', skiprows=2, dtype=spbuild.xyz_dtype)
        true_pos = np.round(np.column_stack([true_xyz['x'], true_xyz['y'], true_xyz['z']]), 2)
        test_pos = np.round(np.column_stack([test_xyz['x'], test_xyz['y'], test_xyz['z']]), 2)
        true_elem = true_xyz['atom_type']
        test_elem = test_xyz['atom_type']
        check_pos = true_pos == test_pos
        check_elem = true_elem == test_elem
        self.assertTrue(check_pos.all(),msg='x,y,z positions do not match')
        self.assertTrue(check_elem.all(), msg='atom types do not match')

    def test_unitcell_transform(self):
        cifpath = 'cif_files/mvc-9418.cif'
        spbuild = SupercellBuilder(cifpath, debug=False)
        spbuild.transform_unit_cell(centering_type='A', scaling=np.array([3, 3, 3]))
        spbuild.build_unit_cell()
        test_xyz = spbuild.xyz_positions
        true_xyz = np.loadtxt('cif_files/mvc-9418_A_3_3_3.xyz', skiprows=2, dtype=spbuild.xyz_dtype)
        true_pos = np.column_stack([true_xyz['x'], true_xyz['y'], true_xyz['z']])
        test_pos = np.column_stack([test_xyz['x'], test_xyz['y'], test_xyz['z']])
        true_elem = true_xyz['atom_type']
        test_elem = test_xyz['atom_type']
        check_pos = np.sum(true_pos - test_pos)
        check_elem = true_elem == test_elem
        self.assertLessEqual(check_pos, true_xyz.size*0.01, msg='x,y,z positions do not match')
        self.assertTrue(check_elem.all(), msg='atom types do not match')

    def test_orthogonal_supercell(self):
        cifpath = 'cif_files/mvc-9418.cif'
        spbuild = SupercellBuilder(cifpath, debug=False)
        spbuild.transform_unit_cell()
        spbuild.build_unit_cell()
        spbuild.make_orthogonal_supercell()
        test_xyz = spbuild.supercell_xyz_positions
        true_xyz = np.loadtxt('cif_files/supercell_mvc-9418.xyz', skiprows=2, dtype=spbuild.xyz_dtype)
        true_pos = np.column_stack([true_xyz['x'], true_xyz['y'], true_xyz['z']])
        test_pos = np.column_stack([test_xyz['x'], test_xyz['y'], test_xyz['z']])
        true_elem = true_xyz['atom_type']
        test_elem = test_xyz['atom_type']
        check_pos = np.sum(true_pos - test_pos)
        check_elem = true_elem == test_elem
        self.assertLessEqual(check_pos, true_xyz.size * 0.01, msg='x,y,z positions do not match')
        self.assertTrue(check_elem.all(), msg='atom types do not match')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSupercellBuilder)
    unittest.TextTestRunner(verbosity=3).run(suite)



