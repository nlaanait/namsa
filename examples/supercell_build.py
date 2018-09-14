from namsa import SupercellBuilder


def main():
    cifpath = 'cif_files/mvc-9418.cif'
    spbuild = SupercellBuilder(cifpath, verbose=True)
    spbuild.transform_unit_cell()
    spbuild.build_unit_cell()
    spbuild.make_orthogonal_supercell(xyz='examples/test.xyz')


if __name__ == '__main__':
    main()