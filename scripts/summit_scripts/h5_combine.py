#from namsa import SupercellBuilder, MSAGPU
#from utils import *
import numpy as np
from time import time
import sys, os, re
import h5py
#from mpi4py import MPI
#from itertools import chain
#import tensorflow as tf
#import lmdb

#comm = MPI.COMM_WORLD
#comm_size = comm.Get_size()
#comm_rank = comm.Get_rank()


def main(h5_trg, h5_dir):
    h5_paths = [os.path.join(h5_dir, itm) for itm in os.listdir(h5_dir)] 
    #if os.path.exists(h5path):
    #    mode ='r+'
    #else:
    #    mode ='w'
    mode = 'w'
    i = 0
    with h5py.File(h5_trg, mode=mode) as f_trg:
        for h5_path_src in h5_paths:
            try:
                with h5py.File(h5_path_src, mode='r') as f_src:
                    for (_, g_trg) in f_src.items():
                        #print([itm for _, itm in g_trg.attrs.items()])
                        f_src.copy(g_trg,f_trg['/'], name='sample_%d'%i)
                        i +=1
                print('copied contents of %s' % h5_path_src)
            except Exception as e:
                print('skipped %s, error=%s' % (h5_path_src, format(e)))

if __name__ == "__main__":
    if len(sys.argv) > 2:
        h5_trg, h5_dir = sys.argv[-2:]
        main(h5_trg, h5_dir)
    else:
        print("Pass directory paths for sim input files and h5 output files")
