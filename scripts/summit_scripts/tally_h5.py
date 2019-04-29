import os, sys
import subprocess, shlex
import h5py 
import multiprocessing as mp
import numpy as np

def read_h5(args):
    h5_path, delete = args[:]
    mode = 'r'
    with h5py.File(h5_path, mode=mode) as f:
        num_samples = int(len(list(f.items()))//2)
    return (num_samples, h5_path)


def main(file_dir, delete=False):
    files = os.listdir(file_dir)
    paths = [os.path.join(file_dir, path) for path in files]
    processes = min(mp.cpu_count(), len(files))
    pool = mp.Pool(processes=processes)
    tasks = [(path, delete) for path in paths]
    chunk = max(np.int(np.floor(len(files) / processes)), 1)
    jobs = pool.imap(read_h5, tasks, chunksize=chunk)
    tally = [j for j in jobs]
    pool.close()
    tally = np.array(tally,  dtype=[('num_samples', 'i4'), ('filepath', np.dtype('U100'))])
    np.save('tally_%s.npy' % file_dir.split('/')[-1], tally )
    def get_paths(mode='_train_'):
        mask = np.array([itm.find(mode) for itm in tally['filepath']])
        mask[mask >= 0] = 1 
        mask[mask < 0] = 0 
        mask = mask.astype(np.bool)
        mode_files = tally[mask]
        return mode_files
    train_files = get_paths()
    test_files = get_paths(mode='_test_')
    dev_files = get_paths(mode='_dev_')
    for name, arr_files in zip(['train', 'test', 'dev'], [train_files, test_files, dev_files]): 
        if arr_files['num_samples'].size != 0:
            print("stats of %s samples (total, min, max, mean): %d, %2.2f, %2.2f, %2.2f" %(name, arr_files['num_samples'].sum(), 
                arr_files['num_samples'].min(), arr_files['num_samples'].max(), arr_files['num_samples'].mean()))
    

if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[-2], delete=bool(int(sys.argv[-1])))
    else:
        main(sys.argv[-1])
        print('DONE')
