import os, sys
import subprocess, shlex
import lmdb
import multiprocessing as mp
import numpy as np

def read_lmdb(args):
    lmdb_path, delete = args[:]
    env = lmdb.open(lmdb_path, readahead=False, readonly=True, writemap=False, lock=False)
    num_samples = env.stat()['entries'] - 4 ## TODO: remove hard-coded # of headers by storing #_headers key
    num_samples = num_samples//2
    return (num_samples, lmdb_path)

def replace_lmdb(args):
    src, trg = args[:]
    if src.size > 2:
        ind = np.random.randint(0,src.size)
        src = src[ind]
    else:
        src = src[0]
    rm_args = "rm -r %s" % trg
    rm_args = shlex.split(rm_args)
    cp_args = "cp -r %s %s" %(src, trg)
    cp_args = shlex.split(cp_args)
    try:
        subprocess.run(rm_args, check=True)
    except subprocess.SubprocessError as e:
        print("subprocess error: %s" % format(e))
    try:
        subprocess.run(cp_args, check=True)
        print("replaced %s" % trg)
    except subprocess.SubprocessError as e:
        print("subprocess error: %s" % format(e))


def main(lmdb_dir, delete=False):
    lmdb_files = os.listdir(lmdb_dir)
    lmdb_paths = [os.path.join(lmdb_dir, path) for path in lmdb_files]
    processes = min(mp.cpu_count(), len(lmdb_files))
    pool = mp.Pool(processes=processes)
    tasks = [(lmdb_path, delete) for lmdb_path in lmdb_paths]
    chunk = max(np.int(np.floor(len(lmdb_files) / processes)), 1)
    jobs = pool.imap(read_lmdb, tasks, chunksize=chunk)
    tally = [j for j in jobs]
    pool.close()
    tally = np.array(tally,  dtype=[('num_samples', 'i4'), ('filepath', np.dtype('U100'))])
    np.save('tally_%s.npy' % lmdb_dir.split('/')[-1], tally )
    mask = np.array([itm.find('_train_') for itm in tally['filepath']])
    mask[mask >= 0] = 1 
    mask[mask < 0] = 0 
    mask = mask.astype(np.bool)
    train_files = tally[mask]
    #print(train_files['filepath'])
    eval_files = tally[np.logical_not(mask)]
    #print(eval_files['filepath'])
    if train_files['num_samples'].size != 0:
        print("stats of train samples (total, min, max, mean): %d, %2.2f, %2.2f, %2.2f" %(train_files['num_samples'].sum(), 
            train_files['num_samples'].min(), train_files['num_samples'].max(), train_files['num_samples'].mean()))
        std = train_files['num_samples'].std()
        mean = train_files['num_samples'].mean()
    #up = min(train_files['num_samples'].max(), mean + 2 *std)
    #down = max(train_files['num_samples'].min(), mean - 2 * std)
    if delete and train_files['num_samples'].size != 0:
        up = 15
        down= 10 
        cutoff = 4 
        #rep = train_files[np.logical_and(train_files['num_samples'] >= down, train_files['num_samples'] <= up)]
        rep = train_files[train_files['num_samples'] > down]
        if rep.size > 1:
            files_to_repl = train_files[train_files['num_samples'] < cutoff]
            if files_to_repl.size > 0:
                print('Replacing Training Data')
                pool = mp.Pool(processes=processes)
                tasks = [(rep['filepath'], itm['filepath']) for itm in files_to_repl]
                chunk = max(np.int(np.floor(files_to_repl.size / processes)), 1)
                jobs = pool.imap(replace_lmdb, tasks, chunksize=chunk)
                _ = [j for j in jobs]
                pool.close()
    if eval_files['num_samples'].size != 0: 
        print("stats of eval samples (total, min, max, mean):%d, %2.2f, %2.2f, %2.2f" %(eval_files['num_samples'].sum(), 
            eval_files['num_samples'].min(), eval_files['num_samples'].max(), eval_files['num_samples'].mean()))
        std = eval_files['num_samples'].std()
        mean = eval_files['num_samples'].mean()
    #up = min(eval_files['num_samples'].max(), mean + 2 *std)
    #down = max(eval_files['num_samples'].min(), mean - 2 * std)
    if delete and eval_files['num_samples'].size != 0:
        up = 20
        down= 2 
        cutoff = 2 
        rep = eval_files[eval_files['num_samples'] > down]
        if rep.size > 1:
            files_to_repl = eval_files[eval_files['num_samples'] < cutoff]
            if files_to_repl.size > 0:
                print('Replacing Eval Data')
                pool = mp.Pool(processes=processes)
                tasks = [(rep['filepath'], itm['filepath']) for itm in files_to_repl]
                chunk = max(np.int(np.floor(files_to_repl.size / processes)), 1)
                jobs = pool.imap(replace_lmdb, tasks, chunksize=chunk)
                _ = [j for j in jobs]
                pool.close()
    

if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[-2], delete=bool(int(sys.argv[-1])))
    else:
        main(sys.argv[-1])
        print('DONE')
