import os, sys
import lmdb
import multiprocessing as mp
import numpy as np

def read_lmdb(args):
    lmdb_path, delete = args[:]
    if delete:
        env = lmdb.open(lmdb_path, map_size=int(100e9), readahead=False, readonly=False, writemap=False, lock=True)
    else:
        env = lmdb.open(lmdb_path, readahead=False, readonly=True, writemap=False, lock=False)
    num_samples = env.stat()['entries'] - 4 ## TODO: remove hard-coded # of headers by storing #_headers key
    first_record = 0
    records = np.arange(first_record, num_samples//2)
    data_specs={'label_shape': [1,512,512], 'image_shape': [1024, 512, 512], 
          'label_dtype':'float16', 'image_dtype': 'float16', 'label_key':'potential_', 'image_key': 'cbed_'}
    print('file=%s, samples=%d' %(lmdb_path.split('/')[-1], num_samples//2))
    with env.begin(write=delete, buffers=True) as txn:
        for idx in records:
            image_key = bytes(data_specs['image_key']+str(idx), "ascii")
            label_key = bytes(data_specs['label_key']+str(idx), "ascii")
            try:
                image_bytes = txn.get(image_key)
                label_bytes = txn.get(label_key)
                label = np.frombuffer(label_bytes, dtype=data_specs['label_dtype'])
                label = label.reshape(data_specs['label_shape'])
                image = np.frombuffer(image_bytes, dtype=data_specs['image_dtype'])
                image = image.reshape(data_specs['image_shape'])
                print('file=%s, sample=%d, status=success' %(lmdb_path, idx))
            except Exception as e:
                print('file=%s, sample=%d, status=fail, error=%s' %(lmdb_path.split('/')[-1], idx, format(e)))
                if delete:
                    image = np.random.uniform(low=-1.0, high=1.0, size=data_specs['image_shape']).astype(data_specs['image_dtype'])
                    label = np.random.uniform(low=0.0, high=1.0, size=data_specs['label_shape']).astype(data_specs['label_dtype']) 
                    txn.put(image_key, image.flatten().tostring())
                    txn.put(label_key, label.flatten().tostring())

def main(lmdb_dir, delete=False):
    lmdb_files = os.listdir(lmdb_dir)
    lmdb_paths = [os.path.join(lmdb_dir, path) for path in lmdb_files]
    processes = min(mp.cpu_count(), len(lmdb_files))
    pool = mp.Pool(processes=processes)
    tasks = [(lmdb_path, delete) for lmdb_path in lmdb_paths]
    chunk = max(np.int(np.floor(len(lmdb_files) / processes)), 1)
    jobs = pool.imap(read_lmdb, tasks, chunksize=chunk)
    _ = [j for j in jobs]
    pool.close()

if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[-2], delete=bool(int(sys.argv[-1])))
    else:
        main(sys.argv[-1])
