import os, sys
import lmdb
import multiprocessing as mp
import numpy as np

def read_lmdb(args):
    lmdb_path, delete = args[:]
    if delete:
        env = lmdb.open(lmdb_path, map_size=int(100e9), readahead=False, readonly=False, 
                        writemap=True, lock=True, map_async=True)
    else:
        env = lmdb.open(lmdb_path, readahead=False, readonly=True, writemap=False, lock=False)
    with env.begin(write=False) as txn:
        input_shape = np.frombuffer(txn.get(b"input_shape"), dtype='int64')
        output_shape = np.frombuffer(txn.get(b"output_shape"), dtype='int64')
        input_dtype = np.dtype(txn.get(b"input_dtype").decode("ascii"))
        output_dtype = np.dtype(txn.get(b"output_dtype").decode("ascii"))
        output_name = txn.get(b"output_name").decode("ascii")
        input_name = txn.get(b"input_name").decode("ascii")
        num_headers = int.from_bytes(txn.get(b"header_entries"),"little")
#     num_samples = (env.stat()['entries'] - 6)//2 ## TODO: remove hard-coded # of headers by storing #samples key, val
    num_samples = int((env.stat()['entries'] - 6)/2)
    first_record = 0
    records = np.arange(first_record, num_samples)
    data_specs={'label_shape': list(output_shape), 'image_shape': list(input_shape),
            'label_dtype':output_dtype, 'image_dtype': input_dtype, 'label_key':output_name, 'image_key': input_name}
    print('file=%s, samples=%d' %(lmdb_path.split('/')[-1], num_samples))
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
                    print('file=%s, sample=%d, replaced' %(lmdb_path.split('/')[-1], idx))
                    env.sync()

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
