import lmdb
import numpy as np
import sys, os, subprocess, shlex

def read_lmdb(save_dir, lmdb_path, index=None):
    env = lmdb.open(lmdb_path, create=False, readahead=False, readonly=True, writemap=False, lock=False)
    num_samples = (env.stat()['entries'] - 4)//2 ## TODO: remove hard-coded # of headers by storing #samples key, val
    first_record = 0
    if index == 'random':
        index = [np.random.randint(first_record, num_samples)]
    else:
        index = np.arange(first_record, num_samples)
    data_specs={'label_shape': [512,512], 'image_shape': [1024, 512, 512],
            'label_dtype':'float16', 'image_dtype': 'float16', 'label_key':'potential_', 'image_key': 'cbed_'}
    for idx in index:
        image_key = bytes(data_specs['image_key']+str(idx), "ascii")
        label_key = bytes(data_specs['label_key']+str(idx), "ascii")
        with env.begin(write=False, buffers=True) as txn:
            image_bytes = txn.get(image_key)
            label_bytes = txn.get(label_key)
            label = np.frombuffer(label_bytes, dtype=data_specs['label_dtype'])
            label = label.reshape(data_specs['label_shape'])
            image = np.frombuffer(image_bytes, dtype=data_specs['image_dtype'])
            image = image.reshape(data_specs['image_shape'])
        np.save(os.path.join(save_dir,'cbed_%d.npy'%idx), image)
        np.save(os.path.join(save_dir,'pot_%d.npy'%idx), label)
    return
if __name__ == "__main__":
    lmdb_path = sys.argv[-2]
    save_dir = sys.argv[-1]
    read_lmdb(save_dir, lmdb_path)
    tar_args = "tar -czvf %s %s" %(os.path.join(os.getcwd(),"saved_arr.tar"), save_dir)
    tar_args = shlex.split(tar_args)
    try:
        subprocess.run(tar_args, check=True)
    except subprocess.SubprocessError as e:
        print("%s" % format(e))
