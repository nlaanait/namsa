import os, sys
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import multiprocessing as mp
import numpy as np

def read_tfrecord(args):
    tf_filename, _ = args[:]
    record_iterator = tf.python_io.tf_record_iterator(path=tf_filename)
    image_specs = {'key':'cbed', 'shape':[1024,512,512], 'dtype':'float16'}
    label_specs = {'key':'2d_potential', 'shape':[1,512,512], 'dtype':'float16'}
    image_key, label_key = image_specs['key'], label_specs['key'] 
    label_dtype = tf.as_dtype(label_specs['dtype'])
    image_shape = image_specs['shape'] 
    label_shape = label_specs['shape'] 
    image_dtype = tf.as_dtype(image_specs['dtype'])
    image_size =  np.prod(np.array(image_shape))
    label_size = np.prod(np.array(label_shape))
    for (i,string_record) in enumerate(record_iterator):
        example = tf.train.Example()
        example.ParseFromString(string_record)
        label = np.fromstring(example.features.feature[label_key].bytes_list.value[0],dtype=np.float16)
        image = np.fromstring(example.features.feature[image_key].bytes_list.value[0], dtype=np.float16)
        #    if i > 2:
        #        break
        checks = np.all(np.isnan(label)) or np.all(np.isnan(image))
        if checks:
            print('tfrecord %s contains records with nan' % tf_filename)
            return
        #if label.size != label_size or image.size != image_size:
        #    print_rank('image size and label size are not as expected.')
        #    print_rank('found: %s, %s' %(format(image.size) ,format(label.size)))
        #    print_rank('expected: %s, %s' %(format(image_size) ,format(label_size)))
        #    sys.exit()
        #else:
        #    print('read image and label with sizes: %s, %s' %(format(image.size) ,format(label.size)))

def main(tfrecord_dir, delete=False):
    tfrecord_files = os.listdir(tfrecord_dir)
    tfrecord_paths = [os.path.join(tfrecord_dir, path) for path in tfrecord_files]
    processes = min(mp.cpu_count(), len(tfrecord_files))
    pool = mp.Pool(processes=processes)
    tasks = [(tfrecord_path, delete) for tfrecord_path in tfrecord_paths]
    chunk = max(np.int(np.floor(len(tfrecord_files) / processes)), 1)
    jobs = pool.imap(read_tfrecord, tasks, chunksize=chunk)
    _ = [j for j in jobs]
    pool.close()

if __name__ == "__main__":
    #main(sys.argv[-2], delete=bool(sys.argv[-1]))
    main(sys.argv[-1])
