
NODES=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
BUILDS=${PROJWORK}/lrn001/nl/builds

### modules ###
module unload darshan-runtime
module load gcc/6.4.0
module load fftw hdf5 cuda

### python ###
PYTHON=${BUILDS}/miniconda3
export PATH=$PYTHON/bin:$PATH 
export PYTHONIOENCODING="utf8"
export LD_LIBRARY_PATH=${PYTHON}/lib:$LD_LIBRARY_PATH
#CONDA_ENV_NAME="torch1p0"
CONDA_ENV_NAME="tf1p12"
source activate $CONDA_ENV_NAME
echo $(which python)

LOG="$(pwd)/output_sim.log"

### namsa ###
CIFDIR="$(pwd)/data/materialsgenomics"
#SAVEDIR="$(pwd)/data/h5_files"
SAVEDIR="$(pwd)/data/lmdb_test"
#SAVEDIR="$(pwd)/data/tfrecords"
#SAVEDIR="/mnt/bb/${USER}"
#SAVEMODE="tfrecord"
#SAVEMODE="h5"
SAVEMODE="lmdb"
export PYCUDA_DISABLE_CACHE=1
EXEC="python -W ignore -u sim_batch_debug.py $CIFDIR $SAVEDIR $SAVEMODE"
   
### run ###
jsrun -n${NODES} -a6 -c42 -g6 -r1 --bind=proportional-packed:7 --launch_distribution=packed ${EXEC} > $LOG 
