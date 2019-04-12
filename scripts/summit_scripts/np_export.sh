
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

LOG="$(pwd)/export.log"
SAVEDIR="$(pwd)/saved_arr"
FILE="$(pwd)/data/lmdb_fix/batch_train_1181.db"
EXEC="python -W ignore -u lmdb_export.py $FILE $SAVEDIR"
   
### run ###
jsrun -n${NODES} -a1 -c42 -g1 -r1 --bind=proportional-packed:7 --launch_distribution=packed ${EXEC} > $LOG 
