#!/bin/bash
#PBS -N yolo_finetune
#PBS -l select=1:ncpus=3:ngpus=1:gpu_mem=64gb:mem=64gb:scratch_local=10gb
#PBS -l walltime=4:00:00
# TODO add the label redo, and add data.yaml and toher things
RUN_NAME='new_run'

DATADIR=/auto/brno2/home/xbahou00/POVa # substitute username and path to your real username and path
    
 # append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of the node it is run on, and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails, and you need to remove the scratch directory manually 
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt
    
module add mambaforge

 # test if the scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

 # copy input file "h2o.com" to scratch directory
# if the copy operation fails, issue an error message and exit
cp $DATADIR/train.py  $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }
cp $DATADIR/process_labels.py  $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }
cp $DATADIR/data.yaml  $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }


cp $DATADIR/Citypersons.v1i.yolov8.zip  $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }

cd $SCRATCHDIR

unzip Citypersons.v1i.yolov8.zip

mamba activate /storage/brno2/home/xbahou00/env_new

python process_labels.py > $RUN_NAME.out || { echo >&2 "Preparing the labels (with a code $?) !!"; exit 3; }


python train.py > $RUN_NAME.out || { echo >&2 "Training ended eaely (with a code $?) !!"; exit 3; }

# Rename the folder
mv runs $RUN_NAME

cp "$RUN_NAME.out" "$DATADIR" || { echo >&2 "Output log copy failed!"; exit 5; }


cp -r $RUN_NAME $DATADIR || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }

clean_scratch
