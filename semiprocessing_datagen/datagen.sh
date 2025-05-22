#!/bin/bash

# Executable for parallel processing of Morris' *.gz files
# Takes as argument the integer in the file name
# For a new dataset, adjust EOSDIR and make sure Morris has given you explicit access

i=$1

source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
#source /cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc8-opt/setup.sh

outdir=/eos/user/d/dshekar/dataset_3su/parquet_temp/
#outdir=/eos/project/s/smartpix-box/pixelAV_datasets/dataset_9s/parquet_temp/

mkdir unflipped

#point to the correct path
EOSDIR=/eos/user/d/dshekar/dataset_3su/dataset_3su_50x12P5/
#EOSDIR=/eos/project/s/smartpix-box/pixelAV_datasets/dataset_7s/dataset_7s_50x12P5/
#EOSDIR=/eos/user/d/dshekar/dataset_3sr/dataset_3sr_100x25/
#EOSDIR=/eos/user/d/dshekar/dataset_5s/dataset_5s_100x25/
#EOSDIR=/eos/user/d/dshekar/dataset_4s/dataset_4s_50x12P5/
#EOSDIR=/eos/user/d/dshekar/dataset_3s/dataset_3s_50x12P5/
#EOSDIR=/eos/user/d/dshekar/dataset_2s/dataset_2s_50x25/
#EOSDIR=/eos/user/d/dshekar/dataset14_50x12P5

xrdcp root://eosuser.cern.ch/$EOSDIR/pixel_clusters_d${i}.out.gz pixel_clusters_d${i}.out.gz
pwd
#xrdcp /eos/user/d/dshekar/dataset14_50x12P5/pixel_clusters_d${i}.out.gz pixel_clusters_d${i}.out.gz
gunzip pixel_clusters_d${i}.out.gz

python datagen.py $i

xrdcp -f unflipped/labels_d${i}.parquet root://eosuser.cern.ch/$outdir/unflipped/labels_d${i}.parquet
xrdcp -f unflipped/recon2D_d${i}.parquet root://eosuser.cern.ch/$outdir/unflipped/recon2D_d${i}.parquet
xrdcp -f unflipped/recon2D_uncentered_d${i}.parquet root://eosuser.cern.ch/$outdir/unflipped/recon2D_uncentered_d${i}.parquet
xrdcp -f unflipped/recon3D_d${i}.parquet root://eosuser.cern.ch/$outdir/unflipped/recon3D_d${i}.parquet
xrdcp -f unflipped/recon3D_uncentered_d${i}.parquet root://eosuser.cern.ch/$outdir/unflipped/recon3D_uncentered_d${i}.parquet
xrdcp -f unflipped/offset_histogram_d${i}.png root://eosuser.cern.ch/$outdir/unflipped/offset_histogram_d${i}.png

