#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3
EXPERIMENT=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}

case ${DATASET} in
  visualgenome)
    TRAIN_IMDB="visualgenome_train"
    TEST_IMDB="visualgenome_test"
    STEPSIZE="[4,8,12,16]"
    ITERS=20
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  clevr)
    TRAIN_IMDB="clevr_train"
    TEST_IMDB="clevr_test"
    STEPSIZE="[2,4,6,8]"
    ITERS=20
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  vrd)
    TRAIN_IMDB="vrd_train"
    TEST_IMDB="vrd_test"
    STEPSIZE="[10,20,30,40]"
    ITERS=50
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_${ITERS}.ckpt

set -x

CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
  --weight data/imagenet_weights/${NET}.ckpt \
  --imdb ${TRAIN_IMDB} \
  --imdbval ${TEST_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/${NET}.yml \
  --net ${NET} \
  --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} TRAIN.SNAPSHOT_PREFIX ${EXPERIMENT} \
  TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
