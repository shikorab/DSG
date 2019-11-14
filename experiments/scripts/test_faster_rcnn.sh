#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3
NET_FINAL=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  visual_genome)
    TEST_IMDB="visual_genome_test"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  visualgenome)
    DIR="VisualGenome"
    TEST_IMDB="visualgenome_test"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  clevr)
    DIR="clevr"
    TEST_IMDB="clevr_test"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  vrd)
    DIR="VRD"
    TEST_IMDB="vrd_test"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac
LOG="experiments/logs/test_${NET}_${TEST_IMDB}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
  NET_FINAL=output/${NET}/${DIR}/default/${NET_FINAL}.ckpt
set -x

CUDA_VISIBLE_DEVICES=${GPU_ID} time python -m pdb ./tools/test_net.py \
  --imdb ${TEST_IMDB} \
  --model ${NET_FINAL} \
  --cfg experiments/cfgs/${NET}.yml \
  --net ${NET} \
  --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
        ${EXTRA_ARGS}


