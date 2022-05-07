#!/usr/bin/env bash
#source venv/bin/activate

PROGRAM='main.py'
NET='CDINet' #CDINet or AlexNet    CDINet is the baseline
DATASET='CDI'
BATCH_SIZE=4
OPTIM='sgd'
MOMENTUM=0.9
LR=1e-3
SEED=1
TRAIN=0
EVAL=$((1-$TRAIN))
RESUME=0
SAVE_NTH_EPOCH=5
TEST_NTH_EPOCH=$SAVE_NTH_EPOCH
TEST_SPLIT='val' #train, val. Train is to overfit
NWORKERS=1
EPOCHS=50


python -u $PROGRAM --net $NET --seed $SEED --resume $RESUME --eval $EVAL --batch_size $BATCH_SIZE --dataset $DATASET --epochs $EPOCHS --nworkers $NWORKERS --save_nth_epoch $SAVE_NTH_EPOCH --test_nth_epoch $TEST_NTH_EPOCH --train $TRAIN --resume $RESUME \
--optim $OPTIM --lr $LR --momentum $MOMENTUM --test_split $TEST_SPLIT
