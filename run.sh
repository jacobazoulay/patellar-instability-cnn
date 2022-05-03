#!/usr/bin/env bash
#source venv/bin/activate

PROGRAM='main.py'
NET='AlexNet'
DATASET='DogCat'
BATCH_SIZE=4
OPTIM='sgd'
MOMENTUM=0.9
LR=1e-3
SEED=1
TRAIN=1
EVAL=$((1-$TRAIN))
RESUME=0
SAVE_NTH_EPOCH=1000
TEST_NTH_EPOCH=$SAVE_NTH_EPOCH
NWORKERS=1
EPOCHS=100


python -u $PROGRAM --net $NET --seed $SEED --resume $RESUME --eval $EVAL --batch_size $BATCH_SIZE --dataset $DATASET --epochs $EPOCHS --nworkers $NWORKERS --save_nth_epoch $SAVE_NTH_EPOCH --test_nth_epoch $TEST_NTH_EPOCH --train $TRAIN --resume $RESUME \
--optim $OPTIM --lr $LR --momentum $MOMENTUM
