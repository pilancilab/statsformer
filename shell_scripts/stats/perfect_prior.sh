#!/bin/bash

BASE_DIR=data/experiments/perfect_prior/featurewise
EXTRA_ARGS="--n-trials 100 --nonlinear featurewise"

# generic underdetermined
N=100
P=1000
PEFF=20
IMBALANCE=0.2
python scripts/stats/perfect_prior.py \
    --save-dir $BASE_DIR/N-${N}__P-${P}__PEFF-${PEFF}__IMBALANCE-${IMBALANCE}/ \
    --n $N --p $P --p-eff $PEFF \
    $EXTRA_ARGS


# "small-scale"
N=500
P=30
PEFF=15
IMBALANCE=0.5
python scripts/stats/perfect_prior.py \
    --save-dir $BASE_DIR/N-${N}__P-${P}__PEFF-${PEFF}__IMBALANCE-${IMBALANCE}/ \
    --n $N --p $P --p-eff $PEFF \
    --imbalance $IMBALANCE \
    $EXTRA_ARGS


# medium
N=300
P=500
PEFF=50
IMBALANCE=0.3
python scripts/stats/perfect_prior.py \
    --save-dir $BASE_DIR/N-${N}__P-${P}__PEFF-${PEFF}__IMBALANCE-${IMBALANCE}/ \
    --n $N --p $P --p-eff $PEFF \
    --imbalance $IMBALANCE \
    $EXTRA_ARGS


# small n, dense signal
N=80
P=50
PEFF=40
IMBALANCE=0.5
python scripts/stats/perfect_prior.py \
    --save-dir $BASE_DIR/N-${N}__P-${P}__PEFF-${PEFF}__IMBALANCE-${IMBALANCE}/ \
    --n $N --p $P --p-eff $PEFF \
    --imbalance $IMBALANCE \
    $EXTRA_ARGS

# underdetermined
N=200
P=500
PEFF=5
IMBALANCE=0.5
python scripts/stats/perfect_prior.py \
    --save-dir $BASE_DIR/N-${N}__P-${P}__PEFF-${PEFF}__IMBALANCE-${IMBALANCE}/ \
    --n $N --p $P --p-eff $PEFF \
    --imbalance $IMBALANCE \
    $EXTRA_ARGS


# extreme underdetermined
N=50
P=500
PEFF=5
IMBALANCE=0.5
python scripts/stats/perfect_prior.py \
    --save-dir $BASE_DIR/N-${N}__P-${P}__PEFF-${PEFF}__IMBALANCE-${IMBALANCE}/ \
    --n $N --p $P --p-eff $PEFF \
    --imbalance $IMBALANCE \
    $EXTRA_ARGS