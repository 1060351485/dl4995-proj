#!/bin/sh
python train-shuffle.py > shuffle-train.log
python test-shuffle.py > shuffle-test.log
python train-all-in-one.py > all-in-one-train.log
python test-all.py > all-in-one-test.log
python test_old_model.py > old-model.log 

python train-shuffle-more-epoch.py > shuffle-more-train.log

python test-fin-all.py > test-fin-all.log
python test-fin-shuffle.py > test-fin-shuffle.log
python test-fin-base.py > test-fin-base.log

python train-all-in-one-val.py > train-all-in-one-val.log
python test-fin-all-val.py > test-fin-all-val.log
