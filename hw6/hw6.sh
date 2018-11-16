#!/bin/bash
wget -O model_best.h5 'https://www.dropbox.com/s/2e6pavktg5vvzrr/model_best.h5?dl=1'
python3 MF_test.py $1 $2 $3 $4
