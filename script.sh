#!/bin/bash
tar zxvf dlcv_final_2_dataset.tar.gz
python3 preproc.py -m train
python3 preproc.py -m val
