#!/bin/bash
tar zxvf dlcv_final_2_dataset.tar.gz
rm dlcv_final_2_dataset.tar.gz
mv dlcv_final_2_dataset dataset
python3 preproc.py -m train
python3 preproc.py -m val
