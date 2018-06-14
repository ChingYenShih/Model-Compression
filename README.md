# DLCV_FinalProject
DLCV_FinalProject

## Usage
### Preproccessing (save data as npz file)
1. download dlcv_final_2_dataset.tar.gz to DLCV_FinalProject
2. Save npz file at preproc_data
```
./preproc_script.sh
```
### Training basic model (without compression)
```
python3 basic_train.p -b <batch_size> -d <GPU device id> -m <saved/model/path>
    -m <file>
        default: 'saved_model/basic.model'
    -b <batch size>
        default: 32
    -d <GPU device>
        default: 0'
```
### Testing trained model
```
python3 test.py -i <test/img/dir/> -m <trained/model/path> -o <output/csv/path>
    -i <file>
        Read testing image from <file>
        default: 'dataset/test'
    -m <file>
        Read trained model
    -o <file>
        Output csv result path
        default: 'result/result.csv'
```
