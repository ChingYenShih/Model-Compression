# Model-Compression
This repository is the implementation of [paper](https://github.com/ChingYenShih/Model-Compression/blob/master/paper/FLAME%20A%20NEW%20CNN%20MODULE%20FOR%20LARGE%20MODEL%20COMPRESSION.pdf).
We developed CNN module by combining merits of Fire module from SqueezeNet and depthwise separable convolutions from MobileNets. Our CNN module can compress models a lot. In our experiment, we compressed VGG11 model by 91.2% of parameters with less than 1% accuracy drop by applying proposed our CNN module and log min-max quantization.

## Usage
### Preproccessing (save data as npz file)
1. download dlcv_final_2_dataset.tar.gz to Model-Compression
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
