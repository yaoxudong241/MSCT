# MSCT
## _Single-Image Super Resolution for Remote Sensing via Multi-Scale CNN-Transformer Feature Fusion_

## Environment
- python 3.8
- pytorch=2.1.0
- GDAL=3.1.4

## Model
![Structure of MSCT](https://github.com/user-attachments/assets/579b3bad-c284-4b85-b457-03ac66185622)
Architecture of our MSCT. Among them, ARDB, MSTB and LGEB stand for the adaptive residual dense block, the multi-scale Transformer block and the local-global information enhancement block, respectively.

## Train
- dataset:Sen2venus
- prepare

```sh
python train.py --root /path/to/train/ --rootval /path/to/val/
```
## Test

```sh
python test.py --test_hr_folder /path/to/hrimg/ --test_lr_folder /path/to/lrimg/ --output_folder /path/to/srimg/ --checkpoint /path/to/model/
```


## Visual comparison
![playground](https://github.com/user-attachments/assets/68204de0-f7db-4919-95a9-c10047b5c5ce)
![road](https://github.com/user-attachments/assets/c1183624-16cd-4eda-aeb3-a711048c93b9)
![ship](https://github.com/user-attachments/assets/1d434a3a-3998-4074-8312-29736034bcb7)
Qualitative evaluation results for different SISR methods on AID dataset at a scale of 4. (a), (b) and (c) correspond to the names of images in the dataset. Our results restore sharper and more accurate boundaries, which are closer to the ground truth.

## Acknowledgements
This code is built on [ESRT (Torch)](https://github.com/luissen/ESRT). We thank the authors for sharing their codes of ESRT PyTorch version.












