# Supervised Contrastive Learning for Snapshot Spectral Imaging Face Anti-Spoofing [CVPR2024 Workshop]
This repository is the official implementation of Supervised Contrastive Learning for Snapshot Spectral Imaging Face Anti-Spoofing(https://arxiv.org/abs/2405.18853). We rank first at the Chalearn Snapshot Spectral Imaging Face Anti-spoofing Challenge on CVPR 2024; the paper is accepted by CVPR 2024 workshop.

## Reproduction Codes for Snapshot Spectral Imaging Face Anti-spoofing Challenge

Here we provide the training and testing codes for the reproduction of our result. To facilitate the reproduction of the result, we also provide the training log file `log.csv` and the result file `result.txt`.

### 1. Environment Setting
In order to reproduce the same result, the environment setting is important. Please make sure the basic environment setting is the environment `py310 + torch2.2.1 + cuda12.1` with `1 NVIDIA Tesla A100 80G GPU`, and our algorithm can be deterministic under this environment.
You can run the following script to configure the necessary python package:
```
pip install albumentations==1.3.0
pip install kornia==0.7.2
pip install pytorch_metric_learning==2.4.1
pip install torch_ema==0.3
pip install numpy==1.26.4
pip install pandas==2.2.1
pip install timm==0.9.16
```

### 2. Training
Before runing the training script `python train.py`, you need to make some revisions for the preparation.

- training image path: please download and extrat the HySpeFAS training data `HySpeFAS_trainval.zip`, and revise training image path `/path/to/HySpeFAS_trainval/images/` at the line 118 and 121 of `train.py`.

- model saving path: please revise model saving path at the line 283 of `train.py`.

- ImageNet pretrained model path: please download the ImageNet pretrained model via the url `https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth`, and revise the ImageNet pretrained model path at the line 327 of `efficientnet/utils.py`

After that, you can run the the training script `python train.py` to train the target model. The model and log files will be saved in the model saving path above.

### 3. Testing
Before runing the testing script `python test.py`, you need to make some revisions for the preparation.

- testing image path: please download and extrat the HySpeFAS testing data `HySpeFAS_test.zip`, and revise testing image path `/path/to/HySpeFAS_test/images/` at the line 49 and 52 of `test.py`.

- inference model path: Based on the path of the model file at the last training epoch, please revise inference model path `/path/to/mat_b3_256_sfas_model_0/model_epoch_30.pth` at the line 91 of `test.py`.

- inference threshold (optional) : you also need to revise probability threshold for testing data at the line 119 of `test.py`, based on the validation threshold in the training log, but we have done it for you.

After that, you can run the the testing script `python test.py` to get the result file `result.txt`.
