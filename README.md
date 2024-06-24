# LimitNet
Welcome to the LimitNet GitHub repository! This is the repository for our paper: **[LimitNet: Progressive, Content-Aware Image Offloading for Extremely Weak Devices & Networks (MobiSys2024)](https://doi.org/10.1145/3643832.3661856)**. 


## Table of Contents

- [LimitNet](#limitnet)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Training on CIFAR100](#training-on-cifar100)
    - [Training on ImageNet](#training-on-imagenet)
    - [Evaluation on CIFAR100](#evaluation-on-cifar100)
  - [Project Structure](#project-structure)
  - [License](#license)

## Installation

To get started, clone the repository and install the required dependencies:

```sh
git clone https://github.com/yourusername/LimitNet.git
cd LimitNet
pip install -r requirements.txt
```


## Installation
LimitNet uses BASNet as the teacher for traning the salicny detector branch. We used this model to extract the salicny maps on a subset of the ImageNet dataset which you can download it from here:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12206178.svg)](https://doi.org/10.5281/zenodo.12206178)


### Training

To train the LimitNet model on the CIFAR100 dataset, run the following command:

```sh
python train.py --model cifar --batch_size 32 --imagenet_root <IMAGENET_ROOT> --checkpoint_dir checkpoint --wandb_name LimitNet --wandb_project LimitNet --cifar_classifier_model_path ./EfficentNet-CIFAR100
```
To train the LimitNet model on the ImageNet dataset, run the following command:

```sh
python train.py --model imagenet --batch_size 32 --imagenet_root <IMAGENET_ROOT> --checkpoint_dir checkpoint --wandb_name LimitNet --wandb_project LimitNet --cifar_classifier_model_path ./EfficentNet-ImageNet
```

### Evaluation on CIFAR100

To evaluate the LimitNet model on the CIFAR100 dataset, run the following command:

```sh
python eval.py --model cifar --model_path './LimitNet-CIFAR100' --batch_size 32 --test_batch_size 32 --imagenet_root <IMAGENET_ROOT> 
```

``` sh
python eval.py --model cifar --model_path './LimitNet-ImageNet'  --batch_size 32 --test_batch_size 32 --imagenet_root <IMAGENET_ROOT> 
```

## Results
### CIFAR-100 Results
![CIFAR-100 Results](image_size_vs_accuracy_cifar-100.png)

### ImageNet Results
![ImageNet Results](image_size_vs_accuracy_ImageNet1k.png)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
