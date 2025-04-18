# Welcome to the DSH-Net

## Installation

We recommend installation of the required packages using the conda package manager, available through the Anaconda Python distribution. Anaconda is available free of charge for non-commercial use through [Anaconda Inc](https://www.anaconda.com/products/individual). After installing Anaconda and cloning this repository, For use as integrative framework：

```
git clone https://github.com/springbreeze7111/DSH-Net
cd DSH-Net
pip install -e.
cd dynamic-network-architectures
pip install -e.
```

## Dataset download

Datasets can be acquired via following links:

**Dataset I** [MSD](http://medicaldecathlon.com/)

**Dataset II** [BRATS2021](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)

## Setting up the datasets

```
./DSH-Net/
./DATASET/
  ├── nnUNet_raw/
       ├── nnUNet_raw_data/
           ├── Task01_BrainTumour/
  ├── nnUNet_trained_models/
  ├── nnUNet_preprocessed/
```

Setting your environment variables

```
export nnUNet_raw="/root/autodl-tmp/nnUNet_raw"
export nnUNet_preprocessed="/root/autodl-tmp/nnUNet_preprocessed"
export nnUNet_results="/root/autodl-tmp/nnUNet_results"
```



## How does DSH-Net?
First we need to convert the MSD dataset format conversion to the format supported by the nnUNetv2 framework（You can refer to the usage of nnUNetv2）

```
nnUNetv2_convert_MSD_dataset -i \nnUNet\nnUNet_raw\Task01_BrainTumour
```

where -i is followed by the path to the dataset under your own nnUNet_raw_data/.

 nnU-Net will extract a dataset fingerprint (a set of dataset-specific properties such as image sizes, voxel spacings, intensity information etc). This information is used to design three U-Net configurations. Each of these pipelines operates on its own preprocessed version of the dataset.

```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

**We can modify our desired network configuration (patch_size, batch_size and some network parameters) in nnUNetPlans.json in the generated nnUNet_preprocessed folder**

Training models is done with the `nnUNetv2_train` command. The general structure of the command is:

```
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]
```

UNET_CONFIGURATION is a string that identifies the requested U-Net configuration (defaults: 2d, 3d_fullres, 3d_lowres, 3d_cascade_lowres). DATASET_NAME_OR_ID specifies what dataset should be trained on and FOLD specifies which fold of the 5-fold-cross-validation is trained.

## Acknowledgements

Code copied a lot from  [nnUnetv2](https://github.com/MIC-DKFZ/nnUNet)

## Cite

Literature citation will be added to the paper upon acceptance
