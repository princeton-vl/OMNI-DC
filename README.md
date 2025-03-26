# 

<p align="center">

  <h1 align="center">OMNI-DC: Highly Robust Depth Completion with Multiresolution Depth Integration</h1>
  <p align="center">
    <a href="https://zuoym15.github.io/"><strong>Yiming Zuo</strong></a>
    ·
    <a href="https://www.linkedin.com/in/liu-willow-yang-787b82210/"><strong>Willow Yang</strong></a>
    ·
    <a href="https://mazeyu.github.io/"><strong>Zeyu Ma</strong></a>
    ·
    <a href="https://www.cs.princeton.edu/~jiadeng/"><strong>Jia Deng</strong></a>    
  </p>
  <p align="center">
    <a href="https://pvl.cs.princeton.edu/">Princeton Vision & Learning Lab (PVL)</a>    
  </p>
</p>

<h3 align="center"><a href="https://arxiv.org/abs/2411.19278">Paper</a>  · </a><a href="https://zuoym15.github.io/OMNI-DC-Homepage/">Project</a> </a></h3>

<p align="center">
  <a href="https://arxiv.org/abs/2411.19278">
    <img src="./figures/teaser.png" alt="Logo" width="98%">
  </a>
</p>

## TL;DR
We present a depth completion model that works well on unseen datasets and various depth patterns (zero-shot). It can be used to regularize Gaussian Splatting models to achieve better rendering quality, or work with LiDARs for dense mapping.

## Change Logs
- 3/21/2025 v1.1 relased

## V1.1

We find that concatenating the output of Depth Anyting v2 to the sparse depth map improves the results consistently, especially when the input depth is sparse (error reduced by 50% on NYU with 5 points compared to v1.0). 

Download the [Depth Anything checkpoint](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) to `src/depth_models/depth_anything_v2/checkpoints`.
Usage is the same as v1.0. See `testing_scripts/test_v1.1.sh` for an example. Checkpoint can be downloaded [here](https://drive.google.com/file/d/1ssJYFB3rQD5JEYgG7W6tRJg1hpQKvqPD/view?usp=sharing).
<details>
  <summary>Detailed Accuracy Comparison Between v1.0 and v1.1</summary>
  
  ETH3D: RMSE/REL; KITTIDC: RMSE/MAE

| Method  | ETH3D_SfM_In | ETH3D_SfM_Out | KITTIDC_64 | KITTIDC_16 | KITTIDC_8 |
|----------|----------|----------|----------|----------|----------|
| v1.0 | 0.605/0.090 | **1.069**/0.053 | **1.191/0.270** | 1.682/0.441 | 2.058/0.597 |
| v1.1 | **0.488/0.061** | 1.092/**0.035** | 1.235/0.276 | **1.659/0.430** | **1.897/0.546** |

VOID: RMSE/MAE

| Method  | VOID_1500 | VOID_500 | VOID_150 |
|----------|----------|----------|----------|
| v1.0 |  0.555/0.150 | 0.551/0.164 | 0.650/0.211 |  
| v1.1 |  **0.540/0.143** | **0.512/0.144** | **0.581/0.171** |

NYU: RMSE/REL

| Method  | NYU_500 | NYU_200 | NYU_100 | NYU_50 | NYU_5 |
|----------|----------|----------|----------|----------|----------|
| v1.0 | **0.111/0.014** | 0.147/0.021 | 0.180/0.029 | 0.225/0.041 | 0.536/0.142 |
| v1.1 | 0.112/**0.014**  | **0.143/0.019**  | **0.167/0.024** | **0.193/0.029** | **0.330/0.070** |

Virtual Depth Pattens (Avg on 4 datasets): RMSE/REL

| Method  | 0.7% | 0.1% | 0.03% | 5% Noise |  10% Noise |
|----------|----------|----------|----------|----------|----------|
| v1.0 | 0.135/0.010 | 0.211/0.020 | 0.289/0.034 | 0.141/**0.010** | 0.147/0.011 | 
| v1.1 | **0.126/0.009** | **0.172/0.016** | **0.213/0.025** | **0.130/0.010** | **0.134/0.010** |
| **Method** | **ORB** | **SIFT** | **LiDAR-64** | **LiDAR-16** | **LiDAR-8** |
| v1.0 | 0.247/0.045 | 0.211/0.037 | 0.121/**0.008** | 0.164/0.014 | 0.231/0.023 |
| v1.1 | **0.176/0.028** | **0.161/0.024** | **0.114/0.008** | **0.146/0.012** | **0.180/0.017** |
</details>

## Environment Setup
We recommend creating a python enviroment with anaconda.
```shell
conda create -n OMNIDC python=3.8
conda activate OMNIDC
# For CUDA Version == 11.3
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install mmcv==1.4.4 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html 
pip install mmsegmentation==0.22.1 
pip install timm tqdm thop tensorboardX tensorboard opencv-python ipdb h5py ipython Pillow==9.5.0 plyfile einops
```

#### NVIDIA Apex

We used NVIDIA Apex for multi-GPU training. Apex can be installed as follows:

```shell
git clone https://github.com/NVIDIA/apex
cd apex
git reset --hard 4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a
conda install cudatoolkit-dev=11.3 -c conda-forge
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ 
```

You may face the bug `ImportError: cannot import name 'container_abcs' from 'torch._six'`. In this case, change line 14 of apex/apex/_amp_state.py to `import collections.abc as container_abcs` and re-install apex.

#### Backbone Initialization Files
Download these `.pt` files to `src/pretrained`:
```
https://drive.google.com/drive/folders/1z2sOkIJHtg1zTYiSRhZRzff0AANprx4O?usp=sharing
```



## Reproduce Results in the Paper 

#### Download checkpoints
Download from 
```
# v1.0
https://drive.google.com/file/d/1SBRfdhozd-3j6uorjKOMgYGmrd578NvG/view?usp=sharing

# v1.1
https://drive.google.com/file/d/1ssJYFB3rQD5JEYgG7W6tRJg1hpQKvqPD/view?usp=sharing
```
and put it under the `checkpoints` folder.

#### Prepare the datasets

We save all evaluation datasets in a unified form (uniformat), and you can directly download it from [here](https://drive.google.com/file/d/1hCNnjKy0R8yU1_lqqTIUUBYJwrz2Bz71/view?usp=sharing).
Put all npy files under the `datasets/uniformat_release` folder:

```
uniformat_release
  ├───ARKitScenes_test_100
  │   ├──000000.npy
 ...  ├──000001.npy
      └──...           
```





We also provide instructions on how to process the original datasets to get these npy files, check [this link](src/robust_dc_protocol/README.md):  

#### Testing
```
cd src

# the real and virtual patterns from the 5 datasets reported in tab.1 and tab.2 in the paper
sh testing_scripts/test_robust_DC.sh

# additional results on the void and nyuv2 datasets
sh testing_scripts/test_void_nyu.sh
```
## Test on Your Own Dataset

We recommend writing a dataloader to save your own dataset into the unified format we use (uniformat). You will need to provide an RGB image and a sparse depth map (with 0 indicating invalid pixels). A good starting point is the [ibims dataloader](src/data/ibims.py). 

Then follow the instructions [here](src/robust_dc_protocol/README.md) to convert your dataset into uniformat. Specifically, look into `src/robust_dc_protocol/save_uniformat_dataset.py`.

Once done, you can run evaluation just as on any other datasets we tested on. See examples under `src/testing_scripts`.

If you want to use our model for view synthesis (e.g., Gaussian Splatting), you may find the instructions [here](src/robust_dc_protocol/README.md) helpful. The ETH3D section describes how to 
convert COLMAP models to sparse depth maps.

## Train from Scratch 

#### Resource requirements
We use 10x48GB GPUs (e.g., RTX A6000) and ~6 days. You can adjust the batch size depending on the memory and the numbers of GPU cards you have. 

#### Dataset preparation
See [here](Training.md) for instructions.

#### Training
```
cd src
sh training_scripts/train_full.sh 
```

## Citation 
If you find our work helpful please consider citing our paper:
```
@article{zuo2024omni,
   title={OMNI-DC: Highly Robust Depth Completion with Multiresolution Depth Integration},
   author={Zuo, Yiming and Yang, Willow and Ma, Zeyu and Deng, Jia},
   journal={arXiv preprint arXiv:2411.19278},
   year={2024}
}
```
