## Validation Datasets

We save all datasets into a unified format (uniformat) for easier testing. You can either download 
the datasets we pre-genenerated at [this link](https://drive.google.com/file/d/1hCNnjKy0R8yU1_lqqTIUUBYJwrz2Bz71/view?usp=sharing), or you can download the original datasets and process them with the following steps.

### Step 0:
Create a folder named `datasets` and put all datasets under it.


### Step 1: download all raw datasets

Details provided later on this page.

### Step 2: extract the uniformat datasets

Under the `src` folder, run `sh robust_dc_protocol/save_uniformat_datasets_all.sh`. This will save many npy files containing the rgb, sprse depth, and gt under `datasets/uniformat_release`. 

### Details on how to download the raw data

#### KITTI

Download the following files and unzip under the `kitti_depth` folder:

[data_depth_annotated](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip), 
[data_depth_velodyne](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_velodyne.zip), 
[data_depth_selection](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip)

Finally, download kitti raw images by:

```
cd datasets/kitti_depth
wget https://github.com/youmi-zym/CompletionFormer/files/12575038/kitti_archives_to_download.txt
wget -i kitti_archives_to_download.txt -P kitti_raw/
cd kitti_raw
unzip "*.zip"
```

The overall data directory is structured as follows:

```
kitti_depth
  ├──data_depth_annotated
  |     ├── train
  |     └── val
  ├── data_depth_velodyne
  |     ├── train
  |     └── val
  ├── data_depth_selection
  |     ├── test_depth_completion_anonymous
  |     |── test_depth_prediction_anonymous
  |     └── val_selection_cropped
  └── kitti_raw
        ├── 2011_09_26
        ├── 2011_09_28
        ├── 2011_09_29
        ├── 2011_09_30
        └── 2011_10_03
```

#### iBims-1
Download following its official link: https://www.asg.ed.tum.de/lmf/ibims1/

Unzip the file to get the following structure:
```
iBims
  ├──gt_depth
  │     ├── corridor_01.mat
  │     └── ...
  └──imagelist.txt
```

#### DIODE
We use the pre-processed images from [marigold](https://github.com/prs-eth/Marigold) following [this link](https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/):

Unzip the tar file under `datasets/marigold` and you will get

```
marigold
  └──diode
      ├── indoors
      │      ├──scene_00019
      │      └──...
      └── outdoors
             ├──scene_00022
             └──...
```

#### ETH3D
First download follow [this link by marigold](https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/):

Unzip the tar file under `datasets/marigold` and you will get

```
marigold
  └──eth3d
      ├──depth
      │   ├── courtyard_dslr_depth
      │   └──...
      └──rgb
          ├── courtyard
          └──...
```
To extract the sparse depth from COLMAP, download the COLMAP scenes from the [ETH3d website](https://www.eth3d.net/datasets#high-res-multi-view-training-data). Specifically, follow [this link](https://www.eth3d.net/data/multi_view_training_dslr_jpg.7z) to get the COLMAP models for the distorted rgbs.

Put the downloaded files under `datasets/eth3d_raw` and get:
```
eth3d_raw
  ├──courtyard
  │   └── dslr_calibration_jpg # the colmap model
  └──...
```

Finally, run `sh write_eth3d_colmap_sparse_depth_all.sh` under this `src/robust_dc_protocol` folder, which will write the sparse depth maps under the marigold folder:

```
marigold
  └──eth3d
      ├──depth
      │   ├── courtyard_dslr_depth
      │   └──...
      ├──rgb
      │   ├── courtyard
      │   └──...
      └──sparse_depth
          ├── courtyard
          └──...
```

#### ARKitScenes
Follow the official instruction [here](https://github.com/apple/ARKitScenes/blob/main/depth_upsampling/README.md). 
You only need the `Validation` split of the `depth upsampling` subset.

Expect something like this:

```
ARKitScenes
  └──depth
      └──upsampling
          ├──Validation
          │    ├──41069021
          │    └──...
          └──metadata.csv
```



#### VOID 
First download the zip files (you can use [gdown](https://github.com/wkentaro/gdown)) under `datasets`:
```
cd datasets
https://drive.google.com/open?id=1rzTFD35OCxMIguxLDcBxuIdhh5T2G7h4
```
Under the `datasets` folder, unzip the downloaded file and you will get:
```
void_release
    ├── void_150
    │    ├── data
    │    │     ├── birthplace_of_internet
    │    │     └── ...
    │    ├── test_absolute_pose.txt      
    │    └── ...
    ├── void_500
    │    └── ...
    └── void_1500
         └── ...
```

#### NYUv2

We used preprocessed NYUv2 HDF5 dataset provided by [Fangchang Ma](https://github.com/fangchangma/sparse-to-dense).

```shell
cd datasets
wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
tar -xvf nyudepthv2.tar.gz
```

After that, you will get a data structure as follows:

```
nyudepthv2_h5
  ├── train
  │    ├── basement_0001a
  │   ...   ├── 00001.h5
  │         └── ...
  └── val
       └── official
            ├── 00001.h5
            └── ...
```