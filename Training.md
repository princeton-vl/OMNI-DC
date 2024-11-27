## Training 

### Dataset Preparation
You need to download the 5 datasets we used for training, and the NYU dataset for validation. Here is how:

#### Step 0: 
Create a folder named `datasets` and put all datasets under it.

#### Hypersim
Follow the [official link](https://github.com/apple/ml-hypersim). To save some disk space, note you only need two modalities: the tonemapped jpg files
and the depth hdf5 file. Also clone their [Github repo](https://github.com/apple/ml-hypersim), which contains useful metadata. The final structure looks like this:

```
hypersim
  ├──dataset
  │   ├──ai_001_001
  │   │   └──images
  │  ...        ├──scene_cam_00_final_preview
  │             │   └── frame.0000.tonemap.jpg
  │             └──scene_cam_00_geometry_hdf5
  │                 └── frame.0000.depth_meters.hdf5
  └──ml-hypersim            
```

#### BlendedMVS
Download all BlendedMVS, BlendedMVS+, and BlendedMVS++ low-res subsets from [the official repo](https://github.com/YoYo000/BlendedMVS).
Unzip all under the same folder:

```
blendedmvs
  ├──000000000000000000000000
  ├──000000000000000000000001 
 ...             
  └──5c34529873a8df509ae57b58
```

#### IRS

Either go to the [official repo](https://github.com/HKBU-HPML/IRS) or use our [`download_irs.sh`](download_irs.sh). Note we drop the "office" subset because we find the depth annotation are buggy for some scenes. Put the downloaded files under the `datasets/irs` folder
and unzip them. Also download the file lists from [here](https://github.com/HKBU-HPML/IRS/tree/master/lists):

```
irs
  ├──filelist
  │   ├──home_all.txt
  │   ├──restaurant_all.txt
  │   └──store_all.txt
  ├──Home
  ├──Restaurant        
  └──Store
```

#### TartanAir
Follow the instructions [here](https://theairlab.org/tartanair-dataset/). 

```
tartanair
  ├──abandonedfactory
  │   ├──Easy
  │   └──Hard
 ...
  └──westerndesert
      ├──Easy
      └──Hard
```

#### Virtual KITTI
Go [here](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) and download the `vkitti_2.0.3_rgb.tar` and 
`vkitti_2.0.3_depth.tar` to the `datasets/vkitti` folder.
```
vkitti
  ├──depth
  ├──rgb
  ├──vkitti_train.txt
  └──vkitti_val.txt
```

#### NYUv2
We use the version prepared by the Marigold authors [here](https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/): 
```
marigold
  └──nyuv2
      ├──test
      ├──train
      └──...           
```
