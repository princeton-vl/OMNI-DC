## void
#for sample in 1500 500 150
#do
#  python robust_dc_protocol/save_uniformat_datasets.py \
#    --val_data_name VOID \
#    --dir_data ../datasets/void_release/void_${sample} \
#    --val_depth_pattern $sample \
#    --benchmark_gen_split test \
#    --benchmark_save_name "VOID_sample${sample}"
#done
#
## nyu
#for sample in 500 200 100 50 5
#do
#  python robust_dc_protocol/save_uniformat_datasets.py \
#    --val_data_name NYU \
#    --dir_data ../datasets/nyudepthv2_h5 \
#    --split_json ../data_json/nyu.json \
#    --val_depth_pattern $sample \
#    --benchmark_gen_split test \
#    --benchmark_save_name "NYU_test_$sample"
#done
#
## kitti - real lidar
#data_name=KITTIDC
#for split in val test
#do
#  for lidar_line in 64 32 16 8
#  do
#     python robust_dc_protocol/save_uniformat_datasets.py \
#     --val_data_name $data_name \
#     --benchmark_gen_split $split \
#     --benchmark_save_name "${data_name}_${split}_LiDAR_${lidar_line}" \
#     --dir_data ../datasets/kitti/kitti_depth --split_json ../data_json/kitti_dc.json \
#     --patch_height 240 --patch_width 1216 --lidar_lines $lidar_line \
#     --top_crop 96 --test_crop
#  done
#done

# eth3d - real sfm
for data_name in ETH3D_SfM_Indoor ETH3D_SfM_Outdoor
do
  for split in val test
  do
     python robust_dc_protocol/save_uniformat_datasets.py \
     --val_data_name $data_name \
     --benchmark_gen_split $split \
     --benchmark_save_name "${data_name}_${split}"
  done
done

# virtual part
all_virtual_datasets=(ARKitScenes iBims ETH3D_Indoor ETH3D_Outdoor DIODE_Indoor DIODE_Outdoor)

for data_name in "${all_virtual_datasets[@]}"
do
  # part 1: random density
  for sample_pattern in 2150 300 100
  do
     python robust_dc_protocol/save_uniformat_datasets.py \
     --val_data_name $data_name \
     --val_depth_pattern $sample_pattern \
     --benchmark_gen_split test \
     --val_depth_noise 0.0 \
     --benchmark_save_name "${data_name}_test_${sample_pattern}"
  done

  # part 2: random noise
  for noise in 0.01 0.05 0.1
  do
     python robust_dc_protocol/save_uniformat_datasets.py \
     --val_data_name $data_name \
     --val_depth_pattern 2150 \
     --benchmark_gen_split test \
     --val_depth_noise $noise \
     --benchmark_save_name "${data_name}_test_noise${noise}"
  done

  # part 3: sift/orb/lidar
  for pattern in sift orb LiDAR_64 LiDAR_32 LiDAR_16 LiDAR_8
  do
     python robust_dc_protocol/save_uniformat_datasets.py \
     --val_data_name $data_name \
     --val_depth_pattern $pattern \
     --benchmark_gen_split test \
     --val_depth_noise 0.0 \
     --benchmark_save_name "${data_name}_test_${pattern}"
  done
done


