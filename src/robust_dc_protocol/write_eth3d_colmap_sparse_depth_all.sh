for scene in courtyard delivery_area electro facade kicker meadow office pipes playground relief relief_2 terrace terrains
do
  python read_write_colmap_model.py \
  --input_model "../../datasets/eth3d_raw/${scene}/dslr_calibration_jpg" \
  --save_sparse_depth_dir "../../datasets/marigold/eth3d/sparse_depth/${scene}"
done