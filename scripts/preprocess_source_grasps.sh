category=$1

python grasp_transfer/RT_to_g.py --category $category
python isaac_sim/source_grasp_filter.py --category $category --pkl_root grasp_data/$category/ # x 2
python isaac_sim/source_grasp_filter.py --category $category --pkl_root grasp_data/$category/
python grasp_transfer/transfer_to_template.py --category $category
python grasp_transfer/refine_template.py --category $category