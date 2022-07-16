category=$1
exp_name=$2

### Direct Mapping + Refine.
python grasp_transfer/refine_main.py --category ${category} --dm

### Simulation
# parameters of test_all_grasps.sh: 1.category,2.grasp_data_dir,3.suffix,4.result_file_name
# Direct Mapping Simulation
sh scripts/test_all_grasps.sh ${category} grasp_data/${exp_name} dm ${exp_name}_dm
# Grasp Transfer Simulation
sh scripts/test_all_grasps.sh ${category} grasp_data/${exp_name} tf ${exp_name}_tf
# Direct Mapping + Refine Simulation
sh scripts/test_all_grasps.sh ${category} grasp_data/${exp_name}_refine dm_refine ${exp_name}_dm_refine
# Grasp Transfer + Refine Simulation
sh scripts/test_all_grasps.sh ${category} grasp_data/${exp_name}_refine tf_refine ${exp_name}_tf_refine