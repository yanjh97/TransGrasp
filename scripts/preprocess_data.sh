ShapeNetCore_v1_root=path/to/obj_models
ShapeNetSem_root=path/to/source_Meshs_and_Grasp_labels/ShapeNetSem
ACRONYM_root=path/to/source_Meshs_and_Grasp_labels/ACRONYM
cat=$1

# Align the source model with target models, and add the source into train split.
python preprocess/transform_source_meshes.py --category ${cat} --acronym_root ${ACRONYM_root} --shapenetsem_root ${ShapeNetSem_root}
# Augment object models. See supplemental for details.
python preprocess/augment_object_meshes.py --mode train --category ${cat} --shapenetcore_root ${ShapeNetCore_v1_root}
python preprocess/augment_object_meshes.py --mode eval --category ${cat} --shapenetcore_root ${ShapeNetCore_v1_root}
python preprocess/write_model_points.py --category ${cat}   
# Generate SDFs which are used for training DIF-decoder.
python preprocess/generate_SDFs.py --category ${cat} --mode train
python preprocess/generate_SDFs.py --category ${cat} --mode eval
# Obtain single-view point clouds by rendering meshes.
python preprocess/render_depth_images.py --category ${cat} --mode train --render_num 100
python preprocess/render_depth_images.py --category ${cat} --mode eval --render_num 100