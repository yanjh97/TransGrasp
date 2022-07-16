from mesh_to_sdf import get_surface_point_cloud

import trimesh
import scipy.io as sio
import pyrender
import numpy as np
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--category", type=str, default='mug')
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--vis", action="store_true", default=False)
args = parser.parse_args()

CATEGORY_TO_ID = {'mug': '03797390', 'bottle': '02876657', 'bowl': '02880940'}
data_root = 'datasets'
split_path = 'DIF_decoder/split'

def sample_surface_sdf(mesh, number_of_points = 500000, surface_point_method='scan', sign_method='normal', scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11, min_size=0, return_gradients=False):
    if surface_point_method == 'sample' and sign_method == 'depth':
        print("Incompatible methods for sampling points and determining sign, using sign_method='normal' instead.")
        sign_method = 'normal'
    surface_point_cloud = get_surface_point_cloud(mesh, surface_point_method, 1.0, scan_count, scan_resolution, sample_point_count, calculate_normals=sign_method=='normal' or return_gradients)
    surface_point_cloud.points *= 2
    return surface_point_cloud.sample_sdf_near_surface(number_of_points, surface_point_method=='scan', sign_method, normal_sample_count, min_size, return_gradients), \
        surface_point_cloud

def sample_sdf(file_path):
    mesh = trimesh.load(file_path)
    points_with_sdf, points_with_normal = sample_surface_sdf(mesh, number_of_points=250000)
    return points_with_sdf, points_with_normal

def write_mat_data(save_path, file, points_with_sdf, points_with_normal):
    pts_sdf = np.hstack((points_with_sdf[0], points_with_sdf[1][:,np.newaxis]))
    pts_normal = np.hstack((points_with_normal.points, points_with_normal.normals))
    free_points_path = os.path.join(save_path, 'free_space_pts', file + '.mat')
    surface_points_path = os.path.join(save_path, 'surface_pts_n_normal', file + '.mat')
    sio.savemat(free_points_path, {'p_sdf':pts_sdf})
    sio.savemat(surface_points_path, {'p':pts_normal})

def show_sdf(points_free, sdf_free, points_surface=None):
    colors = np.zeros(points_free.shape)
    colors[sdf_free < 0, 2] = 1
    colors[sdf_free > 0, 0] = 1
    scene = pyrender.Scene()
    cloud_free = pyrender.Mesh.from_points(points_free, colors=colors)
    scene.add(cloud_free)
    if points_surface is not None:
        colors = np.zeros(points_surface.shape)
        colors[:, 1] = 1
        cloud_surface = pyrender.Mesh.from_points(points_surface, colors=colors)
        scene.add(cloud_surface)
    pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

def write_split(obj_path, category, mode):
    obj_path = os.path.join(obj_path, category, mode)
    aug_inst_list = []
    os.makedirs(os.path.join(split_path, mode), exist_ok=True)
    for inst_name in os.listdir(obj_path):
        for part in os.listdir(os.path.join(obj_path, inst_name)):
            aug_inst_list.append(inst_name + '/' + part.split('.')[0])
    # with open(os.path.join(split_path, mode, category + '.txt'), 'w') as f:
    #     for aug_inst in aug_inst_list[:-1]:            
    #         f.write(aug_inst + '\n')
    #     f.write(aug_inst_list[-1])
    return aug_inst_list

def main():
    save_path = os.path.join(data_root, 'sdf', args.category, args.mode)
    obj_path = os.path.join(data_root, 'obj', args.category, args.mode)
    aug_inst_list = write_split(os.path.join(data_root, 'obj'), args.category, args.mode)

    os.makedirs(os.path.join(save_path, 'free_space_pts'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'surface_pts_n_normal'), exist_ok=True)

    print('---Generate {} {} SDFs---'.format(args.category, args.mode))
    with tqdm(total=len(aug_inst_list)) as pbar:
        for file in aug_inst_list:
            id, part = file.split('/')
            os.makedirs(os.path.join(save_path, 'free_space_pts', id), exist_ok=True)
            os.makedirs(os.path.join(save_path, 'surface_pts_n_normal',id), exist_ok=True)
            if os.path.exists(os.path.join(save_path, 'free_space_pts', file + '.mat')):
                pbar.update(1)
                continue
            points_with_sdf, points_with_normal = sample_sdf(os.path.join(obj_path, file + '.ply'))
            write_mat_data(save_path, file, points_with_sdf, points_with_normal)
            if args.vis:
                show_sdf(points_with_sdf[0], points_with_sdf[1], points_with_normal.points)
            pbar.update(1)

if __name__ == '__main__':
    main()