import os, sys, argparse
from tqdm import tqdm
import trimesh
import numpy as np
import copy
import logging

sys.path.append('./')
from tools.utils import trimesh_scene_to_mesh, save_ply


parser = argparse.ArgumentParser()
parser.add_argument("--category", type=str, default='mug')
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--shapenetcore_root", type=str, default='')
args = parser.parse_args()

CATEGORY_TO_ID = {'mug': '03797390', 'bottle': '02876657', 'bowl': '02880940'}
save_root = 'datasets/obj'

def deform_3D(pts, scale_xOz, scale_y):
    '''
    Args:
        pts: [N,3] array, vertices of mesh
        scale_xOz: float, scale size of top face of 3D bbox
        scale_y: float, scale size along Y axis
    Return:
        pts_deformed: [N,3]
    '''
    pts_deformed = pts.copy()
    y_max, y_min = np.amax(pts[:,1]), np.amin(pts[:,1])
    deform_func_xOz = lambda x : ((scale_xOz - 1) / (y_max - y_min) * x + (y_max - scale_xOz * y_min) / (y_max - y_min))
    pts_deformed[:,(0,2)] = pts_deformed[:,(0,2)] * deform_func_xOz(pts_deformed[:,1])[:, np.newaxis]
    pts_deformed[:,1] = pts_deformed[:,1] * scale_y
    return pts_deformed

def norm_pts(pts, norm_size=0.5):
    '''
    Args:
        pts: [N,3]
        norm_size: the sphere's radius of 3D bbox of normed pts
    Return:
        pts_normed: [N,3]
    '''
    radius_bbox = np.amax(abs(pts), axis=0)
    pts_normed = pts / np.linalg.norm(radius_bbox) * norm_size
    return pts_normed

def main():
    # Redirect trimesh log to file.
    log = logging.getLogger('trimesh')
    file_handler = logging.FileHandler('trimesh_logs.log')
    log.addHandler(file_handler)

    obj_mode = 'val' if args.mode == 'eval' else 'train'
    obj_path = os.path.join(args.shapenetcore_root, obj_mode, CATEGORY_TO_ID[args.category])
    save_path = os.path.join(save_root, args.category, args.mode)
    os.makedirs(save_path, exist_ok=True)

    shapenetcore_inst_list = []
    with open('raw_obj_splits/{}_{}.txt'.format(args.category, args.mode), 'r') as f:
        inst_name = f.readline().replace('\n', '')
        while inst_name:
            shapenetcore_inst_list.append(inst_name)
            inst_name = f.readline().replace('\n', '')

    for inst_name in shapenetcore_inst_list:
        if os.path.exists(os.path.join(save_path, inst_name, '0.ply')):
            continue
        path_mesh_source = os.path.join(obj_path, inst_name, 'model.obj')
        mesh_source = trimesh.load(path_mesh_source, file_type='obj', skip_materials=True)
        os.makedirs(os.path.join(save_path, inst_name), exist_ok=True)
        mesh_source = trimesh_scene_to_mesh(mesh_source)
        if args.category == 'mug': # Shift the mug model to make center of its body be origin.
            model_points = mesh_source.vertices
            shift_x = (np.amin(model_points[:, 2]) - np.amax(model_points[:, 2])) / 2 - np.amin(model_points[:, 0])
            shift = np.array([shift_x, 0.0, 0.0])
            model_points += shift
            size = 2 * np.amax(np.abs(model_points), axis=0)
            scale = 1 / np.linalg.norm(size)
            model_points *= scale
        save_ply(mesh_source.vertices, mesh_source.faces, os.path.join(save_path, inst_name, '0.ply'))

    if args.category == 'mug' or args.category == 'bowl':
        deform_list = [(0.7, 1), (0.8, 1), (1.2, 1), (1.4, 1),
                       (1, 0.7), (1, 0.8), (1, 1.2), (1, 1.4),]
    elif args.category == 'bottle':
        deform_list = [(1, 0.7), (1, 0.8), (1, 1.2), (1, 1.4),]
    else:
        raise ValueError('No defined category.')

    ids_list = os.listdir(save_path)
    print('---Augment {} {} models---'.format(args.category, args.mode))
    with tqdm(total=len(deform_list) * len(ids_list)) as pbar:
        for id in ids_list:
            path_mesh_origin = os.path.join(save_path, id, '0.ply')
            mesh_origin = trimesh.load(path_mesh_origin, file_type='ply')
            mesh_target = copy.deepcopy(mesh_origin)
            for i, (scale_xOz, scale_y) in enumerate(deform_list):
                pbar.update(1)
                path_mesh_target = os.path.join(save_path, id, '{0}.ply'.format(i+1))
                if os.path.exists(path_mesh_target):
                    continue                
                mesh_target.vertices = deform_3D(mesh_origin.vertices, scale_xOz, scale_y)
                mesh_target.vertices = norm_pts(mesh_target.vertices)
                save_ply(mesh_target.vertices, mesh_target.faces, path_mesh_target)

if __name__ == '__main__':
    main()