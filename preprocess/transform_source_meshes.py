'''
Align source models in ShapeNetSem with models in ShapeNetCore.v1.
'''
import sys, os
import argparse
import numpy as np
import trimesh
import logging

sys.path.append('./')
from tools.utils import rot_X, rot_Y, rot_Z, load_grasps, create_gripper_marker, save_ply, trimesh_scene_to_mesh


parser = argparse.ArgumentParser()
parser.add_argument("--category", type=str, default='mug')
parser.add_argument("--acronym_root", type=str, default='')
parser.add_argument("--shapenetsem_root", type=str, default='')
parser.add_argument("--vis", action="store_true", default=False)
parser.add_argument("--num_grasps", type=int, default=50, help="Number of grasps to show.")
args = parser.parse_args()

SOURCE_DICT = {
    'mug' : {
        'grasp_file_name': 'Mug_62634df2ad8f19b87d1b7935311a2ed0_0.02328042176991366.h5',
        'mesh_file_name' : '62634df2ad8f19b87d1b7935311a2ed0.obj',
    },
    'bottle' : {
        'grasp_file_name': 'Bottle_3108a736282eec1bc58e834f0b160845_0.01685461529193486.h5',
        'mesh_file_name' : '3108a736282eec1bc58e834f0b160845.obj',
    },
    'bowl' : {
        'grasp_file_name': 'Bowl_8bb057d18e2fcc4779368d1198f406e7_0.00039522838962311325.h5',
        'mesh_file_name' : '8bb057d18e2fcc4779368d1198f406e7.obj',
    },
}
def save_transform(cate, mesh_path, scale):
    mesh_tmp = trimesh.load_mesh(mesh_path, file_type='obj', skip_materials=True)
    model_size = np.linalg.norm(mesh_tmp.bounds[1] - mesh_tmp.bounds[0])
    mesh_tmp.apply_scale(scale)
    if cate == 'mug':
        max_v = mesh_tmp.bounds[1]
        t = np.array([-max_v[0]/2, -(max_v[1]-max_v[0]/2), -max_v[2]/2],dtype=np.float32)
    elif cate == 'bottle':
        center_points = (mesh_tmp.bounds[1] + mesh_tmp.bounds[0]) / 2
        t = -center_points
    elif cate == 'bowl':
        center_points = (mesh_tmp.bounds[1] + mesh_tmp.bounds[0]) / 2
        t = -center_points
    trans1 = np.eye(4)
    trans1[:3,3] = t
    mesh_tmp.apply_transform(trans1)
    trans2 = np.eye(4)
    rot_dict = {
        'mug': rot_X(-np.pi/2) @ rot_Z(np.pi/2),
        'bottle': np.eye(3),
        'bowl': rot_X(-np.pi/2),
    }
    rot = rot_dict[cate]
    trans2[:3,:3] = rot
    mesh_tmp.apply_transform(trans2)
    os.makedirs('datasets/acronym/{}'.format(cate), exist_ok=True)
    np.savez('datasets/acronym/{}/transform_matrix'.format(cate), RT=trans2 @ trans1, scale=1/(scale*model_size))
    return mesh_tmp

def main():
    # Redirect trimesh log to file.
    log = logging.getLogger('trimesh')
    file_handler = logging.FileHandler('trimesh_logs.log')
    log.addHandler(file_handler)

    grasp_file_name = SOURCE_DICT[args.category]['grasp_file_name']
    mesh_file_name = SOURCE_DICT[args.category]['mesh_file_name']
    grasp_file_path = os.path.join(args.acronym_root, grasp_file_name)
    mesh_file_path = os.path.join(args.shapenetsem_root, mesh_file_name)
    obj_id = mesh_file_name.replace('.obj', '')
    scale = float(grasp_file_name.split('_')[-1].rstrip('.h5'))
    obj_mesh = save_transform(args.category, mesh_file_path, scale)
    obj_mesh = trimesh_scene_to_mesh(obj_mesh)
    norms = np.linalg.norm(obj_mesh.bounds[1] - obj_mesh.bounds[0])
    obj_mesh.apply_scale(1 / norms)
    os.makedirs('datasets/obj/{}/train/{}'.format(args.category, obj_id), exist_ok=True)
    if not os.path.exists('datasets/obj/{}/train/{}/0.ply'.format(args.category, obj_id)):
        save_ply(obj_mesh.vertices, obj_mesh.faces, 'datasets/obj/{}/train/{}/0.ply'.format(args.category, obj_id))        
    os.system('cp {} {}'.format(grasp_file_path, 'datasets/acronym/' + args.category + '/'))

    if args.vis:
        T, success = load_grasps(grasp_file_path)
        trans_matrix = np.load('datasets/acronym/{}/transform_matrix.npz'.format(args.category))
        successful_grasps = [
            create_gripper_marker(color=[0,220,0,200], tube_radius=0.005, sections=8).apply_transform(trans_matrix['RT'] @ t).apply_scale(trans_matrix['scale'])
            for t in T[np.random.choice(np.where(success == 1)[0], args.num_grasps)]
        ]
        obj_mesh.visual.face_colors = np.array([0xA0, 0xA0, 0xA0, 255])
        trimesh.Scene([obj_mesh] + successful_grasps).show()

if __name__ == "__main__":
    main()