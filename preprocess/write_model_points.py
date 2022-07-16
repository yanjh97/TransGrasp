import os, sys, argparse
import pickle
from plyfile import PlyData
import numpy as np
from tqdm import tqdm

sys.path.append('./')
from pose_estimator.utils import sample_points_from_mesh

parser = argparse.ArgumentParser()
parser.add_argument("--category", type=str, default='mug')
args = parser.parse_args()

data_root = 'datasets'
category = args.category
sample_num = 2048

def write_model_points(mode):
    ply_root = os.path.join(data_root, 'obj', category, mode)
    points_dict = dict()
    inst_list = os.listdir(ply_root)
    part_list = os.listdir(os.path.join(ply_root, inst_list[0]))
    with tqdm(total=len(inst_list) * len(part_list)) as pbar:
        for inst_name in inst_list:
            for ply_file in os.listdir(os.path.join(ply_root, inst_name)):
                _ply_path = os.path.join(ply_root, inst_name, ply_file)
                ply = PlyData.read(_ply_path)
                point_ele = ply['vertex']
                vertices = np.stack([point_ele['x'],point_ele['y'],point_ele['z']],axis=1)
                faces = np.stack(ply['face']['vertex_indices'],axis=0)
                points = sample_points_from_mesh(vertices, faces, sample_num)
                points_dict[inst_name + '/' + ply_file.split('.')[0]] = points
                pbar.update(1)
    os.makedirs(os.path.join(data_root, 'model_points', category), exist_ok=True)
    with open(os.path.join(data_root, 'model_points', category, '{}.pkl'.format(mode)), 'wb') as f:
        pickle.dump(points_dict, f)

if __name__ == '__main__':
    write_model_points('train')
    write_model_points('eval')