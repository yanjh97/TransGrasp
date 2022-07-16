import re
import torch.utils.data as data
import numpy as np
import os, random
import pickle
import math

CLASS_MAP_FOR_CATEGORY = {'bottle':1, 'bowl':2, 'camera':3, 'can':4, 'laptop':5, 'mug':6}

class PartPointsDatset(data.Dataset):
    def __init__(self, data_root, category, mode, n_pts=1024, m_pts=1024):

        self.category = category
        self.mode = mode
        self.n_pts = n_pts
        self.m_pts = m_pts
        self.category = CLASS_MAP_FOR_CATEGORY[category]
        self.sym_list = [1, 2, 4]

        assert mode in ['train', 'eval', 'val']

        _mode = 'eval' if mode == 'val' else mode
        self.data_root = os.path.join(data_root, 'render_pc', category, _mode)
        self.ply_root = os.path.join(data_root, 'obj', category, _mode)
        independent_inst_names = os.listdir(self.data_root)

        self.pc_list_path = []
        
        if mode != 'eval':
            with open(os.path.join(data_root, 'model_points', category, '{}.pkl'.format(_mode)), 'rb') as f:
                self.all_mesh_points = pickle.load(f)
            with open(os.path.join(data_root, 'gt_codes', '{0}_{1}.pkl'.format(category, _mode)), 'rb') as f:
                self.gt_codes = pickle.load(f)

        eval_sample_list = ['000']
        for inst in independent_inst_names:
            for part in os.listdir(os.path.join(self.data_root, inst)):
                if self.mode == 'eval':
                    for number in eval_sample_list:
                        file = 'PC_cam_sRT' + '_' + number + '.npz'
                        self.pc_list_path.append(os.path.join(inst, part, file))
                else:
                    for file in os.listdir(os.path.join(self.data_root, inst, part)):
                        if file.find('PC_cam_sRT') != -1:
                            self.pc_list_path.append(os.path.join(inst, part, file))

        if self.mode == 'val':
            self.pc_list_path = random.sample(self.pc_list_path, len(self.pc_list_path) // 4)


    def __len__(self) -> int:
        return len(self.pc_list_path)

    def __getitem__(self, index: int):
        
        _path = os.path.join(self.data_root, self.pc_list_path[index])

        independent_inst_name, part, pc_filename = self.pc_list_path[index].split('/')
        number = re.search("\d{3}", pc_filename)

        label_data = np.load(_path)
        pc_cam = label_data['pc']
        if len(pc_cam) >= self.n_pts:
            sample_index = np.random.choice(pc_cam.shape[0], self.n_pts, replace=False)
        else:
            sample_index = np.random.choice(pc_cam.shape[0], self.n_pts, replace=True)
        pc_cam = pc_cam[sample_index, :]

        if self.mode == 'eval':
            gt_label = {
                'scale':label_data['scale'].astype(np.float32),
                'rotation':label_data['rotation'].astype(np.float32),
                'translation':label_data['translation'].astype(np.float32),
            } 
            return pc_cam.astype(np.float32), gt_label, os.path.join(independent_inst_name, part), number.group()

        nocs_from_cam_pc = (pc_cam - label_data['translation']) @ label_data['rotation']
        nocs_from_cam_pc = nocs_from_cam_pc / label_data['scale']
        
        mesh_points = self.all_mesh_points[independent_inst_name+'/'+part]
        if len(mesh_points) >= self.m_pts:
            sample_index = np.random.choice(mesh_points.shape[0], self.m_pts, replace=False)
        else:
            sample_index = np.random.choice(mesh_points.shape[0], self.m_pts, replace=True)
        mesh_points = mesh_points[sample_index, :]

        gt_code = self.gt_codes[os.path.join(independent_inst_name, part)]

        rotation = label_data['rotation']
        if self.category in self.sym_list:
            # assume continuous axis rotation symmetry
            theta_x = rotation[0, 0] + rotation[2, 2]
            theta_y = rotation[0, 2] - rotation[2, 0]
            r_norm = math.sqrt(theta_x**2 + theta_y**2)
            s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                              [0.0,            1.0,  0.0           ],
                              [theta_y/r_norm, 0.0,  theta_x/r_norm]])
            rotation = rotation @ s_map

            nocs_from_cam_pc = nocs_from_cam_pc @ s_map

        if self.mode == 'train':
            add_t = np.clip(0.001 * np.random.randn(pc_cam.shape[0], 3), -0.005, 0.005)
            pc_cam = np.add(pc_cam, add_t)

        gt_label = {
            'scale':label_data['scale'].astype(np.float32),
            'rotation':rotation.astype(np.float32),
            'translation':label_data['translation'].astype(np.float32),
            'model_points': mesh_points.astype(np.float32),
            'code':gt_code.astype(np.float32),
            'nocs':nocs_from_cam_pc.astype(np.float32),
        }
        return pc_cam.astype(np.float32), gt_label, os.path.join(independent_inst_name, part), number.group()

if __name__ == '__main__':
    import trimesh

    dataset = PartPointsDatset('datasets/', 'mug', 'eval', 1024)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    print(len(dataset))

    for i, (cam_pcs, gt_label, _, _) in enumerate(dataloader):
        print(cam_pcs.shape, gt_label['translation'].shape)

        for i in range(cam_pcs.shape[0]):
            pc = trimesh.PointCloud(gt_label['nocs'][i])
            ply_points = trimesh.PointCloud(gt_label['model_points'][i], colors=[255,0,0,255])
            trimesh.Scene([pc,ply_points]).show()
        break