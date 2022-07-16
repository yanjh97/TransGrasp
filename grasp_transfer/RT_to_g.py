import numpy as np
from tqdm import tqdm
import os, sys, yaml, configargparse, pickle
import torch
import trimesh

sys.path.append('./')
from tools.utils import *

p = configargparse.ArgumentParser()
p.add_argument('--category', type=str, default='')
p.add_argument('--vis', action='store_true', default=False)
p.add_argument('--config_root', type=str, default='DIF_decoder/configs/generate/')
p.add_argument("--num_grasps", type=int, default=30, help="Number of grasps.")
opt = p.parse_args()
GRASP_DATA_ROOT = 'grasp_data'

def panda_projection(category, meta_params, params):

    model = get_model(meta_params)

    # load grasps
    for file in os.listdir('datasets/acronym/{0}'.format(category)):
        if file.endswith('h5'):
            grasp_file = file
            break
    grasp_pose, success = load_grasps('datasets/acronym/{0}/{1}'.format(category, grasp_file))
    print("total grasp pose num: {0}, success num: {1}, fail num : {2}".format(grasp_pose.shape[0], np.sum(success == 1),
                                                                                                    np.sum(success == 0)))
    sRT = np.load('datasets/acronym/{0}/transform_matrix.npz'.format(category))

    shape_code = load_code(model, meta_params, params)

    gripper_line_start = np.array([[-4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
                                   [4.100000e-02, -7.27595772e-12, 6.59999996e-02]],dtype=np.float32)
    gripper_line_end = np.array([[-4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
                                 [4.100000e-02, -7.27595772e-12, 1.12169998e-01]],dtype=np.float32)
    
    sdf_thres = 0.5 * 1e-2
    row_split_num = 100
    col_split_num = 200
    obj_line_start = gripper_line_start
    obj_line_end = gripper_line_end

    panda_file_info={'filename': params['filename'], 'grasp_params':[], 'RTs':[]}

    with tqdm(total=len(grasp_pose[np.where(success==1)[0]])) as pbar:
        for t in grasp_pose[np.where(success==1)[0]]:
            pbar.update(1)
            trans = sRT['RT'] @ t
            tf_gripper_line_start = gripper_line_start @ trans[:3,:3].T + trans[:3,3]
            tf_gripper_line_end = gripper_line_end @ trans[:3,:3].T + trans[:3,3]
            # get one bound line where gipper interact with objects
            # from gripper_line_start to gripper_line_end
            index_start = -1
            for i in range(0, row_split_num+1):
                line = tf_gripper_line_start * (1 - i / row_split_num) + tf_gripper_line_end * (i / row_split_num)
                line = line * sRT['scale'] * 2  # X2 is to calculate SDF value of input points
                points_list = np.linspace(line[0], line[1], num=col_split_num).astype(np.float32)
                points_list_cuda = torch.from_numpy(points_list).cuda()
                sdf = model.inference(points_list_cuda, shape_code).squeeze().detach().cpu().numpy().astype(np.float32)
                if np.amin(np.abs(sdf)) < sdf_thres:
                    obj_line_start = line
                    index_start = i
                    break
            # get another bound line where gipper interact with objects
            # from gripper_line_end to gripper_line_start
            index_end = -100
            for i in range(0, row_split_num+1):
                line = tf_gripper_line_start * (i / row_split_num) + tf_gripper_line_end * (1 - i / row_split_num)
                line = line * sRT['scale'] * 2  # "X 2" is to calculate SDF value of input points
                points_list = np.linspace(line[0], line[1], num=col_split_num).astype(np.float32)         
                points_list_cuda = torch.from_numpy(points_list).cuda()
                sdf = model.inference(points_list_cuda, shape_code).squeeze().detach().cpu().numpy().astype(np.float32)
                if np.amin(np.abs(sdf)) < sdf_thres:
                    obj_line_end = line
                    index_end = row_split_num - i
                    break

            if index_start > index_end: 
                tqdm.write("there is no object part included by gripper") 
                continue
            else:

                line_middle = (obj_line_start + obj_line_end) / 2
                points_list = np.linspace(line_middle[0], line_middle[1], num=col_split_num).astype(np.float32)
                points_list_cuda = torch.from_numpy(points_list).cuda()
                sdf = model.inference(points_list_cuda, shape_code).squeeze().detach().cpu().numpy().astype(np.float32)

                grasp_points_index = np.where(np.abs(sdf)<sdf_thres)[0]

                split_left_and_right_index = -1
                for i in range(grasp_points_index.shape[0]):
                    if grasp_points_index[i] - grasp_points_index[i-1] > 2:
                        split_left_and_right_index = i
            
                if split_left_and_right_index == -1:
                    tqdm.write("there is only one grasp point!")
                    continue
                else:
                    left_nearest_index = np.argmin(np.abs(sdf[grasp_points_index[:split_left_and_right_index]]))
                    right_nearest_index = np.argmin(np.abs(sdf[grasp_points_index[split_left_and_right_index:]]))

                    left_points_index = grasp_points_index[:split_left_and_right_index][left_nearest_index]
                    right_points_index = grasp_points_index[split_left_and_right_index:][right_nearest_index]
                    
                    depth = (gripper_line_start[0,2] * (1-index_start/row_split_num) + gripper_line_end[0,2] * (index_start/row_split_num) + \
                            gripper_line_start[0,2] * (1-index_end/row_split_num) + gripper_line_end[0,2] * (index_end/row_split_num)) / 2

                    panda_file_info['grasp_params'].append({
                        'left_points':points_list[left_points_index],
                        'right_points':points_list[right_points_index],
                        'approach_vector':trans[:3, 2],
                        'depth':depth
                    })
                    obj_grasp_points = np.array([points_list[left_points_index], points_list[right_points_index]])
            if opt.vis:
                # obj_recon = load_recon_ply(meta_params, params)
                # obj_recon.apply_scale(0.5)
                obj = trimesh.load('datasets/obj/{0}/train/{1}.ply'.format(category, params['filename']))
                gripper = create_gripper_marker(color=[0, 255, 0]).apply_transform(sRT['RT'] @ t).apply_scale(sRT['scale'])
                color_sdf = np.zeros([points_list.shape[0], 4])
                color_sdf[:,:] = np.array([0,255,0,255], dtype=np.uint8)
                color_sdf[left_points_index,:] = np.array([255,0,0,255], dtype=np.uint8)
                color_sdf[right_points_index,:] = np.array([0,0,255,255], dtype=np.uint8)
                gripper_points = trimesh.points.PointCloud(points_list/2, colors=color_sdf)
                obj_points = trimesh.points.PointCloud(obj_grasp_points/2, colors=[0,0,0,255])
                trimesh.Scene([obj, gripper, gripper_points, obj_points]).show()


    with open(os.path.join(save_path, pms['part'] + '.pkl'), 'wb') as f:
        pickle.dump(panda_file_info, f)

if __name__ == '__main__':

    opt.config = os.path.join(opt.config_root, opt.category+'.yml')

    with open(os.path.join(opt.config),'r') as stream: 
        meta_params = yaml.safe_load(stream)

    cond_mkdir(os.path.join(GRASP_DATA_ROOT, meta_params['experiment_name'], 'train'))

    # There are objects from ACRONYM (which uses OBJ from ShapeNetSem) below.
    grasp_source_info = {
        'mug':{'mode':'train', 'inst_name':'62634df2ad8f19b87d1b7935311a2ed0', 'part':'0',
                'filename':'62634df2ad8f19b87d1b7935311a2ed0/0'},
        'bottle':{'mode':'train', 'inst_name':'3108a736282eec1bc58e834f0b160845', 'part':'0',
                'filename':'3108a736282eec1bc58e834f0b160845/0'}, 
        'bowl':{'mode':'train', 'inst_name':'8bb057d18e2fcc4779368d1198f406e7', 'part':'0',
                'filename':'8bb057d18e2fcc4779368d1198f406e7/0'},
    }

    pms=grasp_source_info[opt.category]

    with open('DIF_decoder/split/train/{}.txt'.format(opt.category), 'r') as f:
        i = 0
        line = f.readline().rstrip('\n')
        while (line):
            if line == pms['filename']:
                pms['subject_idx'] = i
                break
            i += 1
            line = f.readline().rstrip('\n')
    print('[{}] source object idx : {}'.format(opt.category,  pms['subject_idx']))

    save_path = os.path.join(GRASP_DATA_ROOT, meta_params['experiment_name'], pms['mode'], pms['inst_name'])
    if os.path.exists(os.path.join(save_path, pms['part'] + '.pkl')):
        exit()
    os.makedirs(save_path, exist_ok=True)
    panda_projection(opt.category, meta_params, pms)