import pickle, argparse, os, sys, yaml
from tqdm import tqdm
import numpy as np
import trimesh
import torch

sys.path.append('./')
sys.path.append('./DIF_decoder')
from shape_encoder.model import ShapeEncoder
from shape_encoder.data import PartPointsDatset
from DIF_decoder.sdf_meshing import create_mesh
from tools.utils import cond_mkdir, get_model

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='datasets')
parser.add_argument('--category', type=str, default='')
parser.add_argument('--mode', type=str, default='eval')
parser.add_argument('--num_points', type=int, default=1024)
parser.add_argument('--shape_model_path', type=str, default='')
parser.add_argument('--resolution', type=int, default=64)
parser.add_argument('--vis', action="store_true", default=False)
opt = parser.parse_args()

GRASP_DATA_ROOT = 'grasp_data'

if __name__ == '__main__':
    print(opt)
    sources_dict = {
        'mug': '62634df2ad8f19b87d1b7935311a2ed0/0.pkl',
        'bottle': '3108a736282eec1bc58e834f0b160845/0.pkl',
        'bowl': '8bb057d18e2fcc4779368d1198f406e7/0.pkl',
    }
    source_filename = sources_dict[opt.category]

    opt.dif_config = 'DIF_decoder/configs/generate/{0}.yml'.format(opt.category)

    dataset = PartPointsDatset(opt.data_root, opt.category, opt.mode, opt.num_points)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    shape_encoder = ShapeEncoder()
    shape_encoder.load_state_dict(torch.load(opt.shape_model_path))
    shape_encoder.eval()
    shape_encoder.cuda()

    with open(os.path.join(opt.dif_config),'r') as stream: 
        meta_params = yaml.safe_load(stream)
    dif_model = get_model(meta_params)
    dif_model.cuda()
        
    temp_grasps_dir = meta_params['experiment_name'] + '_refine'
    opt.save_dir = os.path.join(GRASP_DATA_ROOT, meta_params['experiment_name'])
    temp_path = os.path.join(GRASP_DATA_ROOT, temp_grasps_dir, 'train/template/template.pkl')
    with open(temp_path, 'rb') as f:
        panda_file_info_temp = pickle.load(f)
    src_path = os.path.join(GRASP_DATA_ROOT, meta_params['experiment_name'], 'train', source_filename)
    with open(src_path, 'rb') as f:
        panda_file_info_src = pickle.load(f)

    recon_mesh_filename = 'grasp_transfer/mesh_holder' if opt.vis else ''

    with tqdm(total=len(dataloader)) as pbar:
        for obj_pcs, file_names, numbers in dataloader:
            
            cuda_obj_pcs = obj_pcs.cuda()
            cuda_pred_codes = shape_encoder(cuda_obj_pcs)

            for i in range(obj_pcs.size()[0]):
                arr_mesh_points = create_mesh(dif_model, filename=recon_mesh_filename, 
                                            embedding=cuda_pred_codes[i], N=opt.resolution,
                                            get_color=False)
                
                if opt.vis:
                    pc = trimesh.PointCloud(obj_pcs[i].cpu().numpy(), colors=[0x98, 0x9F, 0xD9, 211])
                    mesh = trimesh.load('grasp_transfer/mesh_holder.ply')
                    trans = np.eye(4); trans[0,3]+=0.7
                    mesh.apply_scale(0.5).apply_transform(trans)
                    mesh.visual.face_colors = np.array([0xDB, 0xBA, 0xCE, 255])
                    gt_mesh = trimesh.load('{0}/obj/{1}/{2}/{3}.ply'.format(opt.data_root, opt.category, opt.mode, file_names[i]))
                    trans = np.eye(4); trans[0,3]+=1.4
                    gt_mesh.apply_transform(trans)
                    gt_mesh.visual.face_colors = np.array([0x98, 0x9F, 0xD9, 180])
                    trimesh.Scene([pc, mesh, gt_mesh]).show()

                tensor_ins_points = torch.FloatTensor(arr_mesh_points).cuda()
                tensor_deform_points = dif_model.get_template_coords(tensor_ins_points, cuda_pred_codes[i]).squeeze().detach().cpu().numpy().astype(np.float32)

                transfer_info = {'filename': file_names[i], 
                                 'code': cuda_pred_codes[i].detach().cpu().numpy(), 
                                 'grasp_params':[]}
                for index, grasp_info in enumerate(panda_file_info_temp['grasp_params'][:1000]):
                    index_left = np.argmin(np.linalg.norm(tensor_deform_points - grasp_info['left_points'], axis=1))
                    index_right = np.argmin(np.linalg.norm(tensor_deform_points - grasp_info['right_points'], axis=1))

                    if index_left == index_right:
                        # print('failure grasp index: ', index)
                        continue
                    obj_left_points = arr_mesh_points[index_left]
                    obj_right_points = arr_mesh_points[index_right]
                    transfer_info['grasp_params'].append({
                        'left_points':obj_left_points,
                        'right_points':obj_right_points,
                        'approach_vector':grasp_info['approach_vector'],
                        'depth':grasp_info['depth'],
                    })
                direct_info = {'filename': file_names[i],
                               'code': cuda_pred_codes[i].detach().cpu().numpy(), 
                               'grasp_params':panda_file_info_src['grasp_params'][:1000],
                                }
                pbar.update(1)

                if not opt.vis:
                    gt_sRT = np.load(os.path.join(opt.data_root, 'render_pc', opt.category, opt.mode, file_names[i], 'PC_cam_sRT_{0}.npz'.format(numbers[i])))
                    transfer_info.update({'gt_scale': gt_sRT['scale']})
                    direct_info.update({'gt_scale': gt_sRT['scale']})
                    save_path = os.path.join(opt.save_dir, opt.mode, file_names[i])
                    cond_mkdir(save_path)
                    with open(os.path.join(save_path, numbers[i]+'_tf.pkl'), 'wb') as f:
                        pickle.dump(transfer_info, f)
                    with open(os.path.join(save_path, numbers[i]+'_dm.pkl'), 'wb') as f:
                        pickle.dump(direct_info, f)