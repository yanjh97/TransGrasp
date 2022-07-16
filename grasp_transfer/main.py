'''
Script for the whole framework.
'''
import pickle, argparse, os, yaml, sys
import numpy as np
import torch.nn.functional as F
import torch
import time

sys.path.append('./')
sys.path.append('./DIF_decoder')
from shape_encoder.model import ShapeEncoder
from pose_estimator.data import PartPointsDatset
from pose_estimator.model import DeformNet
from pose_estimator.utils import estimateSimilarityTransform
from DIF_decoder.sdf_meshing import create_mesh
from tools.utils import cond_mkdir, get_model
from grasp_transfer.refine_module import PandaRefine, get_best_results_from_pth

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='datasets')
parser.add_argument('--category', type=str, default='mug')
parser.add_argument('--mode', type=str, default='eval')
parser.add_argument('--num_points', type=int, default=1024, help='points number for point cloud')
parser.add_argument('--pose_model_path', type=str, default='')
parser.add_argument('--shape_model_path', type=str, default='')
parser.add_argument('--resolution', type=int, default=64)
parser.add_argument('--vis', action="store_true", default=False)
parser.add_argument("--batch_size", type=int, default = 1)
parser.add_argument("--lr", type=int, default = 1e-3)
parser.add_argument("--steps", type=int, default = 10)
opt = parser.parse_args()
GRASP_DATA_ROOT = 'grasp_data'

def q2R(pred_r):
    bs, _ = pred_r.size()
    pred_r = pred_r / (torch.norm(pred_r, dim=1).view(bs, 1))
    R_martix = torch.cat(((1.0 - 2.0*(pred_r[:, 2]**2 + pred_r[:, 3]**2)).view(bs, 1),\
            (2.0*pred_r[:, 1]*pred_r[:, 2] - 2.0*pred_r[:, 0]*pred_r[:, 3]).view(bs, 1), \
            (2.0*pred_r[:, 0]*pred_r[:, 2] + 2.0*pred_r[:, 1]*pred_r[:, 3]).view(bs, 1), \
            (2.0*pred_r[:, 1]*pred_r[:, 2] + 2.0*pred_r[:, 3]*pred_r[:, 0]).view(bs, 1), \
            (1.0 - 2.0*(pred_r[:, 1]**2 + pred_r[:, 3]**2)).view(bs, 1), \
            (-2.0*pred_r[:, 0]*pred_r[:, 1] + 2.0*pred_r[:, 2]*pred_r[:, 3]).view(bs, 1), \
            (-2.0*pred_r[:, 0]*pred_r[:, 2] + 2.0*pred_r[:, 1]*pred_r[:, 3]).view(bs, 1), \
            (2.0*pred_r[:, 0]*pred_r[:, 1] + 2.0*pred_r[:, 2]*pred_r[:, 3]).view(bs, 1), \
            (1.0 - 2.0*(pred_r[:, 1]**2 + pred_r[:, 2]**2)).view(bs, 1)), dim=1).contiguous().view(bs, 3, 3)
    return R_martix

def refine_grasp(grasp_info, shape_code, model, refine_pth_dir, lr=1e-3, steps=10):

    pth_dir = os.path.join(refine_pth_dir, 'refine.pth')
    # obj_scale = torch.FloatTensor(grasp_info['pred_scale']).cuda()
    obj_scale = grasp_info['pred_scale']

    model_refine = PandaRefine(grasp_info, obj_scale, shape_code)
    model_refine.cuda()
    optimizer = torch.optim.Adam(model_refine.parameters(), lr=lr)
    loss_min = 1e10

    global_best_idx = -1
    for step in range(steps):
        optimizer.zero_grad()
        losses = model_refine(model)
        loss_refine = losses['loss']
        loss_choose = losses['loss_choose']
        best_idx = losses['best_idx'] 
        if steps > 1:
            loss_refine.backward(retain_graph=True)
            optimizer.step()
        
        # Save the best model
        if loss_choose < loss_min:
            loss_min = loss_choose
            global_best_idx = best_idx
            torch.save(model_refine.state_dict(), pth_dir)
    deltas = torch.load(pth_dir)
    grasp_info['grasp_params'] = get_best_results_from_pth(grasp_info['grasp_params'], deltas, global_best_idx)
    for k, v in grasp_info['grasp_params'].items():
        grasp_info['grasp_params'][k]=v.detach().cpu().numpy() 
    return grasp_info

def batch_pairwise_dist(x, y, use_cuda=False):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    if use_cuda:
        dtype = torch.cuda.LongTensor
    else:
        dtype = torch.LongTensor
    diag_ind_x = torch.arange(0, num_points_x).type(dtype)
    diag_ind_y = torch.arange(0, num_points_y).type(dtype)

    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P

if __name__ == '__main__':
    mean_shapes = np.load('pose_estimator/assets/mean_points_emb.npy')
    CLASS_MAP_FOR_CATEGORY = {'bottle':1, 'bowl':2, 'camera':3, 'can':4, 'laptop':5, 'mug':6}
    prior = mean_shapes[CLASS_MAP_FOR_CATEGORY[opt.category]-1]
    prior = torch.from_numpy(prior).float().cuda()

    ############################################################
    # PoseNet >>> Shape Encoder >>> DIF-Decoder # DEFINATION   #
    ############################################################
    pose_net = DeformNet()
    pose_net.load_state_dict(torch.load(opt.pose_model_path))
    pose_net.eval()
    pose_net.cuda()
    shape_encoder = ShapeEncoder()
    shape_encoder.load_state_dict(torch.load(opt.shape_model_path))
    shape_encoder.eval()
    shape_encoder.cuda()
    opt.dif_config = 'DIF_decoder/configs/generate/{0}.yml'.format(opt.category)
    with open(os.path.join(opt.dif_config),'r') as stream:
        meta_params = yaml.safe_load(stream)
    dif_model = get_model(meta_params)
    dif_model.cuda()
    ############################################################
    # PoseNet >>> Shape Encoder >>> DIF-Decoder # DEFINATION   #
    ############################################################
    
    source_exp_name = meta_params['experiment_name'] + '_refine'
    target_exp_name = meta_params['experiment_name'] + '_select'
    temp_path = os.path.join(GRASP_DATA_ROOT, source_exp_name, 'train/template/template.pkl')
    save_dir = os.path.join(GRASP_DATA_ROOT, target_exp_name, opt.mode)
    
    with open(temp_path, 'rb') as f:
        panda_file_info_temp = pickle.load(f)
    
    temp_left_p = []
    temp_right_p = []
    temp_appro = []
    temp_depth = []
    # panda_file_info_temp['grasp_params'] = random.sample(panda_file_info_temp['grasp_params'],100)
    panda_file_info_temp['grasp_params'] = panda_file_info_temp['grasp_params'][:100]
    for index, grasp_info in enumerate(panda_file_info_temp['grasp_params']):
        temp_left_p.append(grasp_info['left_points'])
        temp_right_p.append(grasp_info['right_points'])
        temp_appro.append(grasp_info['approach_vector'])
        temp_depth.append(grasp_info['depth'])
    
    temp_left_p = torch.FloatTensor(temp_left_p).cuda()
    temp_right_p = torch.FloatTensor(temp_right_p).cuda()
    temp_appro = torch.FloatTensor(temp_appro).cuda()
    temp_depth = torch.FloatTensor(temp_depth).cuda()    
    
    dataset = PartPointsDatset(opt.data_root, opt.category, opt.mode, opt.num_points, 1024)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print('test num: {}'.format(len(dataset)))
    start_time = time.time()
    tf_use_time_total = 0.
    total_time = 0.
    for cam_pcs, gt_labels, file_names, numbers in dataloader:
        per_start_time = time.time()
        print(file_names)
        bs = cam_pcs.shape[0]
        cuda_cam_pcs = cam_pcs.cuda()
        cuda_prior = prior.clone().view(1, 1024, 3).repeat(bs, 1, 1)
        assign_mat, deltas = pose_net(cuda_cam_pcs, cuda_prior)
        inst_shape = cuda_prior + deltas
        assign_mat = F.softmax(assign_mat, dim=2)
        f_coords = torch.bmm(assign_mat, inst_shape)  # bs x n_pts x 3
        f_coords = f_coords.detach().cpu().numpy()
        
        # i: invert transform
        cuda_pred_i_sR = torch.zeros([bs, 3, 3], dtype=torch.float32).cuda()
        cuda_pred_s = torch.zeros([bs], dtype=torch.float32).cuda()
        cuda_pred_t = torch.zeros([bs, 1, 3], dtype=torch.float32).cuda()

        i=0 # bs=1

        _, _, _, pred_sRT = estimateSimilarityTransform(f_coords[i], cam_pcs[i].numpy())
        if pred_sRT is None:
            pred_sRT = np.identity(4, dtype=float)
        s = np.cbrt(np.linalg.det(pred_sRT[:3, :3]))
        R = pred_sRT[:3, :3] / s
        i_sR = R / s
        cuda_pred_i_sR[i] = torch.from_numpy(i_sR).float().cuda()
        cuda_pred_t[i] = torch.from_numpy(pred_sRT[:3,3]).view(1,3).float().cuda()

        cuda_obj_pcs = torch.bmm(torch.add(cuda_cam_pcs, -cuda_pred_t), cuda_pred_i_sR)
        cuda_pred_codes = shape_encoder(cuda_obj_pcs)

        arr_mesh_points = create_mesh(dif_model, filename='', 
                    embedding=cuda_pred_codes[i], N=opt.resolution, get_color=False)

        tensor_ins_points = torch.FloatTensor(arr_mesh_points).cuda()
        tensor_deform_points_cuda = dif_model.get_template_coords(tensor_ins_points, cuda_pred_codes[i]).squeeze()

        dist = batch_pairwise_dist(temp_left_p.unsqueeze(0), tensor_deform_points_cuda.unsqueeze(0), use_cuda=True)[0]
        index_left_p = torch.argmin(dist, dim=1, keepdim=False)
        dist = batch_pairwise_dist(temp_right_p.unsqueeze(0), tensor_deform_points_cuda.unsqueeze(0), use_cuda=True)[0]
        index_right_p = torch.argmin(dist, dim=1, keepdim=False)
        index_p = torch.where(index_left_p!=index_right_p)
        inst_left_p = tensor_ins_points[index_left_p[index_p], :]
        inst_right_p = tensor_ins_points[index_right_p[index_p], :]
        
        tf_use_time = time.time() - per_start_time
        tf_use_time_total += tf_use_time

        grasp_params = {
            'left_points': inst_left_p,
            'right_points': inst_right_p,
            'approach_vector': temp_appro[index_p],
            'depth': temp_depth[index_p],
        }
        pred_trans = pred_sRT
        pred_trans[:3, :3] /= s
        gt_trans = np.eye(4)
        gt_trans[:3,:3] = gt_labels['rotation'][i].numpy()
        gt_trans[:3,3] = gt_labels['translation'][i].numpy()
        trans_offset = np.linalg.inv(gt_trans) @ pred_trans
        grasp_info = {'pred_scale':s, 
                      'gt_scale': gt_labels['scale'][i].numpy(),
                      'trans_offset':trans_offset,
                      'grasp_params': grasp_params}
        
        grasp_info = refine_grasp(grasp_info, cuda_pred_codes[i], dif_model, './grasp_transfer',
                                    lr=opt.lr, steps=opt.steps)

        total_time += time.time() - start_time
        # save result
        save_path = os.path.join(save_dir, file_names[i])
        cond_mkdir(save_path)
        with open(os.path.join(save_path, numbers[i]+'_select.pkl'), 'wb') as f:
            pickle.dump(grasp_info, f)

        start_time = time.time()

    print('transfer average time: {:.2f} s'.format(tf_use_time_total/len(dataset)))
    print('all use time: {:.2f} s, average time: {:.2f} s'.format(total_time, total_time/len(dataset)))
