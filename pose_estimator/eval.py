import argparse, sys
import torch
import numpy as np
import torch.nn.functional as F

sys.path.append('./')
from pose_estimator.data import PartPointsDatset
from pose_estimator.model import DeformNet
from pose_estimator.utils import estimateSimilarityTransform, compute_sRT_errors

parser = argparse.ArgumentParser(description='Pose Network')
parser.add_argument('--data_root', type=str, default = 'datasets')
parser.add_argument('--category', type=str, default='mug')
parser.add_argument('--mode', type=str, default='eval')
parser.add_argument('--num_points', type=int, default=1024)
parser.add_argument('--pose_model_path', type=str, default='')                                                   
opt = parser.parse_args()

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

def eval():

    mean_shapes = np.load('pose_estimator/assets/mean_points_emb.npy')
    CLASS_MAP_FOR_CATEGORY = {'bottle':1, 'bowl':2, 'camera':3, 'can':4, 'laptop':5, 'mug':6}
    prior = mean_shapes[CLASS_MAP_FOR_CATEGORY[opt.category]-1]
    prior = torch.from_numpy(prior).float().cuda()
    
    net = DeformNet()
    net.load_state_dict(torch.load(opt.pose_model_path))
    net.cuda()

    dataset = PartPointsDatset(opt.data_root, opt.category, opt.mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    R_err_list = []
    t_err_list = []
    s_IoU_list = []

    for i, (cam_pcs, gt_labels, ply_paths, _) in enumerate(dataloader):
        
        cuda_cam_pcs = cam_pcs.cuda()
        cuda_prior = prior.clone().view(1, 1024, 3).repeat(cam_pcs.size()[0], 1, 1)
        assign_mat, deltas = net(cuda_cam_pcs, cuda_prior)
        inst_shape = cuda_prior + deltas
        assign_mat = F.softmax(assign_mat, dim=2)
        f_coords = torch.bmm(assign_mat, inst_shape)  # bs x n_pts x 3
        f_coords = f_coords.detach().cpu().numpy()
    
        for i in range(cam_pcs.size()[0]):
            nocs_coords = f_coords[i]
            _, _, _, pred_sRT = estimateSimilarityTransform(nocs_coords, cam_pcs[i].numpy())
            if pred_sRT is None:
                pred_sRT = np.identity(4, dtype=float)
            gt_sRT = np.eye(4)
            gt_sRT[:3,:3] = gt_labels['rotation'][i].numpy() * gt_labels['scale'][i].numpy()
            gt_sRT[:3, 3] = gt_labels['translation'][i].numpy()

            RT_err = compute_sRT_errors(pred_sRT, gt_sRT, opt.category, 1)

            R_err_list.append(RT_err[0])
            t_err_list.append(RT_err[1])
            s_IoU_list.append(RT_err[2])
        
    print('total test instances: ', len(R_err_list))
    print('mean R error: ', np.mean(R_err_list))
    print('mean t error: ', np.mean(t_err_list))
    print('mean s error: ', np.mean(s_IoU_list))

if __name__ == '__main__':
    eval()
