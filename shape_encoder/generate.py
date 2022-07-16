import argparse, yaml, os, sys
import torch
import trimesh
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
sys.path.append('./')
sys.path.append('./DIF_decoder')

from shape_encoder.data import PartPointsDatset
from shape_encoder.model import ShapeEncoder
from DIF_decoder.dif_net import DeformedImplicitField
from DIF_decoder.sdf_meshing import create_mesh
from DIF_decoder.calculate_chamfer_distance import compute_chamfer

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default = 'datasets')
parser.add_argument('--num_workers', type=int, default = 0)
parser.add_argument('--category', type=str, default='')
parser.add_argument('--mode', type=str, default='')
parser.add_argument('--num_points', type=int, default=1024)
parser.add_argument('--encoder_path', type=str, default='shape_encoder/output/')                                                  
opt = parser.parse_args()

def generate():

    encoder = ShapeEncoder()
    encoder.load_state_dict(torch.load(opt.encoder_path))
    encoder.cuda()

    opt.dif_config = 'DIF_decoder/configs/generate/{}.yml'.format(opt.category)
    with open(opt.dif_config,'r') as stream:
        meta_params = yaml.safe_load(stream)
    DIF_decoder = DeformedImplicitField(**meta_params)
    DIF_decoder.load_state_dict(torch.load(meta_params['checkpoint_path']))
    DIF_decoder.cuda()

    dataset = PartPointsDatset(opt.data_root, opt.category, opt.mode, opt.num_points)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    chamfer_dist = []

    vis_list = [1,3,4,7]
    cnt = 0
    for _, (obj_pcs, gt_codes, ply_paths, numbers) in tqdm(enumerate(dataloader)):
        
        
        cuda_obj_pcs = obj_pcs.cuda()
        cuda_gt_codes = gt_codes.cuda()
    
        cuda_pred_codes = encoder(cuda_obj_pcs)

        for i in range(obj_pcs.size()[0]):
            cnt += 1
            # if ply_paths[i] != 'e984fd7e97c2be347eaeab1f0c9120b7/8' \
            #     and ply_paths[i] != 'e9499e4a9f632725d6e865157050a80e/2' \
            #          and ply_paths[i] != 'f7d776fd68b126f23b67070c4a034f08/4':
            #     print(ply_paths[i])
            #     continue
            cuda_pred_code = cuda_pred_codes[i]
            cuda_gt_code = cuda_gt_codes[i]
            ply_path = ply_paths[i]
            # print(ply_path)
            recon_pts = create_mesh(DIF_decoder, 'shape_encoder/generate_plys/pred', embedding=cuda_pred_code, N=128, get_color=True)
            create_mesh(DIF_decoder, 'shape_encoder/generate_plys/DIF_decoder', embedding=cuda_gt_code, N=128)
            
            gt_path = os.path.join('datasets/{}/surface_pts_n_normal/{}.mat'.format(opt.category, ply_path))
            gt_pts = loadmat(gt_path)['p']
            gt_pts = gt_pts[:,:3]
            cd = compute_chamfer(recon_pts, gt_pts)
            print(ply_path,'\tcd:%f'%cd)
            chamfer_dist.append(cd)

            if cnt in vis_list:
                pred_ply = trimesh.load('shape_encoder/generate_plys/pred.ply', file_type='ply')
                pc_input = trimesh.PointCloud(2 * cuda_obj_pcs[i].detach().cpu().numpy() + np.array([0, 1.5, 0],dtype=np.float), 
                                             colors=[0x98, 0x9F, 0xD9, 211])
                dif_ply = trimesh.load('shape_encoder/generate_plys/DIF_decoder.ply', file_type='ply')
                gt_ply = trimesh.load(os.path.join(opt.data_root, 'obj', opt.category, opt.mode, ply_path + '.ply'))
                trans = np.eye(4)
                trans[1,3] = -1.5
                trans[1,3] = 0
                trans[0,3] = 1.5
                pred_ply.apply_transform(trans)
                trans[1,3] = 1.5
                gt_ply.apply_scale(2).apply_transform(trans)
                # trimesh.Scene([dif_ply, pred_ply, pc_input, gt_ply]).show()
                trimesh.Scene([pc_input]).show()
                trimesh.Scene([dif_ply]).show()
                trimesh.Scene([pred_ply]).show()
            # break
        # break
    print(len(chamfer_dist))
    print('average cd:', np.mean(np.array(chamfer_dist)))

if __name__ == '__main__':
    generate()