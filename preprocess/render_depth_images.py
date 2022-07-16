import os, argparse
import trimesh
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, BlendParams,
    MeshRenderer, MeshRasterizer, HardPhongShader
)

parser = argparse.ArgumentParser(description='render point cloud')
parser.add_argument('--category', type=str, default='mug')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--render_num', type=int, default= 100)
parser.add_argument('--vis', action='store_true', default=False)       
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def load_plys_as_meshes(files: list, scales: list, device=None):
    """
    Load meshes from a list of .ply files using the load_ply function, and
    return them as a Meshes object. This only works for meshes which have a
    single texture image for the whole mesh. See the load_ply function for more
    details. material_colors and normals are not stored.

    Args:
        f: A list of file-like objects (with methods read, readline, tell,
        and seek), pathlib paths or strings containing file names.
        device: Desired device of returned Meshes. Default:
            uses the current device for the default tensor type.
        load_textures: Boolean indicating whether material files are loaded

    Returns:
        New Meshes object.
    """
    assert len(files) == len(scales)
    mesh_list = []
    for i, f_obj in enumerate(files):
        verts, faces = load_ply(f_obj)
        mesh = Meshes(verts=[(verts*scales[i]).to(device)], faces=[faces.to(device)])
        mesh_list.append(mesh)
    if len(mesh_list) == 1:
        return mesh_list[0]
    return join_meshes_as_batch(mesh_list)


class DepthRender:
    def __init__(self, dist, elev, azim) -> None:

        self.R, self.T = look_at_view_transform(dist, elev, azim, device=device)
        self.render_num = dist.shape[0]

        width = 192; height = 192; fov = 20
        cx = width / 2 ; cy = height / 2
        fx = cx / np.tan(fov * np.pi / 180 / 2)
        fy = cy / np.tan(fov *np.pi / 180 / 2)

        raster_settings = RasterizationSettings(
            image_size=width,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
            # max_faces_per_bin=20000
        )

        u = np.array(list(np.ndindex((height, width)))).reshape(height, width, 2)[:, :, 1]
        v = np.array(list(np.ndindex((height, width)))).reshape(height, width, 2)[:, :, 0]

        self.cameras = FoVPerspectiveCameras(device=device, R=self.R, T=self.T, fov=fov)
        
        self.xmap = (u - cx) / fx
        self.ymap = (v - cy) / fy

        # self.xmap = np.repeat(self.xmap[np.newaxis, ...], self.render_num, axis=0)
        # self.ymap = np.repeat(self.ymap[np.newaxis, ...], self.render_num, axis=0)

        self.rasterizer = MeshRasterizer(
                     cameras=self.cameras,
                     raster_settings=raster_settings
                    )

    def render(self, meshes):

        fragments = self.rasterizer(meshes)
        # print(fragments.zbuf.size())
        depths = fragments.zbuf[:,:,:,0]
        
        R_list = []
        t_list = []
        depth_list = []
        coords_CAM_list = []
        coords_OBJ_list = []

        for i in range(self.render_num):
            depth = depths[i].cpu().squeeze().numpy()
            X_ = self.xmap[depth > -1]  # exclude infinity
            Y_ = self.ymap[depth > -1]  # exclude infinity
            depth_ = depth[depth > -1]  # exclude infinity

            X = X_ * depth_
            Y = Y_ * depth_
            Z = depth_

            R = torch.mm(self.R[i], torch.FloatTensor([[-1,0,0],[0,-1,0],[0,0,1]]).to(device))
            R = R.transpose(1,0).cpu().numpy()
            t = self.T[i].cpu().numpy()

            coords_CAM = np.stack([X, Y, Z]).T  # shape: num_points * 3
            coords_OBJ = (coords_CAM  - t) @ R

            depth_list.append(depth)
            R_list.append(R)
            t_list.append(t)
            coords_CAM_list.append(coords_CAM)
            coords_OBJ_list.append(coords_OBJ)

        return depth_list, R_list, t_list, coords_CAM_list, coords_OBJ_list


def main():
    vis  = args.vis
    cate = args.category
    mode = args.mode
    render_num = args.render_num
    save_root = os.path.join('datasets/render_pc/{}'.format(cate), mode)
    obj_root = os.path.join('datasets/obj/{}'.format(cate), mode)
    inst_names = os.listdir(obj_root)
    print('---Render {} {} models---'.format(cate, mode))
    for i, inst_name in enumerate(inst_names):
        print(i, '  ', inst_name)
        for ply_file in tqdm(os.listdir(os.path.join(obj_root, inst_name))):
            save_path = os.path.join(save_root, inst_name, ply_file.replace('.ply', ''))
            os.makedirs(save_path, exist_ok=True)

            if cate == 'mug':
                scale_list = np.random.normal(0.195, 0.02, size=render_num)
                scale_list = np.clip(scale_list, 0.165, 0.235)
            elif cate == 'bowl':
                scale_list = np.random.uniform(0.185, 0.238, size=render_num)
            elif cate == 'bottle':
                scale_list = np.random.uniform(0.176, 0.202, size=render_num)
            else:
                raise NotImplementedError

            dist = np.random.uniform(0.6, 1.1, size=render_num)
            elev = np.random.uniform(5.0, 55.0, size=render_num)
            azim = np.random.uniform(0.0, 360.0, size=render_num)

            obj_paths = [os.path.join(obj_root, inst_name, ply_file), ] * render_num
            meshes = load_plys_as_meshes(obj_paths, scale_list, device=device)
            depth_renderer = DepthRender(dist, elev, azim)
            depth_list, R_list, t_list, coords_CAM_list, coords_OBJ_list = depth_renderer.render(meshes)

            for i in range(render_num):
                # for offline visualization
                # depth_save_path = os.path.join(save_path, 'depth_{0}.png'.format(str(i).zfill(3)))
                # plt.imsave(depth_save_path, depth_list[i])

                camera_pc_path = os.path.join(save_path, 'PC_cam_sRT_{0}'.format(str(i).zfill(3)))
                np.savez(camera_pc_path, pc=coords_CAM_list[i], rotation=R_list[i], translation=t_list[i], scale=scale_list[i])

                obj_pc_path = os.path.join(save_path, 'PC_obj_{0}'.format(str(i).zfill(3)))
                np.save(obj_pc_path, coords_OBJ_list[i] / scale_list[i])

                if vis:
                    plt.figure()
                    plt.imshow(depth_list[i])
                    plt.show()
                    coord_cam_from_obj = coords_OBJ_list[i] @ R_list[i].T + t_list[i]
                    src = trimesh.PointCloud(coord_cam_from_obj, colors=[255,0,0,255])
                    tgt = trimesh.PointCloud(coords_CAM_list[i], colors=[0,255,0,255])
                    mesh = trimesh.load(obj_paths[i])
                    pc_obj = trimesh.PointCloud(coords_OBJ_list[i] / scale_list[i], colors=[0,0,255,255])
                    trimesh.Scene([src, tgt, mesh, pc_obj]).show()                

if __name__ == '__main__':
    main()