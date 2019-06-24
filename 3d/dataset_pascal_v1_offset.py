import matplotlib
# matplotlib.use('Agg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ioff()
from pylab import *

import platform
import os
import os.path
import sys
import time
import random as rdn

# import matplotlib
# matplotlib.use('Agg')

import h5py
import yaml
import json
import copy
import collections

import math
from transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from scipy import misc
from skimage import io
import cv2

import numpy as np
import numpy.ma as ma
from skimage import io # imread imsave imshow
from PIL import Image
import scipy.misc
import scipy.io as scio



import torch
import torch.utils.data as data
import torchvision.transforms as transforms


try:
        # for python newer than 2.7
    from collections import OrderedDict
except ImportError:
        # use backport from pypi
    from ordereddict import OrderedDict


# try to use LibYAML bindings if possible
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from yaml.representer import SafeRepresenter
_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

def dict_representer(dumper, data):
    return dumper.represent_dict(data.items())

def dict_constructor(loader, node):
    return OrderedDict(loader.construct_pairs(node))

Dumper.add_representer(OrderedDict, dict_representer)
Loader.add_constructor(_mapping_tag, dict_constructor)

Dumper.add_representer(str,
                       SafeRepresenter.represent_str)


from os.path import join as pjoin
class pascalVOCLoader(object):
    def __init__(
        self,
        root,
        sbd_path=None,
        split="train_aug",
        is_transform=False,
        img_size=512,
        augmentations=None,
        img_norm=True,
        test_mode=False):

        self.root = root
        self.sbd_path = sbd_path
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm  = img_norm
        self.test_mode = test_mode
        self.n_classes = 21
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        if not self.test_mode:
            for split in ["train", "val", "trainval"]:
                path = pjoin(self.root, "ImageSets/Segmentation", split + ".txt")
                file_list = tuple(open(path, "r"))
                file_list = [id_.rstrip() for id_ in file_list]
                self.files[split] = file_list

        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]
        im_path = pjoin(self.root, "JPEGImages", im_name + ".jpg")
        lbl_path = pjoin(self.root, "SegmentationClass/", im_name + ".png")
        im = Image.open(im_path)
        lbl = Image.open(lbl_path)
        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl

    def transform(self, img, lbl):
        if self.img_size == ("same", "same"):
            pass
        else:
            img = img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
            lbl = lbl.resize((self.img_size[0], self.img_size[1]))
        img = self.tf(img)
        lbl = torch.from_numpy(np.array(lbl)).long()
        lbl[lbl == 255] = 0
        return img, lbl


def read_model_view_matrices(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

def read_camera_viewpoint(filename):
    with open(filename, "r") as f:
        lines = f.readline()
        return lines.strip().split()

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)

def projPointOnLine3d(pt, orn_vect=None, base_pt=None):
    """
    Project a 3D point orthogonally onto a 3D line
    """
    # line representation:
    if base_pt is None:
        b = np.array([0, 0, 0.04]).astype(np.float32)
    else:
        b = base_pt
    if orn_vect is None:
        orn_vect = [0, 1, 0]
    #
    v_len     = np.dot(orn_vect, pt) - np.dot(orn_vect, base_pt)
    projPoint = base_pt + v_len * orn_vect

    return projPoint


def distancePointLine3d(pt, orn_vect=None, base_pt=None):
    """
    Euclidean distance between 3D point and line
    """
    projPt = projPointOnLine3d(pt, orn_vect, base_pt)
    dis_pts= np.norm(pt - projPt)

    return dis_pts, projPt

def fitLine3d():
    """
    Fit a 3D line to a set of points
    """
    return None

def generate_gt(pt, r_val=0.1, orn_vect=None, base_pt=None):
    """
    calculate the distance and unit vector from hinge to the point
    r_val is the threshold

    """
    dis_pts, projPt = distancePointLine3d(pt, orn_vect, base_pt)
    H = 1 - dis_pts/r_val
    U = (projPt - pt)/ dis_pts

    return H, U


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 512
img_length = 512

def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]# x, y, w, h
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 512:
        bbx[1] = 511
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 512:
        bbx[3] = 511
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 512:
        delt = rmax - 512
        rmax = 512
        rmin -= delt
    if cmax > 512:
        delt = cmax - 512
        cmax = 512
        cmin -= delt
    return rmin, rmax, cmin, cmax

def getBB_from_mask(mask):
    # here we assume it has single dominant connected shape
    x_set, y_set = np.where(mask>0)
    rmin = min(x_set)
    rmax = max(x_set)
    cmin = min(y_set)
    cmax = max(y_set)

    return rmin, rmax, cmin, cmax


DatasetInfo = collections.namedtuple(
    'DatasetInfo', ['basepath', 'train_size', 'test_size', 'frame_height', 'frame_width', 'num_clouds', 'num_samples', 'channels', 'sequence_size'])

PartsComps = collections.namedtuple('PartsComps', ['base', 'part1'])# now we have 2 parts, what if we have multiple of them --> list
TaskData   = collections.namedtuple('TaskData', ['img', 'cloud_cam', 'cloud_canon', 'model_points', 'choose',  'mask', 'num_parts', 'obj'])

class PoseDataset(data.Dataset):
    def __init__(self, mode, num, add_noise, root, noise_trans, refine):
        """
        num is the number of points chosen feeding into PointNet
        """
        self.objlist     = [0, 1, 2]
        self.objnamelist = ['laptop_challenge']#['laptop_near']# #, 'laptop_near'],
        self.mode        = mode

        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []

        self.list_status = []
        self.list_rank = []
        self.meta_dict = {}
        self.pt_dict   = {}
        self.root = root
        self.noise_trans = noise_trans
        self.refine = refine

        item_count = 0
        for item in self.objnamelist:
            base_path = self.root + '/' + item
            print(base_path)
            meta = {}
            pt   = {}
            # for art_index in os.listdir(base_path):
            for art_index in ['1', '2']:
                sub_dir0 = base_path + '/' + art_index
                if self.mode == 'train':
                    input_file = open(sub_dir0 + '/train.txt')
                else:
                    input_file = open(sub_dir0 + '/test.txt')
                while 1:
                    item_count += 1
                    input_line = input_file.readline()
                    if self.mode == 'test' and item_count % 10 != 0:
                        continue
                    if not input_line:
                        break
                    if input_line[-1:] == '\n':
                        input_line = input_line[:-1]
                    self.list_rgb.append(sub_dir0 + '/rgb/{}.png'.format(input_line))
                    self.list_depth.append(sub_dir0 + '/depth/{}.h5'.format(input_line))
                    if self.mode == 'eval':
                        self.list_label.append(sub_dir0 + '/mask/{}.png'.format(input_line))
                    else:
                        self.list_label.append(sub_dir0 + '/mask/{}.png'.format(input_line))

                    self.list_obj.append(item)
                    self.list_status.append(art_index)
                    self.list_rank.append(int(input_line))

                meta_file  = open(sub_dir0 + '/gt.yml', 'r')
                meta_instance = yaml.load(meta_file)
                meta[art_index] = meta_instance
            # os.listdir(self.root + '/models/' + item)
            for obj_model in ['laptop_top', 'laptop_bottom']:
                pt[obj_model] = ply_vtx(self.root + '/models/' + item + '/' + obj_model + '1.ply')
            self.pt_dict[item]   = pt
            self.meta_dict[item] = meta
            print("Object {0} buffer loaded".format(item))

        self.length = len(self.list_rgb)

        self.cam_cx = 325.26110
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140
        self.cam_fy = 573.57043
        self.height = 512
        self.width  = 512

        self.xmap = np.array([[j for i in range(512)] for j in range(512)])
        self.ymap = np.array([[i for i in range(512)] for j in range(512)])

        self.num = num
        self.add_noise = add_noise
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = 500
        self.num_pt_mesh_small = 500
        self.symmetry_obj_idx = [7, 8]

        # render
        self.render_root = '/work/cascades/lxiaol9/6DPOSE/partnet/VOCdevkit/VOC2012'
        self.renderSource = pascalVOCLoader(root=self.render_root, split = 'trainval', is_transform=False, augmentations=None)

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small

    def __getitem__(self, index):

        item       = self.list_obj[index]
        obj        = 1#self.objnamelist.index(item)
        art_status = self.list_status[index]
        frame_order= self.list_rank[index]
        label      = self.list_label[index]

        # model points
        model_points = self.pt_dict[item]
        part_key     = list(model_points.keys())
        num_parts    = len(part_key)
        model_offsets      = [None]*num_parts
        model_offsets[0]   = np.array([0, 0, -0.035])
        model_offsets[1]   = np.array([0, 0, 0])

        # top to bottom
        parts_model_point  = [None]*num_parts
        parts_world_point  = [None]*num_parts
        parts_target_point = [None]*num_parts

        parts_cloud_cam    = [None]*num_parts
        parts_cloud_world  = [None]*num_parts
        parts_cloud_canon  = [None]*num_parts

        parts_world_pos    = [None]*num_parts
        parts_world_orn    = [None]*num_parts
        parts_model2world  = [None]*num_parts
        parts_target_r     = [None]*num_parts
        parts_target_t     = [None]*num_parts

        parts_mask         = [None]*num_parts
        choose_x           = [None]*num_parts
        choose_y           = [None]*num_parts
        choose_to_whole    = [None]*num_parts

        # ---------------> rgb/depth/label
        # print('current image: ', self.list_rgb[index])
        img = Image.open(self.list_rgb[index])
        if self.add_noise:
            img = self.trancolor(img)
        img=np.array(img)#.astype(np.uint8)
        depth = np.array(h5py.File(self.list_depth[index], 'r')['data'])
        label = np.array(Image.open(self.list_label[index]))

        # ---------------> pose infos
        pose_dict = self.meta_dict[item][art_status]['frame_{}'.format(frame_order)]
        viewMat   = np.array(pose_dict['viewMat']).reshape(4, 4).T
        projMat   = np.array(pose_dict['projMat']).reshape(4, 4).T

        # top:  ['obj'][link][state_index]
        linkcenter_world_pos = np.array(pose_dict['obj'][1][0]).astype(np.float32)
        linkcenter_world_orn = np.array(pose_dict['obj'][1][1]).astype(np.float32)
        ############## for debug use ############
        # linkinertial_pos     = np.array(pose_dict['obj'][0][2]).astype(np.float32)
        # linkinertial_orn     = np.array(pose_dict['obj'][0][3]).astype(np.float32)
        # linkframe_pos        = np.array(pose_dict['obj'][0][4]).astype(np.float32)
        # linkframe_orn        = np.array(pose_dict['obj'][0][5]).astype(np.float32)
        # print_list     = [art_status, linkframe_pos, linkframe_orn]
        # variables_list = ['art_status', 'linkframe_pos', 'linkframe_orn']
        # print(index)
        # for i, x in enumerate(print_list):
        #     print(variables_list[i], ': \n', x)
        # linkframe_pos        = np.array(pose_dict['obj'][1][4]).astype(np.float32)
        # linkframe_orn        = np.array(pose_dict['obj'][1][5]).astype(np.float32)
        # print_list     = [art_status, linkframe_pos, linkframe_orn]
        # variables_list = ['art_status', 'linkframe_pos', 'linkframe_orn']
        # for i, x in enumerate(print_list):
        #     print(variables_list[i], ': \n', x)
        ############## debug end here ############
        # bottom: model coordinates, the quaternion in Pybullet is [x, y, z, w]
        # refer to laptop1_b0.urdf in the densecoord folder
        base_world_pos = np.array([0, 0, 0])
        base_world_orn = np.array([0, 0, 0, 1])
        #
        parts_world_pos[0] = linkcenter_world_pos
        parts_world_orn[0] = linkcenter_world_orn
        parts_world_pos[1] = base_world_pos
        parts_world_orn[1] = base_world_orn
        #
        # fetch model points
        for t in range(num_parts):
            dellist = [j for j in range(0, len(model_points[part_key[t]]))]
            dellist = rdn.sample(dellist, len(model_points[part_key[t]]) - self.num_pt_mesh_small)
            model_points[part_key[t]] = np.delete(model_points[part_key[t]], dellist, axis=0)

        # target rotation, translation, and target points
        for k in range(num_parts):
            # matrix computation
            center_world_orn   = parts_world_orn[k]
            center_world_orn   = np.array([center_world_orn[3], center_world_orn[0], center_world_orn[1], center_world_orn[2]])
            my_model2world_r   = quaternion_matrix(center_world_orn)[:4, :4] # [w, x, y, z]
            my_model2world_t   = parts_world_pos[k]
            my_model2world_mat = my_model2world_r
            for m in range(3):
                my_model2world_mat[m, 3] = my_model2world_t[m]
            my_world2camera_mat   = viewMat
            my_camera2clip_mat    = projMat
            my_model2camera_mat   = np.dot(my_world2camera_mat, my_model2world_mat)
            # points transformation
            my_pcloud             = np.array(model_points[part_key[k]]) + model_offsets[k]
            pcloud_target         = np.concatenate((my_pcloud, np.ones((my_pcloud.shape[0], 1))), axis=1)
            pcloud_target_world   = np.dot(pcloud_target, my_model2world_mat.T)
            parts_world_point[k]  = pcloud_target_world[:, :3]
            pcloud_target         = np.dot(pcloud_target, my_model2camera_mat.T)

            # [w, x, y, z] for model training
            Rq_full               = np.diag([0, 0, 0, 1])
            Rq_full[:3, :3]       = my_model2camera_mat[:3, :3]
            parts_target_r[k]     = quaternion_from_matrix(Rq_full, True)
            parts_target_t[k]     = my_model2camera_mat[:3, 3]
            parts_model2world[k]  = my_model2world_mat
            parts_model_point[k]  = my_pcloud
            parts_target_point[k] = pcloud_target[:, :3]


        # ---------------> depth to cloud data
        mask = np.array((label[:, :]<3) & (label[:, :]>0)).astype(np.uint8)
        mask_whole = np.copy(mask)
        for n in range(num_parts):
            parts_mask[n] = np.array((label[:, :]==(n+1))).astype(np.uint8)
            choose_to_whole[n] = np.where(parts_mask[n]>0)
        ########################## deciding the real input ####################
        rmin, rmax, cmin, cmax = getBB_from_mask(mask) # relative to thefull image
        rmin, rmax, cmin, cmax = get_bbox([cmin, rmin, cmax-cmin,  rmax-rmin])
        #
        if (rmin - 20 >-1) and (rmax + 20 < 512):
            rmin = rmin - 20
            rmax = rmax + 20
        elif (rmin - 20 < 0):
            rmin = 0
            rmax = rmax + 40 - rmin
        elif rmax + 20 > 512:
            rmax = 512
            rmin = rmin - 40 + (512 - rmax)

        if (cmin - 20 > -1) and (cmax + 20 < 512):
            cmin = cmin - 20
            cmax = cmax + 20
        elif (cmin - 20 < 0):
            cmin = 0
            cmax = cmax + 40 - cmin
        elif cmax + 20 > 512:
            cmax = 512
            cmin = cmin - 40 + (512 - cmax)

        h_cropped = rmax - rmin
        w_cropped = cmax - cmin
        img_masked = img[rmin:rmax, cmin:cmax, :]
        new_mask   = label[rmin:rmax, cmin:cmax]
        #
        noise_level  = self.noise_trans
        sourceImg    = np.copy(img_masked)
        #----------------> augment rgb/depth here using renderImg
        index_pascal  = np.random.randint(0, 2912)
        img_pascal, _ = self.renderSource.__getitem__(index_pascal)

        sourceMask   = np.copy(mask_whole)[rmin:rmax, cmin:cmax]
        renderImg    = np.array(img_pascal)
        # print('img_pascal has shape: \n', renderImg.shape)

        renderMask        = 1 - sourceMask
        height_r, width_r = renderImg.shape[0:2]
        height_s, width_s = h_cropped, w_cropped
        random_noise      = np.random.rand(height_s, width_s)
        random_occu_max   = 50
        random_occu_flag  = True
        sourceImg         = sourceImg * (1.0 - noise_level) + random_noise[:,:,np.newaxis] * 255 * noise_level
        if (height_r>=height_s) and (width_r>=width_s):
            start_row         = rdn.randint(0, height_r - height_s)
            start_col         = rdn.randint(0, width_r -width_s)
            renderImg_crop    = np.copy(renderImg)[start_row:start_row+height_s, start_col:start_col+width_s, :]
            renderImg_masked  = renderImg_crop * renderMask[:, :, np.newaxis]
            targetImg         = np.copy(img_masked) * sourceMask[:, :, np.newaxis]
            targetImg         = targetImg + renderImg_masked
            # figure = plt.figure(dpi=150)
            # ax = plt.subplot(141)
            # plt.imshow(sourceImg.astype(np.uint8))
            # plt.title('source image')
            # ax1 = plt.subplot(142)
            # plt.imshow(renderImg_masked)
            # plt.title('render image')
            # ax2 = plt.subplot(143)
            # plt.imshow(sourceMask)
            # plt.title('mask4source')
            # ax3 = plt.subplot(144)
            # plt.imshow(renderMask)
            # plt.title('mask4render')
            # figure.savefig('./vis/test{}.png'.format(index), pad_inches=0)
            # fig2 = plt.figure(dpi=300)
            # ax = plt.subplot(121)
            # plt.imshow(targetImg.astype(np.uint8))
            # plt.title('cropped input image')
            # ax1 = plt.subplot(122)
            # plt.imshow(depth[rmin:rmax, cmin:cmax])
            # plt.title('cropped depth image')
            # plt.show()
            # fig2.savefig('./vis/target{}.png'.format(index), pad_inches=0)
            targetImg         = np.transpose(targetImg,  (2, 0, 1))
        else:
            targetImg         = np.transpose(img_masked, (2, 0, 1))
        #>>>>>>>>>>------- rendering target pcloud from depth image --------<<<<<<<<<#
        # first get projected map
        ymap = self.ymap
        xmap = self.xmap
        h = self.height
        w = self.width
        u_map     = ymap * 2 / w -1
        v_map     = (512 - xmap) * 2 / h -1
        w_channel = depth
        projected_map = np.stack([u_map * w_channel, v_map * w_channel, -depth, w_channel]).transpose([1, 2, 0])
        # -------------------> randomly choose pixels from parts mask regions--> choose_x, choose_y
        # mask on parts and object level
        for s in range(num_parts):
            x_set, y_set   = choose_to_whole[s]
            all_points_ind = list(range(len(x_set)))
            rdn.shuffle(all_points_ind)
            if len(x_set) > self.num/num_parts:
                choose_x[s] = x_set[ all_points_ind[0: int(self.num/num_parts)] ]
                choose_y[s] = y_set[ all_points_ind[0: int(self.num/num_parts)] ]
            elif len(x_set)==0:
                print('dataset has blank space!!!!!!!!!!!!!!!')
                cc = torch.LongTensor([0])
                # return(cc, cc, cc, cc, cc, cc, cc, cc)
                return None
            else:
                print('mask points are not enough!!!!!!!!!!!!!!!')
                choose_x[s] = np.ones((int(self.num/num_parts))) * x_set[all_points_ind[0]]
                choose_y[s] = np.ones((int(self.num/num_parts))) * y_set[all_points_ind[0]]
                choose_x[s][0:len(x_set)] = x_set
                choose_y[s][0:len(x_set)] = y_set
                cc = torch.LongTensor([0])
                # return(cc, cc, cc, cc, cc, cc, cc, cc)
                return None
            # ---------------> from projected map into target part_cloud(cam, world, canon)
            # here we're intereseted in computing the cloud separately for each part
            projected_points = projected_map[choose_x[s][:].astype(np.uint16), choose_y[s][:].astype(np.uint16), :]
            projected_points = np.reshape(projected_points, [-1, 4])
            depth_channel    = - projected_points[:, 3:4]
            cloud_cam      = np.dot(projected_points[:, 0:2], np.linalg.pinv(my_camera2clip_mat[:2, :2].T))
            # print('point_cloud_cam_0:\n', cloud_cam)
            cloud_cam      = np.concatenate((cloud_cam, depth_channel), axis=1)
            cloud_cam_full = np.concatenate((cloud_cam, np.ones((cloud_cam.shape[0], 1))), axis=1)
            cloud_world    = np.dot(cloud_cam_full, np.linalg.pinv(viewMat.T))
            cloud_canon    = np.dot(cloud_world, np.linalg.pinv(parts_model2world[s].T))
            # canon points should be points coordinates centered in the inertial frame

            parts_cloud_cam[s]    = cloud_cam
            parts_cloud_world[s]  = cloud_world[:, :3]
            parts_cloud_canon[s]  = cloud_canon[:, :3]

        # for t in range(num_parts - 1):
        #     parts_cloud_canon[t] = parts_cloud_canon[t] - np.array([0, 0, 0.04])
        # >>>>>>>>>>>>>> add offset computation to hinge >>>>>>>>>>>>>> #
        # by default, we sample 1024 points, which is better for point computation
        # 3. OBB via PCA

        # 1. normalized points input & normalized output
        # 2. PCA computation for surface normals

        # hinge joint on [0, 1, 0], is that correct? we should know joint from URDF
        # point to line distance,
        # K nearest neighbor by ranking the distance
        # decompose into H + U (finally the point could make up the line)
        #

        x_set_pcloud = np.concatenate(choose_x, axis=0)
        y_set_pcloud = np.concatenate(choose_y, axis=0)
        # transformation on point cloud on parts and object level
        # 250 for top, 250 for bottom
        choose = (x_set_pcloud - rmin) * w_cropped + (y_set_pcloud - cmin)
        add_t = np.array([rdn.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])
        ################# for debug only #################
        from mpl_toolkits.mplot3d import Axes3D
        fig1 = plt.figure(dpi=150)
        ax = plt.subplot(111, projection='3d')
        # ax.scatter(parts_model_point[0][:, 0], parts_model_point[0][:, 1], parts_model_point[0][:, 2], c='b', label='top model points')
        # ax.scatter(parts_model_point[1][:, 0], parts_model_point[1][:, 1], parts_model_point[1][:, 2], c='r', label='bottom model points')
        # ax.scatter(parts_cloud_canon[0][:, 0], parts_cloud_canon[0][:, 1], parts_cloud_canon[0][:, 2], c='b', label='top canon points')
        # ax.scatter(parts_cloud_canon[1][:, 0], parts_cloud_canon[1][:, 1], parts_cloud_canon[1][:, 2], c='r', label='bottom canon points')
        # ax.scatter(parts_cloud_cam[0][:, 0], parts_cloud_cam[0][:, 1], parts_cloud_cam[0][:, 2], c='b', label='top camera points')
        # ax.scatter(parts_cloud_cam[1][:, 0], parts_cloud_cam[1][:, 1], parts_cloud_cam[1][:, 2], c='r', label='bottom camera points')
        ax.scatter(parts_cloud_world[0][:, 0], parts_cloud_world[0][:, 1], parts_cloud_world[0][:, 2], c='b', label='top world points')
        ax.scatter(parts_cloud_world[1][:, 0], parts_cloud_world[1][:, 1], parts_cloud_world[1][:, 2], c='r', label='bottom world points')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        # ax1 =  plt.subplot(122, projection='3d')
        # ax1.scatter(parts_pcloud_urdf[0][0][:, 0], parts_pcloud_urdf[0][0][:, 1], parts_pcloud_urdf[0][0][:, 2], c='b', label='bottom urdf points_canon')
        # ax1.scatter(parts_pcloud_urdf[0][1][:, 0], parts_pcloud_urdf[0][1][:, 1], parts_pcloud_urdf[0][1][:, 2], c='b', label='top urdf points_canon')
        # ax1.scatter(parts_pcloud_urdf[1][0][:, 0], parts_pcloud_urdf[1][0][:, 1], parts_pcloud_urdf[1][0][:, 2], c='r', label='bottom urdf points_rbo')
        # ax1.scatter(parts_pcloud_urdf[1][1][:, 0], parts_pcloud_urdf[1][1][:, 1], parts_pcloud_urdf[1][1][:, 2], c='r', label='top urdf points_rbo')
        # ax1.set_xlabel('X Label')
        # ax1.set_ylabel('Y Label')
        # ax1.set_zlabel('Z Label')
        plt.legend(loc=1)
        plt.title('3D points')
        plt.show()
        plt.pause(15)
        if plt.waitforbuttonpress():
            exit
        import keyboard  # using module keyboard
        while True:  # making a loop
            try:  # used try so that if user pressed other than the given key error will not be shown
                if keyboard.is_pressed('q'):  # if key 'q' is pressed
                    print('You Pressed A Key!')
                    break  # finishing the loop
                else:
                    pass
            except:
                break  # if us
        ################ DEBUG ENDS HERE #################
        # the final transformation into Torch format
        for ind in range(num_parts):
            if self.add_noise:
                parts_cloud_cam[ind] = np.add(parts_cloud_cam[ind], add_t)
                parts_cloud_canon[ind]= np.add(parts_cloud_canon[ind], add_t)
            parts_mask[ind]        =torch.from_numpy(parts_mask[ind][rmin:rmax, cmin:cmax].astype(np.float32))
            parts_model_point[ind] =torch.from_numpy(parts_model_point[ind].astype(np.float32))
            # parts_world_point[ind] =torch.from_numpy(parts_world_point[ind].astype(np.float32))
            # parts_target_point[ind]=torch.from_numpy(parts_target_point[ind].astype(np.float32))
        cloud_cam_whole   = torch.from_numpy(np.concatenate(parts_cloud_cam, axis=0).astype(np.float32))
        # cloud_world_whole = torch.from_numpy(np.concatenate(parts_cloud_world, axis=0).astype(np.float32))
        cloud_canon_whole = torch.from_numpy(np.concatenate(parts_cloud_canon, axis=0).astype(np.float32))
        # ['img', 'cloud_canon', 'model_points', 'choose',  'mask', 'num_parts', 'obj']
        input_data = TaskData(
            img=self.norm(torch.from_numpy(targetImg.astype(np.float32))), #
            cloud_cam=cloud_cam_whole,
            cloud_canon=cloud_canon_whole,
            model_points=parts_model_point,
            choose=torch.LongTensor(choose.reshape([1, self.num]).astype(np.int32)),
            mask=parts_mask,
            num_parts=torch.LongTensor([num_parts]),
            obj=torch.LongTensor([obj]))
        return input_data
        # input_data = TaskData_debug(
        #     img=img,
        #     img_masked=img_masked,
        #     depth=depth,
        #     num_parts=num_parts,
        #     model_points=parts_model_point,
        #     world_points=parts_world_point,
        #     target_points=parts_target_point,
        #     mask=parts_mask,
        #     mask_w=mask,
        #     model2world=parts_model2world,
        #     world2cam=viewMat,
        #     cam2clip=projMat,
        #     choose=choose,
        #     boundary=[rmin, rmax, cmin, cmax],
        #     cloud_cam=cloud_cam_whole,
        #     cloud_world=cloud_world_whole
        #     cloud_canon=cloud_canon_whole,
        #     obj=obj)
if __name__ == '__main__':
    dataset_root ='/work/cascades/lxiaol9/6DPOSE/partnet/pose_articulated'
    PoseData = PoseDataset('test', 500, True, dataset_root, 0.03, False)
    wrong_data = []
    valid_data = []
    for i in range(2, len(PoseData.list_rgb)):
        # print('Getting {}th data point'.format(i))
        data = PoseData.__getitem__(i)
        if data != None:
            valid_data.append(i)
            img               =  data.img
            parts_mask        =  data.mask
            num_parts         =  data.num_parts
            parts_model_point =  data.model_points
            point_cloud_canon =  data.cloud_canon
            obj               =  data.obj
            choose            =  data.choose.view(-1)
            if(min(choose)<0 or max(choose)>img.shape[1]*img.shape[2]-1):
                print(i, img.shape[1], img.shape[2], min(choose), max(choose))
                wrong_data.append(i)
    print('wrong_data:\n', wrong_data)
    print('valid_data:\n', len(valid_data))

    # mask_w            =  data.mask_w
    # img_masked        =  data.img_masked.transpose([1, 2, 0])
    # depth             =  data.depth
    # num_parts         =  data.num_parts
    # parts_model_point =  data.model_points
    # parts_world_point =  data.world_points
    # parts_target_point=  data.target_points
    # parts_mask        =  data.mask
    # mask_w            =  data.mask_w
    # parts_model2world =  data.model2world
    # viewMat           =  data.world2cam
    # projMat           =  data.cam2clip
    # choose            =  data.choose
    # boundary          =  data.boundary
    # point_cloud_cam   =  data.cloud_cam
    # point_cloud_world =  data.cloud_world
    # point_cloud_canon =  data.cloud_canon
    # obj               =  data.obj
    #
    # # part_key = list(model_points.keys())
    # part0 = parts_model_point[0]
    # part1 = parts_model_point[1]
    # print('rgb image: \n ',  img.shape)
    # print('masked rgb image: \n ',  img_masked.shape)
    # print('depth: \n', depth.shape)
    # print('label: \n', mask_w.shape)
    #
    # print('num_parts:\n', num_parts)
    # print('choose:\n', choose.shape)
    # print('boundary:\n', boundary)
    #
    # print('model_points 0: \n', parts_model_point[0].shape)
    # print('world_points 0: \n', parts_world_point[0].shape)
    # print('target_points 0:\n', parts_target_point[0].shape)
    # print('model_points 1: \n', parts_model_point[1].shape)
    # print('world_points 1: \n', parts_world_point[1].shape)
    # print('target_points 1:\n', parts_target_point[1].shape)
    # print('objlist.index: \n ', obj)
    #
    # # model points
    # print('part0: \n', part0.shape)
    # print('part1: \n', part1.shape)
    # #
    # print('parts_model2world[0]: \n', parts_model2world[0])
    # print('parts_model2world[1]: \n', parts_model2world[1])
    # print('viewMat: \n', viewMat)
    # print('projMat: \n', projMat)
    # #
    # print('point cloud_cam: \n',   point_cloud_cam[0].shape)
    # print('point_cloud_world: \n', point_cloud_world[0].shape)
    # print('point_cloud_canon: \n', point_cloud_canon[0].shape)
    # %matplotlib inline
    # figure = plt.figure(dpi=300)
    # ax = plt.subplot(131)
    # plt.imshow(img)
    # plt.title('RGB image')
    # ax1 = plt.subplot(132)
    # plt.imshow(mask[0])
    # plt.title('mask for top')
    # ax2 = plt.subplot(133)
    # plt.imshow(mask[1])
    # plt.title('mask for bottom')
    # plt.show()
