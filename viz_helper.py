import os
import yaml
import numpy as np
import time
import h5py

from random import randint, sample
import matplotlib
import matplotlib.pyplot as plt  #
from voxelvis import PointVis

def plot_imgs(imgs, imgs_name, title_name='default', sub_name='default', save_path=None, save_fig=False, axis_off=False, grid_on=False, show_fig=True, dpi=150):
    fig     = plt.figure(dpi=dpi)
    cmap    = plt.cm.jet
    num = len(imgs)
    for m in range(num):
        ax1 = plt.subplot(1, num, m+1)
        plt.imshow(imgs[m].astype(np.uint8))
        if title_name is not None:
            plt.title(f'{title_name} ' + r'$_{{ {t2} }}$'.format(t2=imgs_name[m]))
        else:
            plt.title(imgs_name[m])
        if grid_on:
          plt.grid('on')
    if show_fig:
        plt.show()
    if save_fig:
        if save_path is None:
            if not os.path.exists('./visualizations/'):
                os.makedirs('./visualizations/')
            fig.savefig('./visualizations/{}_{}.png'.format(sub_name, title_name), pad_inches=0)
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig.savefig('{}/{}_{}.png'.format(save_path, sub_name, title_name), pad_inches=0)
    plt.close()

if __name__ == '__main__':
    # my_dir  = '/home/dragon/Dropbox/cvpr2021/viz'
    # filename='flow/5729_viz_data.npy'
    # filename = 'kpconv/08_0000081.npy'
    # my_dir    = '/home/dragon/Downloads/4DSeg/viz_mw'
    # my_dir  = '/home/dragon/Dropbox/cvpr2021/dataset'
    my_dir  = '/home/dragon/Documents/cvpr2021/kpcv16'
    filenames = os.listdir(my_dir)
    for i in range(500, 1000):
        filename = filenames[i]
        viz_file = f'{my_dir}/{filename}'
        print(f'Now checking {viz_file}')
        gt_data_handle  = np.load(viz_file, allow_pickle=True)
        gt_dict         = gt_data_handle.item()
        #
        for key, value in gt_dict.items():
            try:
                print(key, value.shape)
            except:
                print(key, value)
        vis = PointVis(target_pts=None, viz_dict=gt_dict)
        vis.run()
        
    # my_dir  = '/home/dragon/Dropbox/cvpr2021/viz'
    # filename='/test/flow/0_viz_feat.npy'
    #
    # viz_file = f'{my_dir}/{filename}'
    # gt_data_handle  = np.load(viz_file, allow_pickle=True)
    # gt_dict         = gt_data_handle.item()
    # #
    # for key, value in gt_dict.items():
    #     try:
    #         print(key, value.shape)
    #     except:
    #         print(key, value)
    # # further data
    # pixel_feat_map = np.linalg.norm(gt_dict['input'][0], axis=0)
    # # plot_imgs([pixel_feat_map], ['VFE feat'], title_name='input', sub_name=0, save_fig=False, show_fig=True)
    #
    #
    # filename='/test/0_viz_data.npy'
    # viz_file = f'{my_dir}/{filename}'
    # gt_data_handle  = np.load(viz_file, allow_pickle=True)
    # gt_dict         = gt_data_handle.item()
    # #
    # for key, value in gt_dict.items():
    #     try:
    #         print(key, value.shape)
    #     except:
    #         print(key, value)
    # # further data
    # pixel_cat_map = gt_dict['label'][0]
    # pixel_motion_map = gt_dict['motion'][0]
    # mask   = gt_dict['mask'][0, 0]
    # m_mask = gt_dict['m_mask'][0, 0] # (1, 384, 384)
    # input_point_seq  = gt_dict['input']
    # input_index_seq  = gt_dict['coord'] #(3, 150000, 2)
    # # npts = gt_dict['npts'] # (3,)
    #
    # vis = PointVis(target_pts=None, viz_dict=gt_dict)
    # vis.run()
    # for k in range(1):
    #     plot_imgs([pixel_feat_map, pixel_motion_map, mask, m_mask], [f'\tVFEfeat_{k}', f'\tgtmotion_{k}', f'\tmask_{k}', f'\tmotionmask_{k}'], title_name='input', sub_name=k, save_fig=False, show_fig=True)



    # my_dir  = '/Users/dragonx/Dropbox/cvpr2021/viz'
    # filename='0_viz_data.npy'
    #
    # viz_file = f'{my_dir}/{filename}'
    # gt_data_handle  = np.load(viz_file, allow_pickle=True)
    # gt_dict         = gt_data_handle.item()
    # #
    #
    # # further data
    # voxel_cat_map = gt_dict['label']
    # input_rv_map = gt_dict['input']
	# for k in range(16):
	# 	plot_imgs([voxel_cat_map[k, :, :], input_rv_map[k, :, :]], [f'\tgt_{k}', f'\trv_{k}'], title_name='input', sub_name=k, save_fig=False, show_fig=True)
