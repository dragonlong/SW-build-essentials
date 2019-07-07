"""
func: python code to translate json file into urdf format:
- traverse over path and objects
- parse from json file(dict, list, array);
- obj files path;
- inverse computing on rpy/xyz;
- xml writer to urdf
"""
import platform
import os
import os.path
import glob
import sys
import time
import random as rdn

import h5py
import yaml
import json
import copy
import collections

import numpy as np
import math
from transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix

from scipy import misc
from skimage import io
import cv2
# import matplotlib
# matplotlib.use('Agg')
import pywavefront
import matplotlib
import matplotlib.pyplot as plt
plt.ioff()

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, tostring, SubElement, Comment, ElementTree, XML
import xml.dom.minidom



def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def iterdict(d):
    for k,v in d.items():
        if k == 'children':
            if v is not None:
                for child in v:
                    iterdict(child)
        else:
            print (k,":",v)

def traverse_dict(d, link_dict, joint_dict):
    """
    link_dict  = {} # name - attributes;
    joint_list = [] # [{parent-child}, {}, {}]
    link: name + all attrs;
    joints: parent + child;
    """
    link            = {}
    joint           = {}
    for k, v in d.items():
        if k != 'children':
            link[k] = v
        else:
            if v is not None:
                for child in v:
                    traverse_dict(child, link_dict, joint_dict)
                    joint_dict[child['dof_name']] = d['dof_name']
    link_dict[d['dof_name']] = link

if __name__ == '__main__':

    is_debug  = True
    dataset   = 'sample'#'Motion\ Dataset\ v0/'
    obj_name = 'bike'
    index_obj = 1
    import platform
    if platform.uname()[0] == 'Darwin':
        print("Now it knows it's in my local Mac")
        base_path = '/Users/DragonX/Downloads/ARC/6DPOSE'
    elif platform.uname()[1] == 'viz1':
        base_path = '/home/xiaolong/Downloads/6DPOSE'
    elif platform.uname()[1] == 'vllab3':
        base_path = '/mnt/data/lxiaol9/rbo'
    elif platform.uname()[1] == 'xiaolongli.mtv.corp.google.com':
        base_path = '/usr/local/google/home/xiaolongli/Downloads/data/6DPOSE'
    elif platform.uname()[1] == 'xiaolong-simu':
        base_path = '/home/xiaolongli/data/6DPOSE'
    else:
        base_path = '/work/cascades/lxiaol9/6DPOSE/articulated_objects'

    dataset_root = base_path + '/' + dataset + '/objects' # with a lot of different objects, and then 0001/0002/0003/0004/0005
    all_objs     = os.listdir( dataset_root )
    print('all_objs are: ', all_objs)
    instances_per_obj = sorted(glob.glob(dataset_root + '/' + obj_name + '/*'))
    for sub_dir in instances_per_obj[0:1]: # regular expression
        json_name    = glob.glob(sub_dir + '/*.json')
        with open(json_name[0]) as json_file:
            motion_attrs = json.load(json_file)
            # print(json.dumps(motion_attrs, sort_keys=True, indent=4))
            # iterdict(motion_attrs)
            link_dict  = {} # name - attributes;
            joint_dict = {} # child- parent
            joint_list = [] #
            link_list  = [] #
            # dict is a better choice for this kind of iteration
            traverse_dict(motion_attrs, link_dict, joint_dict)
            for child, parent in joint_dict.items():
                joint = {}
                joint['parent'] = parent
                joint['child']  = child
                joint_list.append(joint)
            # for link, params_link in link_dict.items():
            keys_link = ['dof_rootd'] +  list(joint_dict.keys()) #
            #>>>>>>>>>>>>>>>>>> contruct links and joints
            root  = Element('robot', name="block")
            num   = len(joint_list) + 1
            links_name = ["base_link"] + [str(i+1) for i in range(num)]
            all_kinds_joints = ["revolute", "fixed", "prismatic", "continuous", "planar"]
            joints_name = []
            joints_type = []
            joints_pos  = []
            links_pos   = [None] * num
            joints_axis = []
            # parts connection
            for i in range(num-1):
                child_name  = joint_list[i]['child']
                index_parent= keys_link.index(joint_list[i]['parent'])
                index_child = keys_link.index(child_name)
                child_obj   = link_dict[ child_name ]
                vector_pos  = np.array(child_obj['center'])
                links_pos[index_child] = -vector_pos
                joints_name.append("{}_j_{}".format(index_parent, index_child))
                joints_axis.append(child_obj['direction'])
                if child_obj['motion_type'] == "rotation":
                    joints_type.append('revolute')
                else:
                    joints_type.append('prismatic')
                while joint_dict[child_name] !='dof_rootd':
                    print('joint {} now looking at child {}, has parent {}'.format(i, child_name, joint_dict[child_name]))
                    child_name  = joint_dict[ child_name ]
                    child_obj   = link_dict[ child_name ]
                    vector_pos  = vector_pos - np.array(child_obj['center'])
                joints_pos.append(vector_pos)
            # >>>>>>>>>>> start parsing urdf,
            children = [
                Element('link', name=links_name[i])
                for i in range(num)
                ]
            joints = [
                Element('joint', name=joints_name[i], type=joints_type[i])
                for i in range(num-1)
                ]
            # add inertial component
            node_inertial = XML('''<inertial><origin rpy="0 0 0" xyz="0 0 0"/><mass value="1.0"/><inertia ixx="0.9" ixy="0.9" ixz="0.9" iyy="0.9" iyz="0" izz="0.9"/></inertial>''')
            #>>>>>>>>>>>. 1. links
            for i in range(num):
                visual   = SubElement(children[i], 'visual')
                dof_name = link_dict[keys_link[i]]['dof_name']
                if dof_name == 'dof_rootd':
                    origin   = SubElement(visual, 'origin', rpy="0.0 0.0 0.0", xyz="0 0 0")
                else:
                    origin   = SubElement(visual, 'origin', rpy="0.0 0.0 0.0", xyz="{} {} {}".format(links_pos[i][0], links_pos[i][1], links_pos[i][2]))
                geometry = SubElement(visual, 'geometry')
                if i == 0 :
                    mesh     = SubElement(geometry, 'mesh', filename="{}/part_objs/none_motion.obj".format(sub_dir))
                else:
                    mesh     = SubElement(geometry, 'mesh', filename="{}/part_objs/{}.obj".format(sub_dir, dof_name))
                # materials assignment
                inertial = SubElement(children[i], 'inertial')
                node_inertial = XML('''<inertial><origin rpy="0 0 0" xyz="0 0 0"/><mass value="3.0"/><inertia ixx="0.9" ixy="0.9" ixz="0.9" iyy="0.9" iyz="0" izz="0.9"/></inertial>''')
                inertial.extend(node_inertial)
                if i == 0:
                    for mass in inertial.iter('mass'):
                        mass.set('value', "0.0")
                    for inertia in inertial.iter('inertia'):
                        inertia.set('ixx', "0.0")
                        inertia.set('ixy', "0.0")
                        inertia.set('ixz', "0.0")
                        inertia.set('iyy', "0.0")
                        inertia.set('iyz', "0.0")
                        inertia.set('izz', "0.0")
            #>>>>>>>>>>> 2. joints
            for i in range(num - 1):
                index_parent= keys_link.index(joint_list[i]['parent'])
                index_child = keys_link.index(joint_list[i]['child'])
                parent = SubElement(joints[i], "parent", link=links_name[index_parent])
                child  = SubElement(joints[i], "child",  link=links_name[index_child])
                origin = SubElement(joints[i], "origin", xyz="{} {} {}".format(joints_pos[i][0], joints_pos[i][1], joints_pos[i][2]), rpy="0 0 0")
                # we may need to change the joint name
                if joints_type[i]=='revolute':
                    axis = SubElement(joints[i], "axis", xyz="{} {} {}".format(joints_axis[i][0], joints_axis[i][1], joints_axis[i][2]))
                    limit= SubElement(joints[i], "limit", effort="1000.0", lower="-3.1415", upper="3.1415", velocity="0.5")
            #>>>>>>>>>>>> 3. construct the trees
            root.extend(children)
            root.extend(joints)
            xml_string = xml.dom.minidom.parseString(tostring(root))
            xml_pretty_str = xml_string.toprettyxml()
            print(xml_pretty_str)
            tree = ET.ElementTree(root)
            save_dir = dataset_root + '/urdf/{}/{:04d}'.format(obj_name, index_obj)
            # save
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(save_dir + '/syn.urdf', "w") as f:
                f.write(xml_pretty_str)
            #>>>>>>>>>>>>>>>>> coding
            # Create a copy
            for i in range(num):
                member_part = copy.deepcopy(root)
                # remove all visual nodes directly
                for link in member_part.findall('link'):
                    if link.attrib['name']!=links_name[i]:
                        for visual in link.findall('visual'):
                            link.remove(visual)
                xml_string = xml.dom.minidom.parseString(tostring(member_part))
                xml_pretty_str = xml_string.toprettyxml()
                tree = ET.ElementTree(member_part)
                with open(save_dir + '/syn_p{}.urdf'.format(i), "w") as f:
                    f.write(xml_pretty_str)

    # if is_debug:
    #     from mpl_toolkits.mplot3d import Axes3D
    #     #
    #     fig  = plt.figure(dpi=200)
    #     ax   = plt.subplot(111, projection='3d')
    #     cmap = plt.cm.prism
    #     colors = cmap(np.linspace(0., 1., num_parts+1))
    #     # ax.scatter(whole_cloud_urdf[:, 0], whole_cloud_urdf[:, 1], whole_cloud_urdf[:, 2], c='b', label='points_urdf{}'.format(i))
    #     for link in range(0, num_parts):
    #         ax.scatter(np.array(dict_mesh['v'][link])[:, 0], np.array(dict_mesh['v'][link])[:, 1], np.array(dict_mesh['v'][link])[:, 2], cmap=colors[link], label=name_mesh[link])
    #     ax.scatter(origin[:, 0], origin[:, 1], origin[:, 2], marker='^', s=500, cmap=colors[-1], label='origin')
    #     ax.set_xlabel('X Label')
    #     ax.set_ylabel('Y Label')
    #     ax.set_zlabel('Z Label')
    #     set_axes_equal(ax)
    #     plt.legend(loc=0)
    #     plt.title('3D points')
    #     plt.show()
    #     fig.savefig('./vis/norm{}.png'.format(index_obj), pad_inches=0)
