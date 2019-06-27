"""
params:
- number of links;
- link length/width;
- link color and diffusion (maybe)
- joint type of [0, 1, 2];
- articulation angle range;
- Using mesh models from other datasets;
  base-joint-link1-hinge-link2-hinge,
  create a random pool of different number of links, pose infos,
  length/width shape design;

# reference:
- https://pymotw.com/2/xml/etree/ElementTree/create.html
- https://stackoverflow.com/questions/3605680/creating-a-simple-xml-file-using-python
- https://stackabuse.com/reading-and-writing-xml-files-in-python/
"""
import random
from random import randint
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, tostring, SubElement, Comment,ElementTree, XML
import xml.dom.minidom
import copy
import os

#>>>>>>>>>>>>>--------robot class(controllable parameters)
class urdf_obj():
    def __init__(self, link_num, dim_base):
        self.link_num = link_num
        self.dim_base = dim_base
#>>>>>>>>>>>>---------XML oject------------
# ElementTree(tag + attrib(child)) Element, with name/text/properties

def generate_urdf(parts_num, save_ind, base_path, link_w=None):
    robot = urdf_obj(parts_num, [1, 1, 1])
    num   = robot.link_num
    root  = Element('robot', name="block")

    links_name = ["base_link"] + [str(i+1) for i in range(num)]
    links_h_raw  = np.random.rand(num)
    if link_w == None:
        links_w0       = 1
        links_w1       = [1]*parts_num
    else:
        links_w0       = link_w
        links_w1       = 0.5 + np.random.rand(parts_num)
    links_h_raw = links_h_raw / sum(links_h_raw) * 0.3
    links_h_raw[::-1].sort()
    links_h = links_h_raw
    # links_h = np.flip(links_h_raw)
    all_kinds_shape = ["box", "cylinder"]
    links_shape = [all_kinds_shape[randint(0, 0)] for i in range(num)]
    joints_name = ["{}_j_{}".format(i, i+1) for i in range(num-1)]
    all_kinds_joints = ["revolute", "fixed", "prismatic", "continuous", "planar"]
    joints_type = [all_kinds_joints[randint(0, 0)] for i in range(num-1)]
    #
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
    # add color components
    mat_blue      = SubElement(root, 'material',  name="blue")
    color_blue    = SubElement(mat_blue, "color", rgba="0 0 0.8 1")
    mat_black     = SubElement(root, 'material', name="black")
    color_blue    = SubElement(mat_black, "color", rgba="0 0 0 1")
    mat_white     = SubElement(root, 'material', name="white")
    color_white   = SubElement(mat_white, "color", rgba="1 1 1 1")
    material_lib  =['color_blue', 'color_white']
    colors_val  = ["1 1 0", "1 0 1", "0 1 1", "1 0 0", "0 1 0", "0 0 1"]
    colors_name = ["yellow", "magenta", "cyan", "red", "green", "blue"]
    for i in range(len(colors_val)):
        mat_any        = SubElement(root, 'material', name=colors_name[i])
        color_any      = SubElement(mat_any, "color", rgba="{} 1".format(colors_val[i]))
        material_lib.append(colors_name[i])
    random.shuffle(material_lib)
    #>>>>>>>>>>>>>>>>> links properties
    for i in range(num):
        if i == 0:
            visual   = SubElement(children[i], 'visual')
            origin   = SubElement(visual, 'origin', rpy="0.0 0 0", xyz="0 0 {}".format(0))
            geometry = SubElement(visual, 'geometry')
            if links_shape[i] == "cylinder":
                shape    = SubElement(geometry, 'cylinder', length=str(links_h[i]), radius=str(links_h[i] / 4))
            elif links_shape[i] == "box":
                shape    = SubElement(geometry, 'box', size="{} {} {}".format(links_w0, links_w1[i], links_h[i]))
            material = SubElement(visual, 'material', name=material_lib[i])
        else:
            visual = [Element('visual') for j in range(2)]
            # visual for link
            origin   = SubElement(visual[0], 'origin', rpy="0.0 0 0", xyz="0 {} {}".format(links_w1[i]/2, -links_h[i]/2)) #links_h[i]/2
            geometry = SubElement(visual[0], 'geometry')
            shape    = SubElement(geometry, 'box', size="{} {} {}".format(links_w0, links_w1[i], links_h[i]))
            material = SubElement(visual[0], 'material', name=material_lib[i])
            # visual for joint
            origin_joint    = SubElement(visual[1], 'origin', rpy="0.0 1.5707 0", xyz="0 0 0")
            geometry_joint  = SubElement(visual[1], 'geometry')
            shape_joint     = SubElement(geometry_joint, 'cylinder', length=str(links_w0), radius="{}".format(links_h[i]/3))
            material_joint  = SubElement(visual[1], 'material', name=material_lib[i])
            children[i].extend(visual)

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
    #>>>>>>>>>>>>>> joint features
    for i in range(num-1):
        parent = SubElement(joints[i], "parent", link=links_name[i])
        child  = SubElement(joints[i], "child",  link=links_name[i+1])
        if i==0:
            origin = SubElement(joints[i], "origin", xyz="0 {} {}".format(links_w1[i]/2, (links_h[i])/2 ), rpy="0 0 0")
        else:
            origin = SubElement(joints[i], "origin", xyz="0 {} {}".format(links_w1[i], 0), rpy="0 0 0")
        if joints_type[i] == "revolute":
            axis  = SubElement(joints[i], "axis", xyz="1 0 0")
            limit = SubElement(joints[i], "limit", effort="1000.0", lower="-3.14", upper="3.14", velocity="0.5")
    # extend from list with different names
    root.extend(children)
    root.extend(joints)
    xml_string = xml.dom.minidom.parseString(tostring(root))
    xml_pretty_str = xml_string.toprettyxml()
    # print(xml_pretty_str)
    tree = ET.ElementTree(root)
    save_dir = base_path + '/urdf/{:04d}'.format(save_ind)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + '/syn.urdf', "w") as f:
        f.write(xml_pretty_str)
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
    # >>>>>>> only for debug use <<<<<<<<<< #
    print("links_w0: ", links_w0)
    print("links_w1: ", links_w1)
    print("links_h: ", links_h)
    print("links_h_raw: ", links_h_raw)

if __name__ == "__main__":
    # type number of 100 different classes
    import platform
    if platform.uname()[0] == 'Darwin':
        print("Now it knows it's in my local Mac")
        base_path = '/Users/DragonX/Downloads/ARC/6DPOSE/synthetic'
    elif platform.uname()[1] == 'viz1':
        base_path = '/home/xiaolong/Downloads/6DPOSE/synthetic'
    elif platform.uname()[1] == 'vllab3':
        base_path = '/mnt/data/lxiaol9/rbo'
    elif platform.uname()[1] == 'xiaolongli.mtv.corp.google.com':
        base_path = '/usr/local/google/home/xiaolongli/Downloads/data/6DPOSE/synthetic'
    else:
        base_path = '/work/cascades/lxiaol9/6DPOSE/articulated_objects/synthetic'
    for i in range(100):
        parts_num = randint(2, 4)
        link_w_base = 0.3 * randint(1, 5)
        print("Generating {}th urdf file with {} parts>>>>>>>>>".format(i+1, parts_num))
        generate_urdf(parts_num, i, base_path, link_w_base)
