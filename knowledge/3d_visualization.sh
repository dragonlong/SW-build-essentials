wget https://repo.continuum.io/archive/Anaconda3-5.3.1-Linux-x86_64.sh
# run the installer, the same procedure as local installation, say 'yes' for PATH appending
bash Anaconda3-5.3.1-Linux-x86_64.sh

# install in login node(take TF 1.12 for example, it works quite well from my test)
conda create -n pytf_cc python=3.6


source activate pytf_cc
pip install pybullet
pip install https://github.com/majimboo/py-mathutils/archive/2.78a.zip
pip install pycollada
pip install PyQt5
pip install mayavi
pip install h5py PyYaml
pip install cython scikit-image scikit-learn matplotlib bokeh ipython h5py PyYaml nose pandas jupyter

conda install -c anaconda scikit-image
conda install -c conda-forge jupyter_contrib_nbextensions

# jupyter contrib nbextension install --user
jupyter nbextension install --py mayavi --user
jupyter nbextension enable --py mayavi --user
conda install -c conda-forge ipywidgets
jupyter nbextension enable --py widgetsnbextension
conda install -c conda-forge ipyevents

python visualizer.py /mnt/data/lxiaol9/rbo/interactions/book/book02_o/ --rgb --d --js

cd /mnt/data/lxiaol9/rbo/objects/book/meshes/
cd /home/lxiaol9/Downloads/pybullet_robots/configuration_2017-07-30

cp book.urdf book.urdf_bak
sed -i 's|package://articulated_objects_db/data|/mnt/data/lxiaol9/rbo|g' book.urdf




world to camera
(0.5,                0.8660253882408142,    -0.0,                -0.0,
0.0,                 0.0,                   0.9999999403953552,  -0.0,
0.8660253286361694,  -0.4999999701976776,   -0.0,                -0.9999999403953552,
 0.0,                0.0,                   0.0,                 1.0)

projection
- 1.732050895690918 0
- 0.0              1.73
scikit-image
- 0.0                          -1  -0.02
- 0.0                          -1

- 0.0
- 1.732050895690918
- 0.0
- 0.0

- 0.0
- 0.0
- -1.0002000331878662
- -1.0

- 0.0
- 0.0
- -0.020002000033855438
- 0.0

http://www.songho.ca/opengl/gl_anglestoaxes.html
http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-17-quaternions/#so-which-one-should-i-choose-
http://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/#the-view-matrix
http://www.songho.ca/opengl/gl_transform.html
http://vispy.org/modern-gl.html
http://ksimek.github.io/2013/08/13/intrinsic/
On different rendering material characteristics
https://www.tomdalling.com/blog/modern-opengl/07-more-lighting-ambient-specular-attenuation-gamma/

# for reference usage
grep --include={*.env,*.sh,*.txt} -Rnl '/work/cascades/lxiaol9/SIMULIA' /work/cascades/lxiaol9/SIMULIA/Abaqus/2018/linux_a64/SMA/site/ | xargs -I@ sed -i 's|${MY_ABAQUS_DIR}|/opt/apps/abaqus/2018|g' @
grep --include={*.env,*.sh,*.txt} -Rnw '/work/cascades/lxiaol9/SIMULIA' /work/cascades/lxiaol9/SIMULIA/CAE/2018/linux_a64/SMA/site
ls -l $MYDIR | egrep '^d' | awk '{print $8}'
