export http_proxy=http://uaserve.cc.vt.edu:8080
export https_proxy=http://uaserve.cc.vt.edu:8080
# about tensorboard
tensorboard --logdir="/work/cascades/lxiaol9/ARC/EAST/checkpoints/LSTM_east/"
# reasonable training with ConvLSTM
tensorboard --logdir="/home/lxiaol9/ARC/EASTRNN/checkpoints/LSTM"
# Good training with EAST
tensorboard --logdir="/work/cascades/lxiaol9/ARC/EAST/checkpoints/east/"
jupyter notebook --ip=$HOSTNAME --port=5556 --no-browser & >  jupyter.hostname
ssh -N -L localhost:5556:ca234:5556 lxiaol9@cascades1.arc.vt.edu
ssh -N -L localhost:6006:nr025:6006 lxiaol9@newriver1.arc.vt.edu
ssh -N -L localhost:5556:nr026:5556 lxiaol9@newriver1.arc.vt.edu
ssh xiaolong-simu -NfL localhost:50000:localhost:50000
User xiaolongli
scp -i ~/.ssh/google_compute_engine /usr/local/google/home/xiaolongli/Downloads/data/example.lua xiaolongli@35.238.86.75:~
scp -i ~/.ssh/google_compute_engine /usr/local/google/home/xiaolongli/Downloads/data/Motion\ Dataset\ v0.zip xiaolongli@35.238.86.75:~
gcloud compute scp /usr/local/google/home/xiaolongli/Downloads/data/example.lua xiaolong-simu:~
urdf-viz ~/Downloads/data/6DPOSE/shape2motion/urdf/laptop/0001/syn.urdf
urdf-viz ~/Downloads/data/6DPOSE/robot_movement_interface/dependencies/ur_description/urdf/ur10_robot.urdf
#
#
# Some bugs, like python visualization could be solved by
import matplotlib
matplotlib.use('Agg')
https://uechi.io/blog/x11forward
https://zngguvnf.org/2018-07-21--matplotlib-on-remote-machine.html

# jupyter notebook
source activate tf_cc
jupyter kernelspec list # we could change the json file manualy
conda install jupyter notebook
conda install -c conda-forge jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
# for caffe2
module load cuda/9.0.176 cudnn/7.1 Anaconda/5.1.0


pip install tqdm scikit-learn pandas pybind11 projectq future pytest requests scipy
pip install filterpy --upgrade
conda env create -f dlubu36.yml
pip install setuptools=39.1.0

# newriver/cascades
module purge
nvidia-smi
module load gcc cmake
module load cuda/9.0.176 cudnn/7.1
source activate dlubu36
export http_proxy=http://uaserve.cc.vt.edu:8080
export https_proxy=http://uaserve.cc.vt.edu:8080
jupyter notebook --ip=$HOSTNAME --port=5556 --no-browser & >  jupyter.hostname
cd videoText2018/Recurrent-EAST/project/
http://ca227:5556/?token=6cb25e829ccc9e1f5f3a3e938aa7e523dc678acc41f69f62
http://ca214:5556/?token=3cab58205e26170d3bb8512d72509105186cc5f0277934fb
# screen usage
screen -list
source deactivate
source activate tf_cc

scp -r ~/Dropbox/Code/Recurrent-EAST/ lxiaol9@newriver1.arc.vt.edu:~/videoText2018/
scp ~/Dropbox/Code/Recurrent-EAST/project/train_rnn_east_crop_tracking_2013.py lxiaol9@newriver1.arc.vt.edu:~/videoText2018/Recurrent-EAST/project
scp ~/Dropbox/Code/Recurrent-EAST/project/bayes/correlation.py lxiaol9@newriver1.arc.vt.edu:~/videoText2018/Recurrent-EAST/project/bayes/
scp ~/Dropbox/Code/Recurrent-EAST/project/lstm/model_rnn_east.py lxiaol9@newriver1.arc.vt.edu:~/videoText2018/Recurrent-EAST/project/lstm/

#check status
#1. Memory use
free -g
free -t
vmstat 1
#
#2. check job https://wikis.nyu.edu/display/NYUHPC/Copy+of+Tutorial+-+Submitting+a+job+using+qsub
# https://www.osc.edu/supercomputing/batch-processing-at-osc/monitoring-and-managing-your-job
# caffe2 pytroch tensorflow 1.7 is available inside anaconda on p100(newriver) and v100(cascades)
module load cuda/9.0.176 cudnn/7.1 Anaconda/5.1.0
qstat -u $USER
qdel id

####################### local ########################
sshfs lxiaol9@newriver1.arc.vt.edu:/work/cascades/lxiaol9/ ./NR_WORK/ -ocache=no -onolocalcaches -ovolname=ssh
sshfs lxiaol9@newriver1.arc.vt.edu:/home/lxiaol9/ ./NR_HOME/ -ocache=no -onolocalcaches -ovolname=ssh
diskutil unmount /Users/DragonX/Downloads/NR_HOME/
diskutil unmount /Users/DragonX/Downloads/NR_WORK/

# >>>>
