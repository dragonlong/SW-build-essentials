# build summary:
# zip, molden, paraview, vasp-GPU, lammps, Abaqus-2018, NCCL, OpenFoam, Anaconda-5
git clone git@code.vt.edu:ARC/spec.git
#Module hierrarchy:
%define mpi_family_version openmpi1_8
<name><majorVersion>_<minorVersion>
rpmbuild paraview-5.4.1.spec 2>&1 | tee output4.txt
rpm -ivh vt-xxx.rpm
# how to build against all combinations of compilers and MPI stacks
rpmbuild -a foo.spec
# to our home directory for testing
rpm2cpio /opt/build/RPMS/x86_64/vt-xxxx.rpm | cpio -idv
rpm -qlp .rpm
# /path/to/folder/modulefiles
module avail
module load xxxx
module avail
module --ignore-cache avail
# installation
rpm -qa | grep xxxx
sudo rpm -e xxxx
sudo rpm -ivh xxx.rpm
du -a /home | sort -n -r | head -n 5

tar -xzvf -czvf -xzf -C(path)
find ./Abaqus  -type f -exec sed -i -e 's|/work/cascades/lxiaol9/SIMULIA|%{INSTALL_DIR}|g' {} \;
grep --include={*.env,*.sh,*.txt,*.xml,*.bindict,*.log} -Rnl '/work/cascades/lxiaol9/SIMULIA' $RPM_BUILD_ROOT/%{INSTALL_DIR}/ | xargs sed -i 's|/work/cascades/lxiaol9/SIMULIA|%{INSTALL_DIR}|g'
ln -sf "%{INSTALL_DIR}/CAE/2018/linux_a64/code/bin/ABQLauncher" "%{INSTALL_DIR}/Commands/abq2018"
ln -sf "%{INSTALL_DIR}/Commands/abq2018" "%{INSTALL_DIR}/Commands/abaqus"
##### About naming
paraview-5.4.1.spec
NAME:           ParaView
VERSION:        5.4.1
SOURCE0:        %{cname}-superbuild-%{version}.tar.gz
%setup -n %{cname}-superbuild-%{version}

cuda-9_2_148.spec
NAME:           cuda-9_2_148
VERSION:        9.2.148
SOURCE0:        %{cname}-%{version}.tar.gz
%setup -c %{cname}-%{version}

%build
%include inc/build
cd cuda-9.2.148 # why it couldn't open that, it should open that folder directly

/home/dragonx/miniconda3/bin:/home/dragonx/gems/bin:/usr/local/include/:/home/dragonx/bin:/usr/local/cuda-8.0/bin:/home/dragonx/gems/bin:/usr/local/include/:/home/dragonx/bin:/usr/local/cuda-8.0/bin:/home/dragonx/gems/bin:/usr/local/include/:/home/dragonx/bin:/usr/local/cuda-8.0/bin:/home/dragonx/bin:/home/dragonx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/dragonx/Documents/pycharm-edu-2017.3/bin

grant access to users
chmod -R ug+rw
# change the group of the resilio folder
$ chgrp -R rslsync resilio
# set full access to the group
$ chmod -R g+rwx resilio
sudo mount -o force /dev/sdc2 /media/dragonx/dragon/
sudo mount -o remount,rw,force /dev/sdc2
sudo mount -o remount, uid = 1000,gid=1000,rw /dev/sdc2
 sudo umount /dev/sdc2
sudo mount /dev/sdc2 /home/dragonx/Data
sudo mount -o rw,uid=1000,user,exec,umask=003 /dev/sdc2 /home/dragonx/Data
sudo mount -o remount,rw /dev/sdc2
sudo usermod --uid 1000 dragonx
sudo chown -R 1000:dragonx /home/dragonx /media/dragonx
