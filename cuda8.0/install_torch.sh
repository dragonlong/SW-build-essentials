# in a terminal, run the commands WITHOUT sudo
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
# add source file
source ~/.bashrc
luarocks install image
luarocks install inn
luarocks install hdf5
sudo apt-get install libhdf5-serial-dev hdf5-tools
git clone https://github.com/deepmind/torch-hdf5
cd torch-hdf5
luarocks make hdf5-0-0.rockspec LIBHDF5_LIBDIR="/usr/lib/x86_64-linux-gnu/"
luarocks list

#################### Bug fix Log ####################
#-- Generating /home/dragonx/torch/install/lib/luarocks/rocks/hdf5/0-0/lua/hdf5/config.lua
#-- Installing: /home/dragonx/torch/install/lib/luarocks/rocks/hdf5/0-0/lua/hdf5/init.lua
#-- Installing: /home/dragonx/torch/install/lib/luarocks/rocks/hdf5/0-0/lua/hdf5/file.lua
#-- Installing: /home/dragonx/torch/install/lib/luarocks/rocks/hdf5/0-0/lua/hdf5/dataset.lua
#-- Installing: /home/dragonx/torch/install/lib/luarocks/rocks/hdf5/0-0/lua/hdf5/testUtils.lua
#luarocks/rocks/hdf5/0-0/lua/hdf5/group.lua
#Updating manifest for /home/dragonx/torch/install/lib/luarocks/rocks
#hdf5 0-0 is now built and installed in /home/dragonx/torch/install/ (license: BSD)
# SOLVE cuda VERSION problem, and check /home/dragonx/torch/install/share/lua/5.1/hdf5/config.lua
# lua package manager, design principle, config<path>
# https://github.com/soumith/cudnn.torch
# https://github.com/deepmind/torch-hdf5/issues/76
# https://github.com/deepmind/torch-hdf5/blob/master/doc/usage.md
# https://stackoverflow.com/questions/38137828/how-do-i-update-cudnn-to-a-newer-version?answertab=oldest#tab-top

# Demo: Reading from torch
# require 'hdf5'
# local myFile = hdf5.open('/path/to/read.h5', 'r')
# local data = myFile:read('/path/to/data'):all()
# myFile:close()
