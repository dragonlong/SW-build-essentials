# solve the crashed packages
sudo dpkg --purge --force-all astah-community
dpkg --remove --force-remove-reinstreq webmin
sudo apt --fix-broken install
https://www.linuxquestions.org/questions/ubuntu-63/cannot-install-and-remove-package-847795/
