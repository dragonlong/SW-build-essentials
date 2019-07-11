wget https://dl.google.com/linux/direct/chrome-remote-desktop_current_amd64.deb
sudo apt update
sudo dpkg --install chrome-remote-desktop_current_amd64.deb
sudo apt install --assume-yes --fix-broken
sudo DEBIAN_FRONTEND=noninteractive \
    apt install --assume-yes xfce4 desktop-base
echo "xfce4-session" > ~/.chrome-remote-desktop-session
sudo apt install --assume-yes xscreensaver
sudo apt install --assume-yes task-xfce-desktop
# sudo apt autoremove
# https://remotedesktop.google.com/headless
# install atom, terminator, firefox
