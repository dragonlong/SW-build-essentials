ps -ax | g sshfs | awk '{print $1}' | xargs -I@ | echo @
fusermount -u YOUR_MNT_DIR
sudo umount -l YOUR_MNT_DIR
sudo diskutil umount Downloads/ARChome
umount -f ~/Downloads/ARChome
#
sshfs lxiaol9@newriver1.arc.vt.edu:/work/cascades/lxiaol9/ /home/dragon/ARCwork/ -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
sshfs lxiaol9@newriver1.arc.vt.edu:/home/lxiaol9/ /home/dragon/ARChome/ -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
sshfs lxiaol9@newriver1.arc.vt.edu:/work/cascades/lxiaol9/ /Users/dragonx/Downloads/ARCwork/ -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
sshfs lxiaol9@newriver1.arc.vt.edu:/home/lxiaol9/ /Users/dragonx/Downloads/ARChome/ -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
sshfs lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ /Users/dragonx/Downloads/groupARC/ -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3

sshfs xili@cv0.merl.com:/homes/xili/ cv0_home/ -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
sshfs xili@cv0.merl.com:/data2/datasets/tmp datasets/ -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
#
-o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
#https://serverfault.com/questions/6709/sshfs-mount-that-survives-disconnect
#https://forums.bunsenlabs.org/viewtopic.php?id=2808
pdsh -w ca[001..050] 'stat'
`stat`
awk 'print $1'
`` $()
"" is different from '' by specifying the
$
