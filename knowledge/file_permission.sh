chgrp groupA ./folderA
cut -d: -f1 /etc/group | sort
setfacl -m u:USERNAME:rwx, g:USERNAME:r-x DIRECTORY
chmod g+rwx  ./folderA
echo $USER
