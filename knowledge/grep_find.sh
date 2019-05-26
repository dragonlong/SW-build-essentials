grep --include={*.env,*.sh,*.txt} -Rnl '/work/cascades/lxiaol9/SIMULIA' /work/cascades/lxiaol9/SIMULIA/Abaqus/2018/linux_a64/SMA/site/ | xargs -I@ sed -i 's|${MY_ABAQUS_DIR}|/opt/apps/abaqus/2018|g' @
grep --include={*.env,*.sh,*.txt} -Rnw '/work/cascades/lxiaol9/SIMULIA' /work/cascades/lxiaol9/SIMULIA/CAE/2018/linux_a64/SMA/site
