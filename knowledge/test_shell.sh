#!/bin/bash
# floating point
awk "BEGIN {print 10/3}"
echo "10/3.1" | bc -l
# integer arithmetic
echo "$((1 / 3))"
echo "$((1+5))"
x=5
y=10
ans=$(( x + y ))
echo "$x + $y = $ans"
# command with blank space, int without blank
PBS_ARRAYI=1
	# Print this sub-job's task ID
echo "My ARRAY_TASK_ID: " ${PBS_ARRAYI}
LR=$((1*10**${PBS_ARRAYI}))
echo $LR
array=("item 1" "item 2" "item 3")
for i in "${array[@]}"; do   # The quotes are necessary here
    echo "$i"
done
#https://ryanstutorials.net/bash-scripting-tutorial/bash-if-statements.php
if [ $1 -gt 100 ]
then
    echo Hey that\'s a large number.
    pwd
fi
date
