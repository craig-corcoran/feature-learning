#!/bin/bash

lam=${1?Usage: cartpole.sh lambda gamma}
gam=${2?Usage: cartpole.sh lambda gamma}
pre=cartpole-lam$lam-gam$gam

cat > $pre.condor <<EOF
+Group = "GRAD"
+Project = "AI_ROBOTICS"
+ProjectDescription = "cart-pole RL with experimental loss function"

universe = vanilla
notification = complete
requirements = (Precise == true) && (InMastodon == false) && (Memory >= 2000) && (Arch == "X86_64")
rank = KFlops
getenv = True
executable = cartpole-experiment.sh
log = $pre.log
initialdir = $(pwd)
EOF

mkdir -p $pre

for m in 0 4 5 7
do
for g in linear sigmoid
do
for k in 16 64,16 #256,64,16
do
#arguments = -ks $k -nonlin $g -method $m -lam $lam -output $pre/cartpole-\$(Process)
cat >> $pre.condor <<EOF
arguments = -ks $k -nonlin $g -method $m -lam $lam -gam $gam -min-imp 0.01
output = $pre/experiment-m$m-g$g-k$k.\$(Process).out
error = $pre/experiment-m$m-g$g-k$k.\$(Process).err
Queue 31
EOF
done
done
done

#cat $pre.condor
condor_submit $pre.condor
