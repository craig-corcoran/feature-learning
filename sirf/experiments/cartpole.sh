#!/bin/bash

cat > cartpole.condor <<EOF
+Group = "GRAD"
+Project = "AI_ROBOTICS"
+ProjectDescription = "cart-pole RL with experimental loss function"

universe = vanilla
notification = complete
requirements = (InMastodon == false) && (Memory >= 2000) && (Arch == "X86_64")
rank = KFlops
getenv = True
executable = cartpole-experiment.sh
log = cartpole-experiment.log
initialdir = $(pwd)
EOF

mkdir -p cartpole

for m in 5 6 #0 1 2 3 4
do
for n in 32 64 128 256 512
do
for g in linear sigmoid
do
for k in 16 64,16 #256,64,16
do
cat >> cartpole.condor <<EOF
arguments = -ks $k -nonlin $g -n-samples $n -method $m -output cartpole/cartpole-\$(Process)
output = cartpole/experiment-n$n-m$m-g$g-k$k.\$(Process).out
error = cartpole/experiment-n$n-m$m-g$g-k$k.\$(Process).err
Queue 10
EOF
done
done
done
done

#cat cartpole.condor
condor_submit cartpole.condor
