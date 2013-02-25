#!/bin/bash

cat > cartpole.condor <<EOF
+Group = "GRAD"
+Project = "AI_ROBOTICS"
+ProjectDescription = "cart-pole RL with experimental loss function"

universe = vanilla
notification = complete
requirements = (InMastodon == false) && (Memory >= 1024) && (Arch == "X86_64")
rank = KFlops
getenv = True
executable = cartpole-experiment.sh
log = cartpole-experiment.log
initialdir = $(pwd)
EOF

for m in 0 1 3
do
for n in 16 32 64 128 256
do
mkdir -p cartpole/$n
for g in linear sigmoid relu
do
for k in 64 256,64 256,256,64
do
cat >> cartpole.condor <<EOF
arguments = -ks $k -nonlin $g -n-samples $n -method $m -output cartpole/$n/cartpole-\$(Process)
output = cartpole/$n/experiment-n$n-m$m-g$g-k$k.\$(Process).out
error = cartpole/$n/experiment-n$n-m$m-g$g-k$k.\$(Process).err
Queue 5
EOF
done
done
done
done

#cat cartpole.condor
condor_submit cartpole.condor
