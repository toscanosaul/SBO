END=$1
BEGIN=$2
#NumerSamplesF=$3
#NumberTrainingPoints=$2
for i in $(seq $BEGIN $END); do jsub "prog3.nbs $i"; done
