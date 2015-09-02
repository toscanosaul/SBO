END=$1
NumerSamplesF=$3
NumberTrainingPoints=$2
for i in $(seq 1 $END); do jsub "prog.nbs $i $NumberTrainingPoints $NumberSamplesF"; done
