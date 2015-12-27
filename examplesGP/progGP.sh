END=$1
BEGIN=$2

SAMPLES=$3
ITERATIONS=$4

betah=$5
A=$6

CreateFunction=$7

for i in $(seq $BEGIN $END); do jsub "progSBO.nbs $i $SAMPLES $ITERATIONS $betah $A $CreateFunction " -mfail -email toscano.saul@gmail.com -xhost sys_pf -queue long; done
