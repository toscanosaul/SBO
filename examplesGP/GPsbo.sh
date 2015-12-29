END=$1
BEGIN=$2





for i in {0..4}; do
    let "var=2**$i"
    var=$( echo "1.0/$var" | bc -l)    
    for n in {1,2,4,8,16}; do
        for j in {2,4,8,16}; do
            A=$(echo "1.0/($j * $n)" | bc -l)
            for k in $(seq $BEGIN $END);
                do jsub "progGP.nbs $k $n 100 $var $A F" -mfail -email toscano.saul@gmail.com -xhost sys_pf -queue long; done
         done
    done
done


#for i in {0..4}; do
#    let "var=2**$i"
#    var=$( echo "1.0/$var" | bc -l)    
#    for n in {1,2,4,8,16}; do
#        for j in {2,4,8,16}; do
#            A=$(echo "1.0/($j * $n)" | bc -l)
#            for k in $(seq $BEGIN $END);
#                do jsub "progGPkg.nbs $k $n 100 $var $A F" -mfail -email toscano.saul@gmail.com -xhost sys_pf -queue long; done
#         done
#    done
#done
