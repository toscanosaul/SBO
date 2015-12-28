END=$1
BEGIN=$2



for i in {0..10}; do
    let "var=2**$i"
    var=$( echo "1.0/$var" | bc -l)    
    echo $var
    for n in {1,2,4,8,16}; do
        echo $n
        for j in {1..10}; do
            A=$(echo "1.0/($j * $n)" | bc -l)
            echo $A
            jsub "progGP.nbs 1 $n 1 $var $A T" -mfail -email toscano.saul@gmail.com -xhost sys_pf -queue long; 
         done
    done
done


for i in {0..10}; do
    let "var=2**$i"
    var=$( echo "1.0/$var" | bc -l)    
    for n in {1,2,4,8,16}; do
        for j in {1..10}; do
            A=$(echo "1.0/($j * $n)" | bc -l)
            for k in $(seq $BEGIN $END);
                do jsub "progGP.nbs $k $n 100 $var $A F" -mfail -email toscano.saul@gmail.com -xhost sys_pf -queue long; done
         done
    done
done


for i in {0..10}; do
    let "var=2**$i"
    var=$( echo "1.0/$var" | bc -l)    
    for n in {1,2,4,8,16}; do
        for j in {1..10}; do
            A=$(echo "1.0/($j * $n)" | bc -l)
            for k in $(seq $BEGIN $END);
                do jsub "progGPkg.nbs $k $n 100 $var $A F" -mfail -email toscano.saul@gmail.com -xhost sys_pf -queue long; done
         done
    done
done
