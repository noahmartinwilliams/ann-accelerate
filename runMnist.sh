#! /bin/bash

set -e
stack build -j$(nproc)  

TRAINIMG=train-images-idx3-ubyte.gz
TRAINLBL=train-labels-idx1-ubyte.gz
TESTIMG=t10k-images-idx3-ubyte.gz
TESTLBL=t10k-labels-idx1-ubyte.gz
[ -f $(basename $TRAINIMG .gz) ] || ( wget -O $TRAINIMG https://storage.googleapis.com/cvdf-datasets/mnist/$TRAINIMG && gunzip $TRAINIMG )
[ -f $(basename $TRAINLBL .gz) ] || ( wget -O $TRAINLBL https://storage.googleapis.com/cvdf-datasets/mnist/$TRAINLBL && gunzip $TRAINLBL )
[ -f $(basename $TESTIMG .gz) ] || ( wget -O $TESTIMG https://storage.googleapis.com/cvdf-datasets/mnist/$TESTIMG && gunzip $TESTIMG )
[ -f $(basename $TESTLBL .gz) ] || ( wget -O $TESTLBL https://storage.googleapis.com/cvdf-datasets/mnist/$TESTLBL && gunzip $TESTLBL )

X=0
[ -f configsMnist.txt ] || ./genConfigsMnist.pl | shuf | head -n 100 > configsMnist.txt
cat configsMnist.txt | while read CONFIG ; do
	OPTIM=$(echo "$CONFIG" | jshon -e 'optimizer' -u );
	LAYERS=$(echo "$CONFIG" | jshon -e 'layers' -u );
	LR=$(echo "$CONFIG" | jshon -e 'lr' -u );
	BETA1=$(echo "$CONFIG" | jshon -e 'beta1' -u );
	BETA2=$(echo "$CONFIG" | jshon -e 'beta2' -u );
	INPUTAF=$(echo "$CONFIG" | jshon -e 'inputAF' -u );
	COSTF=$(echo "$CONFIG" | jshon -e 'costF' -u);
	stack run --rts-options=-N$(nproc) -- mnist -R 1 -l "$LAYERS" -O "($OPTIM $LR $BETA1 $BETA2)" -s 100 -c "$COSTF" -I "$INPUTAF" | tee /tmp/results-mnist-$X.txt 
	mv mnist.ann nets/$X.ann
	mv /tmp/results-mnist-$X.txt results/

	X=$(($X + 1));
done

X=0
ls results/ | while read RESULT ; do
	#./mkGraph.m results/$RESULT "$X"
	#mv plot.png graphs/mnist-$X.png
	X=$(($X + 1));
done

./mnistTest.sh
