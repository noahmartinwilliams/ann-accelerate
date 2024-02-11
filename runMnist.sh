#! /bin/bash

set -e
cabal build all -j16  

X=0
[ -f configsMnist.txt ] || ./genConfigsMnist.pl | shuf > configsMnist.txt
cat configsMnist.txt | while read CONFIG ; do
	OPTIM=$(echo "$CONFIG" | jshon -e 'optimizer' -u );
	LAYERS=$(echo "$CONFIG" | jshon -e 'layers' -u );
	LR=$(echo "$CONFIG" | jshon -e 'lr' -u );
	BETA1=$(echo "$CONFIG" | jshon -e 'beta1' -u );
	BETA2=$(echo "$CONFIG" | jshon -e 'beta2' -u );
	INPUTAF=$(echo "$CONFIG" | jshon -e 'inputAF' -u );
	COSTF=$(echo "$CONFIG" | jshon -e 'costF' -u);
	./mnist.exe +RTS -N16 -RTS -R 200 -l "$LAYERS" -O "($OPTIM $LR $BETA1 $BETA2)" -s 100 -c "$COSTF" -I "$INPUTAF" | tee /tmp/results-mnist-$X.txt 
	mv mnist.ann nets/$X.ann
	mv /tmp/results-mnist-$X.txt results/
	X=$(($X + 1));
done
mv /tmp/results-mnist-*.txt ./results

X=0
ls results/results-mnist-*.txt | while read RESULT ; do
	./mkGraph.m results/$RESULT "$X"
	mv plot.png graphs/mnist-$X.png
	X=$(($X + 1));
done
