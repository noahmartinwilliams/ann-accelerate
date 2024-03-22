#! /bin/bash

set -e
cabal build all -j16  

X=0
./genConfigs.pl > configs.txt
cat configs.txt | while read CONFIG ; do
	OPTIM=$(echo "$CONFIG" | jshon -e 'optimizer' -u );
	LAYERS=$(echo "$CONFIG" | jshon -e 'layers' -u );
	LR=$(echo "$CONFIG" | jshon -e 'lr' -u );
	BETA1=$(echo "$CONFIG" | jshon -e 'beta1' -u );
	BETA2=$(echo "$CONFIG" | jshon -e 'beta2' -u );
	stack run mkNet -l "$LAYERS" -O "($OPTIM $LR $BETA1 $BETA2)" -s 200 -o nets/untrained$X.ann ;
	stack run genOrGate | stack run trainNet -I nets/untrained$X.ann -O nets/trained$X.ann | tee /tmp/results$X.txt ;
	X=$(($X + 1));
done
mv /tmp/results*.txt ./results

X=0
ls results | while read RESULT ; do
	./mkGraph.m results/$RESULT "$X"
	mv plot.png graphs/$X.png
	X=$(($X + 1));
done

X=0;
ls nets/trained*.ann | while read NET ; do
	stack run genOrGate | cut -f 1 -d '#' | stack run calcNet -I nets/$NET | tee /tmp/outputs$X.txt
	X=$(($X + 1));
done
mv /tmp/outputs*.txt .
