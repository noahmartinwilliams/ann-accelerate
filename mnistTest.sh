#! /bin/bash

set -e
#stack build all -j$(nproc)  

X=0
ls nets/ | sort -n | while read NET ; do
	stack run mnistTest nets/$NET | tee tests/$X
	NUMLINES=$(cat tests/$X | wc -l | sed 's/^\([0-9]*\)[[:space:]].*$/\1/g' )
	NUMTRUES=$(cat tests/$X | grep 'True' | wc -l | sed 's/^\([0-9]\)[[:space:]]*$/\1/g')
	echo "scale = 5 ; $NUMTRUES / $NUMLINES" | bc -lq >> tests/results-$X.txt
	X=$(($X + 1));
done

