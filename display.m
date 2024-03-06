#! /usr/bin/octave -q

while (1)
	x=load(argv(){1});
	h=plot(x);
	figure(1, "position", get(0, "screensize"));
	clear();
	pause(1);
endwhile
