#! /usr/bin/octave -q

x=load(argv(){1});
hf = figure();
h=plot(x);
ylabel("error");
xlabel("number of samples");
title(argv(){2});
print -dpng plot.png
