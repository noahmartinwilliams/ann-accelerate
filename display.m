#! /usr/bin/octave -q

x=dlmread(argv(){1}, ',', 1, 0);
plot(x(:,1));
figure(1, "position", get(0, "screensize"));
clear();
pause(512);
