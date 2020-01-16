function [resultingSig,period] = generateDiff(x,li,ampleft,durleft,durbetween,ampright,durright,shift)
% Call: generateDiff(ampleft,durleft,durbetween,ampright,durright,shift)
% Units are seconds 
% Amp in volts
% Dmytro Kolenov, 17th April 2019
a_qrswav = ampleft;
d_qrswav = durleft;
t_swav   = durbetween;
a_swav   = ampright;
d_swav   = durright;

%period = d_qrswav + (d_swav + t_swav)/2;
period = (d_qrswav + d_swav);

qrswav = qrs_wav(x,a_qrswav, d_qrswav, li);
swav = s_wav(x,a_swav, d_swav,t_swav, li);
% 
% figure(),plot(x,circshift(qrswav,floor(length(qrswav)/2)))
% figure(),plot(x,circshift(swav,floor(length(swav)/2)))
resultingSig = circshift(qrswav, shift) + circshift(swav, shift);