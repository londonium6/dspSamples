%% abstract
% sample MATLAB script for Fiverr demonstration. londonium6@berkeley.edu
% 
% MIT License
% 
% Copyright (c) 2017 mr_matlab
% 
% Permission is hereby granted, free of charge, to any person obtaining a 
% copy of this software and associated documentation files (the 
% "Software"), to deal in the Software without restriction, including 
% without limitation the rights to use, copy, modify, merge, publish, 
% distribute, sublicense, and/or sell copies of the Software, and to 
% permit persons to whom the Software is furnished to do so, subject to 
% the following conditions:
% 
% The above copyright notice and this permission notice shall be included 
% in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
% OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
% IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
% CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
% TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
% SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
%
%

%% clear environment

clear all; close all; clc;

%% set parameters

% Sampling rate (Samples/sec)
fs=1e6;
% Center frequency (Hz)
fc=15;
% Simulation time (sec)
tDur=1.0;
% Signal Power (Watt)
nPwr=1;
% Plotting bounds (Volt)
yBound=5;

%% establish relevant variables/vectors for signal construction

nVec=round(fs*tDur);
fVec=transpose(linspace(fc/fs,fc/fs,nVec));
tVec=transpose(linspace(0,(nVec-1)/fs,nVec));

%% generate signals

% generate initial signals
sigOrig=cos(2*pi*cumsum(fVec));
noiseVec=sqrt(nPwr)*randn(nVec,1);
sigAWGN=sigOrig+noiseVec;

% build relevant low pass filter
hLpf=transpose(fir1(nVec-1,5*fc/fs));
hLpfDelay=round(mean(grpdelay(hLpf,1)));

% utilize fft convolution to apply filter
% assumption: sinusoid repeats ad infinitum, not just for sim time
sigFilt=real(ifft(fft(hLpf).*fft(sigAWGN)));
sigFilt=circshift(sigFilt,[-hLpfDelay,0]);

%% generate plots

% establish figure object for plotting
hFig=figure; 

% Original Signal
subplot(3,1,1);
plot(tVec,real(sigOrig)); 
ylim(yBound*[-1,1]);
xlabel('Time, sec');
title('Original Signal');
grid on;

% Noisy Signal
subplot(3,1,2);
plot(tVec,real(sigAWGN)); 
ylim(yBound*[-1,1]);
xlabel('Time, sec');
title('Signal through AWGN');
grid on;

% Filtered Noise Signal
subplot(3,1,3);
plot(tVec,real(sigFilt)); 
ylim(yBound*[-1,1]);
xlabel('Time, sec');
title('AWGN Signal Post-Filter');
grid on;
