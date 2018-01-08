# abstract
# sample python script for Fiverr demonstration. londonium6@berkeley.edu
# 
# MIT License
# 
# Copyright (c) 2017 mr_matlab
# 
# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the 
# "Software"), to deal in the Software without restriction, including 
# without limitation the rights to use, copy, modify, merge, publish, 
# distribute, sublicense, and/or sell copies of the Software, and to 
# permit persons to whom the Software is furnished to do so, subject to 
# the following conditions:
# 
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
#

# clear workspace
from IPython import get_ipython
get_ipython().magic('reset -sf')

# set up libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scpSignal

# setup input parameters
fs=1.0e6                # sampling rate, Hz
fc=1.0e1                # center frequency, Hz
tDur=1.0                # Sim time, sec
nPwr=0.5e1              # Signal power, W
maxFFT=262144           # Maximum filter size
filterBW=5*fc/fs        # Filter design bandwidth
trWidthPer=200.0        # Filter transition width as fraction of bandwidth

# establish relevant vectors for signal construction
nVec=int(fs*tDur)
fVec=np.linspace(fc/fs,fc/fs,nVec)
tVec=np.linspace(0,(nVec-1)/fs,nVec)

# generate initial signals
sigOrig=np.cos(2*np.pi*np.cumsum(fVec))
noiseVec=np.random.normal(0.0,np.sqrt(nPwr),nVec)
sigAWGN=sigOrig+noiseVec

# build FIR lowpass filter
# determine minimum filter taps as power of 2 (max = maxFFT)
trWidth=filterBW*fs*trWidthPer/100.0
rippleTol=1.0e-4
stopSup=1.0e-6
filterTaps=2.0/3.0*np.log10(1/(10.0*rippleTol*stopSup))*fs/trWidth
filterTaps=int(np.power(2,np.floor(np.log2(filterTaps))+1))
filterTaps=np.amin(np.array([int(filterTaps),maxFFT]))-1
# build standard FIR filter
hLpf=scpSignal.firwin(filterTaps,filterBW)
# compute FIR filter delay
w,hLpfDelayArray=scpSignal.group_delay((hLpf,1))
hLpfDelay=int(np.mean(hLpfDelayArray))

# apply filtering
nFiltLoops=int(nVec/maxFFT)
# if data vector size surpasses max FFT size, 
# then apply filter via overlap-add with FFT/circular convolution 
if nFiltLoops>1:
    nFiltOut=int(maxFFT+filterTaps-1)
    sigFilt=np.zeros(nVec)
    for filtIndex in range(0,nFiltLoops+1):
        startIndex=maxFFT*filtIndex
        finalIndex=np.min(np.array([nFiltOut-1+maxFFT*filtIndex,nVec-1]))
        finConvIndex=np.min(np.array([startIndex-1+maxFFT,nVec-1]))
        sigConv = \
        scpSignal.fftconvolve(sigAWGN[startIndex:finConvIndex],hLpf,'full')
        sigTemp=sigConv[0:(finalIndex-startIndex)]
        sigFilt[startIndex:finalIndex]+=sigTemp
# otherwise, apply filter by linear convolution
else:
    sigFilt=scpSignal.lfilter(hLpf,1,sigAWGN)

# linear shift to the left (account for filter group delay)
sigFilt=np.concatenate((sigFilt[hLpfDelay::],np.zeros(hLpfDelay)),axis=0)

# plot sinusoid with noise
noisyPlt,=plt.plot(tVec,sigAWGN,'b')

# plot filtered sinusoid
cleanPlt,=plt.plot(tVec,sigFilt,'r')

# finalize plot options
plt.ylabel('Voltage, V')
plt.xlabel('Time, sec')
plt.title('Noisy signal vs filtered output')
plt.xlim(np.min(tVec),np.max(tVec))
plt.legend([noisyPlt,cleanPlt],['Noisy Signal','Filtered'])
plt.grid(True)

# display plot
plt.show()
