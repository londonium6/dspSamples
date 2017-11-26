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
fc=50.0                 # center frequency, Hz
tDur=0.1                # Sim time, sec
nPwr=1.0                # Signal power, W
plotFactor=int(1.5e3)   # Plotting downsampling factor

# establish relevant vectors for signal construction
nVec=int(fs*tDur)
fVec=np.linspace(fc/fs,fc/fs,nVec)
tVec=np.linspace(0,(nVec-1)/fs,nVec)

# generate initial signals
sigOrig=np.cos(2*np.pi*np.cumsum(fVec))
noiseVec=np.sqrt(nPwr)*np.random.normal(0.0,0.707,nVec)
sigAWGN=sigOrig+noiseVec

# build FIR lowpass filter
hLpf=scpSignal.firwin(nVec-1,5*fc/fs)
hLpfDelay=int(1/tDur*np.mean(scpSignal.group_delay((hLpf,1))))

# utilize fft convolution to apply FIR filter
# assumption: sinusoid repeats ad infinitum, not just for sim time
sigFilt=(np.fft.ifft(np.fft.fft(hLpf,nVec)*np.fft.fft(sigAWGN))).real
sigFilt=np.roll(sigFilt,int(hLpfDelay))

# plot sinusoid with noise
noisyPlt,=plt.plot(tVec,sigAWGN)

# plot filtered sinusoid
cleanPlt,=plt.plot(tVec[0::plotFactor],sigFilt[0::plotFactor],'ro')

# finalize plot options
plt.ylabel('Voltage, V')
plt.xlabel('Time, sec')
plt.title('Noisy signal vs filtered output')
plt.xlim(np.min(tVec),np.max(tVec))
plt.legend([noisyPlt,cleanPlt],['Noisy Signal','Filtered'])
plt.grid(True)

# display plot
plt.show()
