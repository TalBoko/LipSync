import numpy
import sigproc
from scipy.fftpack import dct
from collections import deque

try:
    xrange(1)
except:
    xrange = range


class REALTIME_MFCC:
    def __init__(self):
        self.energy_queue = deque(maxlen=4)
        self.energy_delta_queue = deque(maxlen=4)
        self.mfcc_queue = deque(maxlen=4)
        self.mfcc_delta_queue = deque(maxlen=4)
        return
    #def mfcc(self, signal, samplerate=8000, winlen=0.025, winstep=0.01, numcep=13,
    #        nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True,
    def mfcc(self, signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
            nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True,
            winfunc=lambda x: numpy.ones((x,))):
        """Compute MFCC features from an audio signal.
        :param signal: the audio signal from which to compute features. Should be an N*1 array
        :param samplerate: the samplerate of the signal we are working with.
        :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
        :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
        :param numcep: the number of cepstrum to return, default 13
        :param nfilt: the number of filters in the filterbank, default 26.
        :param nfft: the FFT size. Default is 512.
        :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
        :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
        :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
        :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
        :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
        :param winfunc: the analysis window to apply to each frame. By default no window is applied.
        :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
        """
        feat, energy = self.fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph, winfunc)
        feat = numpy.log(feat)
        feat = dct(feat, type=2, axis=1, norm='ortho')[:, :numcep]
        feat = self.lifter(feat, ceplifter)
        self.mfcc_queue.append(feat)
        self.energy_queue.append(numpy.log(energy))

        if len(self.mfcc_queue) >= 2:
            delta = numpy.zeros_like(feat)
            for i in range(1, len(self.mfcc_queue)):
                delta += self.mfcc_queue[i] - self.mfcc_queue[i - 1]
            delta /= len(self.mfcc_queue) - 1
            self.mfcc_delta_queue.append(delta)
        else:
            delta = numpy.zeros_like(feat)

        if len(self.mfcc_delta_queue) >= 2:
            acc = numpy.zeros_like(feat)
            for i in range(1, len(self.mfcc_delta_queue)):
                acc += self.mfcc_delta_queue[i] - self.mfcc_delta_queue[i - 1]
            acc /= len(self.mfcc_delta_queue) - 1
        else:
            acc = numpy.zeros_like(feat)
        energy = numpy.log(energy)
        if len(self.energy_queue) >= 2:
            energy_delta = numpy.zeros_like(energy)
            for i in range(1, len(self.energy_queue)):
                energy_delta += self.energy_queue[i] - self.energy_queue[i - 1]
                energy_delta /= len(self.energy_queue) - 1
            self.energy_delta_queue.append(energy_delta)
        else:
            energy_delta = numpy.zeros_like(energy)

        if len(self.energy_delta_queue) >= 2:
            energy_acc = numpy.zeros_like(energy)
            for i in range(1, len(self.energy_delta_queue)):
                energy_acc += self.energy_delta_queue[i] - self.energy_delta_queue[i - 1]
                energy_acc /= len(self.energy_delta_queue) - 1
        else:
            energy_acc = numpy.zeros_like(energy)


        if appendEnergy:
            feat[:, 0] = energy  # replace first cepstral coefficient with log of frame energy
            delta[:, 0] = energy_delta
            acc[:, 0] = energy_acc
        feat = feat[0]
        feat = numpy.append(feat, delta)
        feat = numpy.append(feat, acc)
        return feat

    def fbank(self, signal, samplerate=16000, winlen=0.025, winstep=0.01,
              nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
              winfunc=lambda x: numpy.ones((x,))):
        """Compute Mel-filterbank energy features from an audio signal.

        :param signal: the audio signal from which to compute features. Should be an N*1 array
        :param samplerate: the samplerate of the signal we are working with.
        :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
        :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
        :param nfilt: the number of filters in the filterbank, default 26.
        :param nfft: the FFT size. Default is 512.
        :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
        :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
        :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
        :param winfunc: the analysis window to apply to each frame. By default no window is applied.
        :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
            second return value is the energy in each frame (total energy, unwindowed)
        """
        highfreq = highfreq or samplerate / 2
        try:
            signal = sigproc.preemphasis(signal, preemph)
        except:
            signal = signal
            #print(signal)
        frames = sigproc.framesig(signal, winlen * samplerate, winstep * samplerate, winfunc)
        pspec = sigproc.powspec(frames, nfft)
        energy = numpy.sum(pspec, 1)  # this stores the total energy in each frame
        energy = numpy.where(energy == 0, numpy.finfo(float).eps, energy)  # if energy is zero, we get problems with log

        fb = self.get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
        feat = numpy.dot(pspec, fb.T)  # compute the filterbank energies
        feat = numpy.where(feat == 0, numpy.finfo(float).eps, feat)  # if feat is zero, we get problems with log

        return feat, energy

    def logfbank(self, signal, samplerate=16000, winlen=0.025, winstep=0.01,
                 nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97):
        """Compute log Mel-filterbank energy features from an audio signal.

        :param signal: the audio signal from which to compute features. Should be an N*1 array
        :param samplerate: the samplerate of the signal we are working with.
        :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
        :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
        :param nfilt: the number of filters in the filterbank, default 26.
        :param nfft: the FFT size. Default is 512.
        :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
        :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
        :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
        :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
        """
        feat, energy = self.fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph)
        return numpy.log(feat)

    def ssc(self, signal, samplerate=16000, winlen=0.025, winstep=0.01,
            nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
            winfunc=lambda x: numpy.ones((x,))):
        """Compute Spectral Subband Centroid features from an audio signal.

        :param signal: the audio signal from which to compute features. Should be an N*1 array
        :param samplerate: the samplerate of the signal we are working with.
        :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
        :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
        :param nfilt: the number of filters in the filterbank, default 26.
        :param nfft: the FFT size. Default is 512.
        :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
        :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
        :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
        :param winfunc: the analysis window to apply to each frame. By default no window is applied.
        :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
        """
        highfreq = highfreq or samplerate / 2
        signal = sigproc.preemphasis(signal, preemph)
        frames = sigproc.framesig(signal, winlen * samplerate, winstep * samplerate, winfunc)
        pspec = sigproc.powspec(frames, nfft)
        pspec = numpy.where(pspec == 0, numpy.finfo(float).eps, pspec)  # if things are all zeros we get problems

        fb = self.get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
        feat = numpy.dot(pspec, fb.T)  # compute the filterbank energies
        R = numpy.tile(numpy.linspace(1, samplerate / 2, numpy.size(pspec, 1)), (numpy.size(pspec, 0), 1))

        return numpy.dot(pspec * R, fb.T) / feat

    def hz2mel(self, hz):
        """Convert a value in Hertz to Mels

        :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
        :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
        """
        return 2595 * numpy.log10(1 + hz / 700.0)

    def mel2hz(self, mel):
        """Convert a value in Mels to Hertz

        :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
        :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
        """
        return 700 * (10 ** (mel / 2595.0) - 1)

    def get_filterbanks(self, nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
        """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
        to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

        :param nfilt: the number of filters in the filterbank, default 20.
        :param nfft: the FFT size. Default is 512.
        :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
        :param lowfreq: lowest band edge of mel filters, default 0 Hz
        :param highfreq: highest band edge of mel filters, default samplerate/2
        :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
        """
        highfreq = highfreq or samplerate / 2
        assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

        # compute points evenly spaced in mels
        lowmel = self.hz2mel(lowfreq)
        highmel = self.hz2mel(highfreq)
        melpoints = numpy.linspace(lowmel, highmel, nfilt + 2)
        # our points are in Hz, but we use fft bins, so we have to convert
        #  from Hz to fft bin number
        bin = numpy.floor((nfft + 1) * self.mel2hz(melpoints) / samplerate)

        fbank = numpy.zeros([nfilt, nfft / 2 + 1])
        for j in xrange(0, nfilt):
            for i in xrange(int(bin[j]), int(bin[j + 1])):
                fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
            for i in xrange(int(bin[j + 1]), int(bin[j + 2])):
                fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
        return fbank

    def lifter(self, cepstra, L=22):
        """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
        magnitude of the high frequency DCT coeffs.

        :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
        :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
        """
        if L > 0:
            nframes, ncoeff = numpy.shape(cepstra)
            n = numpy.arange(ncoeff)
            lift = 1 + (L / 2) * numpy.sin(numpy.pi * n / L)
            return lift * cepstra
        else:
            # values of L <= 0, do nothing
            return cepstra
