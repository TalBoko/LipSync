import wave
import numpy as np
from MFCC import REALTIME_MFCC
from collections import deque
from librosa import *
from enum import Enum


class Feature_Type(Enum):
    REAL_TIME = 1
    LIBROSA = 2
    MEL_SPECTOGRAM = 3
    SPECTOGRAM = 4


class WavReader():
    # def __init__(self, queue, file_path, frame_size=200, shift=80):
    def __init__(self, rate, frame_len=0.025, shift_len=0.01, mfcc_type=Feature_Type.REAL_TIME):
        '''
        :param file_path: full wave file path
        :param rate: wav file rate
        :param frame_len: frame len in sec units
        :param shift_len: shift len im sec units
        :param mfcc_type: type of mfcc extraction to use - 'REAL_TIME'\'LIBROSA'\'NO_DCT'
        '''
        self.rate = rate
        self.rt_mfcc = REALTIME_MFCC()
        self.mfcc_type = mfcc_type
        self.jump = shift_len * rate
        self.frame_size = frame_len * rate
        self.place = 0
        return

    def get_all_frame_features(self, file_path, num_mels=None):
        if self.mfcc_type == Feature_Type.REAL_TIME:
            return self.get_all_frames_mfcc_real_time(file_path)
        if self.mfcc_type == Feature_Type.LIBROSA:
            return self.get_all_frames_mfcc_librosa(file_path)
        if self.mfcc_type == Feature_Type.MEL_SPECTOGRAM:
            return self.get_all_frames_mel_spectogram(file_path, num_mels)
        if self.mfcc_type == Feature_Type.SPECTOGRAM:
            return self.get_all_frames_spectogarm_before_mel(file_path)

    def get_all_frames_mfcc_real_time(self,file_path):
        self.read_wav_file(file_path)
        frames_as_mfcc = []
        while not self.place >= self.N - self.frame_size:
            frame = self.data[self.place:int(self.place +self.frame_size)]
            mfcc = self.extract_mfcc_features(frame)
            frames_as_mfcc += [mfcc]
            self.place += int(self.jump)
        return frames_as_mfcc

    def extract_mfcc_features(self, frame):
        input_data = deque(np.zeros(self.frame_size, np.int16), maxlen=self.frame_size)
        input_data.extend(frame)
        mfcc = self.rt_mfcc.mfcc(input_data)

        return mfcc

    def get_all_frames_mfcc_librosa(self, file_path):
        '''
        get mfcc features with "librosa" library 
        :param file_path: audio file path
        :return: list of mfcc for each 512 window size, with shift of 0.01ms(160) 
        '''
        y, sr = load(file_path, sr=self.rate, dtype=np.int16)
        mfccs = feature.mfcc(y=y, sr=self.rate, n_mfcc=13)  # secs*100
        delta = feature.delta(mfccs, order=1)
        delta2 = feature.delta(mfccs, order=2)
        mfcc_with_delta = np.append(mfccs, delta, axis=0)
        mfcc_with_delta = np.append(mfcc_with_delta, delta2, axis=0)
        mfcc_with_delta = mfcc_with_delta.T # from rows to columns

        return mfcc_with_delta

    def get_all_frames_mel_spectogram(self, file_path, num_mels):
        y, sr = load(file_path, sr=self.rate)
        # NOTE:max nftt = 512)
        mfcc_list = feature.melspectrogram(y=y, sr=self.rate, S=None,
                                           n_fft=int(self.frame_size),
                                           hop_length=int(self.jump),
                                           power=2.0, n_mels=num_mels)
        # try  40-80-100(output size) n-mels max value is 257 because n_ftt is 512(512/2)
        return mfcc_list.T

    def get_all_frames_spectogarm_before_mel(self,file_path):
        y, sr = load(file_path, sr=self.rate)
        S, n_fft = spectrum._spectrogram(y=y, n_fft=self.frame_size, hop_length=self.jump,# was before :512,160
                                power=2)
        return S.T

    def read_wav_file(self, file_path):
        self.wav = wave.Wave_read(file_path)
        # rerurns the number of waves in the file:
        self.N = self.wav.getnframes()
        # Reads and returns at most n frames of audio, as a string of byte
        dstr = self.wav.readframes(self.N)
        self.data = np.fromstring(dstr, np.int16)



