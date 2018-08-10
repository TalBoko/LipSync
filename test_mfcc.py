import librosa
import librosa.display
import librosa.core.spectrum
import MFCC
import wave
import numpy as np
import matplotlib.pyplot as plt
SAMPLE_RATE = 16000
def generate_mfcc_features_librosa(file_path):
    # Load the example clip
    y, sr = librosa.load(file_path,sr=SAMPLE_RATE)
    secs = len(y)/SAMPLE_RATE
    mfcc_list = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13)#secs*100
    mfcc_delta = librosa.feature.delta(mfcc_list, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc_list, order=2)
    return mfcc_list,mfcc_delta,mfcc_delta2

def get_mfcc_no_dct(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    secs = len(y) / SAMPLE_RATE
    #mfccs = librosa.core.spectrum.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr))
    mfcc_list = librosa.feature.melspectrogram(y=y, sr=16000) # try  40-80-100(output size)
    return mfcc_list

def generate_mfcc_old_implementation(file_path):
    rt_mfcc = MFCC.REALTIME_MFCC()
    audio_signal= read_wav_file(file_path)
    mfcc_list = rt_mfcc.mfcc(audio_signal)
    return mfcc_list

def read_wav_file(file_path):
    wav = wave.Wave_read(file_path)
    # rerurns the number of waves in the file:
    N = wav.getnframes()
    # Reads and returns at most n frames of audio, as a string of byte
    dstr = wav.readframes(N)
    signal = np.fromstring(dstr, np.int16)
    return signal


file_path = 'C:\Users\user\Documents\quick\speaker1\s0101a.wav'
mfcc_no_dct = get_mfcc_no_dct(file_path)

plt.figure(figsize=(10, 4))

librosa.display.specshow(mfcc_no_dct, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

'''
mfcc_old = generate_mfcc_old_implementation(file_path)
librosa.display.specshow(mfcc_old, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()
'''
mfcc,delta,delta2 = generate_mfcc_features_librosa(file_path)

plt.figure(figsize=(10, 4))

librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

mfcc_with_delta = np.append(mfcc,delta, axis=0)
mfcc_with_delta = np.append(mfcc_with_delta,delta2, axis=0)

librosa.display.specshow(mfcc_with_delta, x_axis='time')

librosa.display.specshow(delta, x_axis='time')
plt.colorbar()
plt.title('DELTA')
plt.tight_layout()
plt.show()

librosa.display.specshow(delta2, x_axis='time')
plt.colorbar()
plt.title('DELTA2')
plt.tight_layout()
plt.show()

print("finished")