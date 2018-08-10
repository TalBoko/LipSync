import pyaudio
import wave

def play_wav(file_path):
    print('start playing wav....')
    #define stream chunk
    chunk = 1024

    #open a wav format music
    #r"C:\Users\user\Desktop\timit-code\data\dr1\si648_new.wav"
    f = wave.open(file_path,"rb")
    #instantiate PyAudio
    p = pyaudio.PyAudio()
    #open stream
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
                    channels = f.getnchannels(),
                    rate = f.getframerate(),
                    output = True)
    #read data
    data = f.readframes(chunk)

    #play stream
    while data:
        stream.write(data)
        data = f.readframes(chunk)

    #stop stream
    stream.stop_stream()
    stream.close()

    #close PyAudio
    p.terminate()
    print('wav ended')