import pyaudio
import wave

def main():
    #record
    CHUNK = 400#1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 8000#16000
    RECORD_SECONDS =10
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("*recording")

    frames = []

    for i in range(0, int(RATE/CHUNK*RECORD_SECONDS)):
     data  = stream.read(CHUNK)
     frames.append(data)
     print("Chunk "  + str(i))

    print("*done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()


    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print("*closed")


if __name__ == '__main__':
    main()
