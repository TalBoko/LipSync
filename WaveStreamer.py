import pyaudio
import wave
import time


def play_wav(file_path):
    print('start playing wav....')
    # NOTE: define stream chunk
    chunk = 1024

    # NOTE: open a wav format music
    time.sleep(1.5)
    f = wave.open(file_path, "rb")
    # NOTE: instantiate PyAudio
    p = pyaudio.PyAudio()
    # NOTE: open stream
    stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                    channels=f.getnchannels(),
                    rate=f.getframerate(),
                    output=True)
    # NOTE: read data
    data = f.readframes(chunk)

    # NOTE: play stream
    while data:
        stream.write(data)
        data = f.readframes(chunk)

    # NOTE: stop stream
    stream.stop_stream()
    stream.close()

    # NOTE: close PyAudio
    p.terminate()
    print('wav ended')
