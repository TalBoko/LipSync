# convert audio to phonemes and send to smart body while playing the audio file
# create git ignore file
# add option to convert unlabeled file to phonemes
import Utils
from model import MyModel
import model_config
from PhonesSet import PhonesSet
import threading
from model_stomp_client import send_visemes_to_smartbody
from WaveStreamer import play_wav


def main():
    audio_path = 'wav_files\\long_dr1_7.wav'
    # NOTE: no overlap in the input vectors frames
    inputs = Utils.audio_to_model_inputs(audio_path, model_config.input_vec_size)
    inputs_transposed = Utils.convert_to_cnn_inputs(inputs)
    input_len = len(inputs_transposed[0])
    num_of_features = len(inputs_transposed[0][0])
    num_of_labels = PhonesSet.get_num_labels()

    model = MyModel(input_len=input_len, num_features=num_of_features, num_labels=num_of_labels)
    model_file = model_config.model_file_name
    model.build_model("CONV1", [], [], weights_path=model_file)

    thread = threading.Thread(target=send_visemes_to_smartbody, args=(model,inputs_transposed))
    thread.daemon = True  # Daemon thread
    thread.start()

    loader_audio_path = 'wav_files\\long_dr1_7_loader.wav'
    play_wav(loader_audio_path)

if __name__ == '__main__':
    main()
