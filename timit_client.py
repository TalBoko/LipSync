import os
import fnmatch
from WavReader import *
from PhonesSet import *
import logging
import model_config


class Timit:
    def __init__(self,dir_path):
        self.path = dir_path
        self.rate = 16000
        self.files_by_speaker_folder = {}
        self.init_files_list()
        self.config_logger()

    def init_files_list(self):
        for root, dirnames, filenames in os.walk(self.path):
            for filename in fnmatch.filter(filenames, '*_new.wav'):
                phone_file_name = filename.split('_new.')[0] + '_phn.labels'
                wav_path = os.path.join(root, filename)
                phone_path = os.path.join(root, phone_file_name)
                if root not in self.files_by_speaker_folder.keys():
                    self.files_by_speaker_folder[root] = []
                if True or not filename.startswith('sa'):
                    self.files_by_speaker_folder[root] += [(wav_path, phone_path)]
                else:
                    print('ignored file ' + filename)

    def config_logger(self):
        # create logger
        logging.basicConfig(filename='logger.log', level=logging.DEBUG)

    def get_corpus(self):
        #logging.info("start loading speaker data :  {} ".format(self.dir_path))

        self.inputs= []
        self.labels = []

        for speaker_folder in self.files_by_speaker_folder.keys():
            speaker_inputs = []
            speaker_labels = []
            for wav_path, phone_path in self.files_by_speaker_folder[speaker_folder]:
                try:
                    inputs, labels = self.generate_labeled_data_from_track(wav_path,phone_path)

                    speaker_inputs.extend(inputs)
                    speaker_labels.extend(labels)

                except Exception as ex:
                    print("failed to generate data from file " + wav_path)
                    print (str(ex))
                    pass
            # NOTE: normalize data per speaker folder
            normalized_inputs = Timit.normalize_features(speaker_inputs)
            self.inputs.extend(normalized_inputs)
            self.labels.extend(speaker_labels)

        return self.inputs, self.labels

    def generate_labeled_data_from_track(self, wav_path, phone_path):
        #load phonemes labels
        labels = []
        inputs = []
        with open(phone_path) as phone_file:
            labels_data = phone_file.read()
        labels_list = labels_data.split('\n')

        # NOTE: load audio features
        feature_type_str = model_config.feature_type
        features_type = Feature_Type[feature_type_str]

        wav_reader = WavReader(16000, frame_len=0.025, shift_len=0.01, mfcc_type=features_type)
        track_mfccs = wav_reader.get_all_frame_features(wav_path, model_config.num_mels)

        # TODO - verify we ignore the right frames
        for i in range(len(labels_list)):
            phoneme_label = PhonesSet.get_mapping(labels_list[i])
            if phoneme_label != PhonesSet.DEFAULT:
                labels.append(phoneme_label)
                inputs.append(track_mfccs[i])
            else:
                print "skipped on unknown frame: {}".format(labels_list[i])
        return inputs, labels

    @staticmethod
    def calc_normalization_properties(inputs):
        window_size = 10000
        sum_features_list = np.sum(inputs[:window_size], axis=0)
        features_average = sum_features_list / window_size
        std_list = np.std(inputs[:window_size], axis=0)
        return features_average,std_list

    @staticmethod
    def normalize_features(inputs):
        average_list, std_list = Timit.calc_normalization_properties(inputs)
        normalized_inputs = inputs - average_list
        normalized_inputs = normalized_inputs / std_list
        return normalized_inputs


# TODO - add config selection for timit/buckeye, copy files to run test on windows
# change in Dataset the use of the correct client

#verfiy normalization required in different featues types.
