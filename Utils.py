import numpy as np
from TimitDataSet import TimitDataSet
from PhonesSet import PhonesSet
import model_config
import os
import timit_client
from WavReader import WavReader, Feature_Type
max_data_len_for_writing = 500000


def get_many_to_one_labels(labels,number_of_labels):
    new_labels = []
    for tags in labels:
        tags_list = tags.tolist()
        is_labels_valid(tags)
        new_label = most_common(tags_list,number_of_labels)
        new_labels += [new_label]#[[most_common(tags_list)]]
    return np.asarray(new_labels)


def is_labels_valid(labels):
    for label in labels:
        assert sum(label) == 1,"invalid label : "+ str(labels)


def most_common(lst,num_of_labels):
    sum_list = [sum(x) for x in zip(*lst)]
    max_index_value = sum_list.index(max(sum_list))
    label_vector = np.zeros(num_of_labels)
    label_vector[max_index_value] = 1
    return label_vector


def get_many_to_one_inputes(inputs):
    flatten_inputs = []
    for input in inputs:
        flat_list = [item for sublist in input for item in sublist]
        flatten_inputs +=[flat_list]

    return flatten_inputs


def save_data_to_files(base_file_name,data):
    file_index = 0
    for i in range(0, len(data), max_data_len_for_writing):
        file_name = base_file_name + str(file_index)
        np.save(file_name, data[:i+max_data_len_for_writing])
        file_index += 1


def load_data_from_files(base_file_name):
    file_index = 0
    file_name = base_file_name+str(file_index)+'.npy'
    X = None
    while os.path.exists(file_name):
        X1 = np.load(file_name)
        if X is None:
            X = X1
        else:
            X = np.concatenate((X,X1))
        file_index+=1
        file_name = base_file_name + str(file_index) + '.npy'

    return X


def convert_to_cnn_inputs_and_labels(inputs, labels):
    inputs_transposed = convert_to_cnn_inputs(inputs)

    labels_many_to_one = get_many_to_one_labels(labels, PhonesSet.get_num_labels())
    num_of_features = len(inputs_transposed[0][0])
    input_len = len(inputs_transposed[0])
    return inputs_transposed, labels_many_to_one, input_len, num_of_features


def convert_to_cnn_inputs(inputs):
    inputs_transposed = []
    for input_seq in inputs:
        inputs_transposed += [input_seq.T]

    inputs_transposed = np.asarray(inputs_transposed)
    return inputs_transposed


def load_data_for_validation(validation_path, input_vec_size):
    dataset = TimitDataSet(validation_path, input_vec_size, overlap=4)
    X, Y = dataset.get_dirs_data(8)

    return X, Y


def convert_audio_to_phonemes(audio_path, input_vec_size):
    pass


def audio_to_model_inputs(audio_path, input_size):
    feature_type_str = model_config.feature_type
    mfcc_type = Feature_Type[feature_type_str]

    wav_reader = WavReader(16000, frame_len=0.025, shift_len=0.01, mfcc_type=mfcc_type)
    frames = wav_reader.get_all_frame_features(audio_path, model_config.num_mels)
    frames = timit_client.Timit.normalize_features(frames)
    num_of_inputs = len(frames)
    input_vectors = []

    for i in range(0, num_of_inputs - input_size + 1, input_size):
        input_vectors += [frames[i:i + input_size]]

    print "num of input vectors = {}".format(len(input_vectors))
    return np.asarray(input_vectors)


def load_training_data(input_dir_path):
    labels_file_name = model_config.labels_file_name
    inputs_file_name = model_config.inputs_file_name

    timit_dataset = TimitDataSet(input_dir_path, model_config.input_vec_size, overlap=model_config.input_overlap)

    if not model_config.load_data_from_file:
        inputs, labels = timit_dataset.get_dirs_data(num_of_dirs=41)

        # NOTE : save loaded data to files
        save_data_to_files(inputs_file_name, inputs)
        save_data_to_files(labels_file_name, labels)
        print("finished loading data, saved to files")
    else:
        # NOTE : load data from files
        inputs = load_data_from_files(inputs_file_name)
        labels = load_data_from_files(labels_file_name)

        print("finished loading data from files")
        print("total num of inputes : " + str(len(inputs)))
    return inputs,labels


def print_compare(compare_file_path, labels, predictions):
    accuracy_per_label = {}

    assert (len(labels) == len(predictions))
    results_str = "labels   prediction \n"
    for i in range(len(labels)):
        label_str = PhonesSet.from_vector_to_label(labels[i])
        pred_str = PhonesSet.from_vector_to_label(predictions[i])
        results_str += '{}\t{}\n'.format(label_str, pred_str)

        label_smart_body = PhonesSet.PHONES_MAPPING.get(label_str, PhonesSet.DEFAULT)
        pred_smart_body = PhonesSet.PHONES_MAPPING.get(pred_str, PhonesSet.DEFAULT)

        if label_str not in accuracy_per_label.keys():
            accuracy_per_label[label_str] = {'true':0,'all':0,'smart_true':0}
        if label_str == pred_str:
            accuracy_per_label[label_str]['true'] += 1
        if label_smart_body == pred_smart_body:
            accuracy_per_label[label_str]['smart_true'] += 1
        accuracy_per_label[label_str]['all'] += 1

    results_str += "----------------------------------------------------\n"
    average_success_rate = 0
    smart_mapping_average_success_rate = 0
    for key in accuracy_per_label.keys():
        average_success_rate += float(accuracy_per_label[key]['true'])/accuracy_per_label[key]['all']
        smart_mapping_average_success_rate += float(accuracy_per_label[key]['smart_true'])/accuracy_per_label[key]['all']
        results_str += '{}\t: success rate {}\t,success to shorten labels set {}\tfrom {} trials \n'.format(key,float(accuracy_per_label[key]['true'])/accuracy_per_label[key]['all'],float(accuracy_per_label[key]['smart_true'])/accuracy_per_label[key]['all'],accuracy_per_label[key]['all'])

    results_str+= 'average succuess rate {}\t average success rate of smart labels {}\n'.format(average_success_rate/len(accuracy_per_label.keys()),smart_mapping_average_success_rate/len(accuracy_per_label.keys()))

    with open(compare_file_path, 'w') as comparison_file:
        comparison_file.write(results_str)
