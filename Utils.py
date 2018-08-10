import numpy as np
from DataSet import DataSet
from PhonesSet import PhonesSet
import os
max_data_len_for_writing = 500000

def get_many_to_one_labels(labels,number_of_labels):
    new_labels = []
    for tags in labels:
        #print(tags)
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
    for i in range(0,len(data),max_data_len_for_writing):
        file_name = base_file_name+ str(file_index)
        np.save(file_name,data[:i+max_data_len_for_writing])
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


def convert_to_cnn_inputs_and_labels(inputs,labels):
    inputs_transposed = []
    for input_seq in inputs:
        inputs_transposed += [input_seq.T]

    inputs_transposed = np.asarray(inputs_transposed)
    labels_many_to_one = get_many_to_one_labels(labels,PhonesSet.get_num_labels())
    num_of_features = len(inputs_transposed[0][0])
    input_len = len(inputs_transposed[0])
    return inputs_transposed,labels_many_to_one,input_len,num_of_features

def load_data_for_validation(validation_path, input_vec_size):
    dataset = DataSet(validation_path, input_vec_size, overlap=0)
    X,Y = dataset.get_dirs_data(8)

    return X,Y

def print_compare(labels, predictions):
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



    with open('test_model.txt', 'w') as comparison_file:
        comparison_file.write(results_str)
