input_vec_size = 9
input_overlap = 4
load_data_from_file = False
labels_file_name = 'labels_vecs_to_orig_'
inputs_file_name = 'inputs_vecs_to_orig_'

dataset = 'timit'

feature_type = 'MEL_SPECTOGRAM'
mapping_type = 'TO_VISEMES'
run_cnn = True
# relevant only for GRU model
many_to_many = False
input_dir_windows = 'C:\\Users\\user\\Desktop\\timit-code\\data'
input_dir_linux = '/local_data/bokobta/Datasets/timit/train/'
continue_train_existing_model = False
num_mels = 60

path_for_validation_windows  = 'C:\\Users\\user\\Desktop\\timit-code\\data2'
path_for_validation_linux  = '/local_data/bokobta/Datasets/timit/test/'
learn_model = False
load_model_and_validate = True
model_file_name = 'model'
