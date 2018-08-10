import Utils
import os
import model_config
from PhonesSet import PhonesSet
from model import MyModel

# NOTE - only for cnn model

if os.name == 'nt':
    validation_path = model_config.path_for_validation_windows
else:
    validation_path = model_config.path_for_validation_linux

num_of_labels = PhonesSet.get_num_labels()
inputs, labels = Utils.load_data_for_validation(validation_path, model_config.input_vec_size)

inputs_transpose, labels_many_to_one, input_len, num_of_features = Utils.convert_to_cnn_inputs_and_labels(inputs, labels)

model = MyModel(input_len=input_len, num_features=num_of_features, num_labels=num_of_labels)
model_file = model_config.model_file_name
model.build_model("CONV1", [], [], weights_path=model_file)

predictions = model.predict(inputs_transpose)
Utils.print_compare('test_model.txt', labels_many_to_one, predictions)