from model import MyModel
import Utils
from PhonesSet import PhonesSet
import os
import model_config

batch_size = 50
validation_perc = 0.2
num_epochs = 120


def build_and_train_conv1_model(inputs, labels, num_of_labels, do_train=True, validation_data=None):
    inputs_transpose, labels_many_to_one, input_len, num_of_features = Utils.convert_to_cnn_inputs_and_labels(inputs,
                                                                                                              labels)
    model = MyModel(input_len=input_len, num_features=num_of_features, num_labels=num_of_labels)
    weights_path = None
    if model_config.continue_train_existing_model:
        weights_path = model_config.model_file_name
    model.build_model("CONV1", [], [], weights_path=weights_path)
    if do_train:
        if validation_data is not None:
            val_inputs_transpose, val_labels_many_to_one, input_len, num_of_features = Utils.convert_to_cnn_inputs_and_labels(
                validation_data[0],
                validation_data[1])
            model.fit(inputs_transpose, labels_many_to_one, model_path=model_config.model_file_name,val_percentage=validation_perc,
                      early_stopping_patience=10, validation_data=(val_inputs_transpose,val_labels_many_to_one), batch_size=batch_size, num_epochs=num_epochs)
        else:
            model.fit(inputs_transpose, labels_many_to_one, model_path=model_config.model_file_name, early_stopping_patience=10,
                  val_percentage=validation_perc, batch_size=batch_size, num_epochs=num_epochs)
    return model


def build_and_train_gru_model(inputs, labels, num_of_labels, do_train=True):
    num_of_features = len(inputs[0][0])
    model = MyModel(input_len=model_config.input_vec_size, num_features=num_of_features, num_labels=num_of_labels)
    if model_config.continue_train_existing_model:
        weights_path = model_config.model_file_name
    if model_config.many_to_many:
        model.build_model("GRU", [350, 300, 250], [0.05, 0.05, 0.05], weights_path)
        if do_train:
            model.fit(inputs, labels, model_path="model", early_stopping_patience=40, val_percentage=validation_perc,
                      batch_size=batch_size, num_epochs=num_epochs)

    else:
        # NOTE: train - many inputs to one label
        labels_many_to_one = Utils.get_many_to_one_labels(labels, num_of_labels)
        model.build_model("GRU_1", [350, 300, 250], [0.05, 0.05, 0.05], weights_path)
        if do_train:
            model.fit(inputs, labels_many_to_one, model_path="model", early_stopping_patience=40, val_percentage=validation_perc,
                  batch_size=batch_size, num_epochs=num_epochs)
    return model

##################################################################################################

# NOTE: load training data

if os.name == 'nt':
    input_dir_path = model_config.input_dir_windows
    validation_path = model_config.path_for_validation_windows
else:
    input_dir_path = model_config.input_dir_linux
    validation_path = model_config.path_for_validation_linux

inputs, labels = Utils.load_training_data(input_dir_path)
inputs_val, labels_val = Utils.load_training_data(validation_path)
validation_data =(inputs_val, labels_val)
# NOTE: build model and train
num_of_labels = PhonesSet.get_num_labels()

if model_config.run_cnn:
    build_and_train_conv1_model(inputs, labels, num_of_labels, do_train=True, validation_data=validation_data)
else:
    build_and_train_gru_model(inputs, labels, num_of_labels)

print('build model successfully, model saved to file')





