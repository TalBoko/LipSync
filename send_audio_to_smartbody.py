from SBStompClient import StompClient
import Utils
from model import MyModel
import config
import PhonesSet
import numpy as np

stompClient = StompClient()
validation_path = config.path_for_validation_windows
X,Y = Utils.load_data_for_validation(validation_path, config.input_vec_size)
inputs_transpose, labels_many_to_one, input_len, num_of_features = Utils.convert_to_cnn_inputs_and_labels(X, Y)
num_of_labels = PhonesSet.get_num_labels()
model = MyModel(input_len=input_len, num_features=num_of_features, num_labels=num_of_labels)
model_file = config.file_name
model.build_model("CONV1", [], [],weights_path=model_file)

predictions = model.predict(inputs_transpose)
Utils.print_compare(labels_many_to_one,predictions)
predictions2 = np.asarray([])
for data_sample in inputs_transpose:
    prediction = model.predict(np.asarray([data_sample]))
    predictions2 = np.append(predictions2, prediction, axis=0)

print('test')