from SBStompClient import StompClient
import Utils
from model import MyModel
import model_config
from PhonesSet import PhonesSet
import numpy as np
import time
'''
stompClient = StompClient()
validation_path = model_config.path_for_validation_windows
X,Y = Utils.load_data_for_validation(validation_path, model_config.input_vec_size)
inputs_transpose, labels_many_to_one, input_len, num_of_features = Utils.convert_to_cnn_inputs_and_labels(X, Y)
num_of_labels = PhonesSet.get_num_labels()
model = MyModel(input_len=input_len, num_features=num_of_features, num_labels=num_of_labels)
model_file = model_config.file_name
model.build_model("CONV1", [], [],weights_path=model_file)

predictions = model.predict(inputs_transpose)
Utils.print_compare(labels_many_to_one,predictions)
predictions2 = np.asarray([])
for data_sample in inputs_transpose:
    prediction = model.predict(np.asarray([data_sample]))
    predictions2 = np.append(predictions2, prediction, axis=0)

print('test')
'''

def send_visemes_to_smartbody(model, data_list):
    print('----start send to smart body')
    start = time.time()
    last_prediction = None
    stomp_client = StompClient()
    end = time.time()
    
    print('intialization took  {}'.format(end - start))
  
    for data_sample in data_list:
        start = time.time()
        time.sleep(0.0865)

        prediction = model.predict(np.asarray([data_sample]))
        pred_str = PhonesSet.from_vector_to_label(prediction)

        if pred_str != last_prediction or pred_str == 'sil':
            last_prediction = pred_str
            stomp_client.send_viseme_command(pred_str)
              
        end = time.time()
        duration = end - start
        if duration < 0.09:
            time.sleep(0.09-duration)

    print('finished sending visemes')
