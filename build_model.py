from model import MyModel
import Utils
from TimitDataSet import TimitDataSet
from PhonesSet import PhonesSet
import os
max_data_len_for_writing = 500000
import config
import numpy as np
import time
import threading
from SBStompClient import StompClient
from WaveStreamer import play_wav
##################################################################################################

input_vec_size = config.input_vec_size
num_of_labels = PhonesSet.get_num_labels()
load_from_file = config.load_from_file
model = None
labels_file_name = config.labels_file_name
inputs_file_name = config.inputs_file_name
inputs_dataset_name = "inputs"
labels_dataset_name = "labels"
if os.name == 'nt':
    dir_path = config.input_dir_windows
    validation_path = config.path_for_validation_windows
else:
    dir_path = config.input_dir_linux
    validation_path = config.path_for_validation_linux
if config.learn_model:
    if not load_from_file:
            #os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda,floatX=float32"
        buckeye_dataset = TimitDataSet(dir_path, input_vec_size, overlap=config.inputes_overlap)#/home/bokobzt/Datasets/buckeye/short",#""/home/bokobzt/Datasets/buckeye/onespeaker",input_vec_size)#"C:\Users\user\Documents\quick","C:\Users\user\Documents\\test_buckeye"///home/bokobzt/Datasets/buckeye/speech
        #C:\Users\user\Documents\10speakers
        inputs,labels = buckeye_dataset.get_dirs_data(num_of_dirs=41)

        #new#Utils.save_data_to_files(inputs_file_name,inputs)
        #with h5py.File(inputs_file_name, 'w') as hf:
        #    hf.create_dataset(inputs_dataset_name,  data=inputs)
        #new#Utils.save_data_to_files(labels_file_name, labels)
        #with h5py.File(labels_file_name, 'w') as hf:
        #    hf.create_dataset(labels_dataset_name, data=labels)
    else:
	pass
        #with h5py.File(inputs_file_name, 'r') as hf:
        #    inputs = hf[inputs_dataset_name][:]
        #new#inputs = Utils.load_data_from_files(inputs_file_name)
        #with h5py.File(labels_file_name, 'r') as hf:
        #    labels = hf[labels_dataset_name][:]
        #new#labels = Utils.load_data_from_files(labels_file_name)

    print("finished loading data, saved to files")
    print("total num of inputes : " + str(len(inputs)))

    run_cnn = config.run_cnn
    many_to_many = config.many_to_many


    if not run_cnn:
        num_of_features = len(inputs[0][0])
        model = MyModel(input_len=input_vec_size, num_features=num_of_features, num_labels=num_of_labels)
        if many_to_many:
            model.build_model("GRU",[350,300,250],[0.05,0.05,0.05])
            model.fit(inputs,labels,model_path="model",early_stopping_patience=40,val_percentage=0.1,batch_size=50,num_epochs=100)

        else:
            labels_many_to_one = Utils.get_many_to_one_labels(labels,num_of_labels)
            model.build_model("GRU_1",[350,300,250],[0.05,0.05,0.05])
            model.fit(inputs,labels_many_to_one,model_path="model",early_stopping_patience=40,val_percentage=0.1,batch_size=50,num_epochs=100)

    else:
        inputs_transpose,labels_many_to_one,input_len,num_of_features = Utils.convert_to_cnn_inputs_and_labels(inputs,labels)
        model = MyModel(input_len=input_len, num_features=num_of_features, num_labels=num_of_labels)
        weights_path = None
        if config.continue_training:
            weights_path = config.file_name
        model.build_model("CONV1", [], [],weights_path=weights_path)
        model.fit(inputs_transpose, labels_many_to_one, model_path=config.file_name, early_stopping_patience=10, val_percentage=0.2, batch_size=50,
                  num_epochs=120)


def send_visemes_to_smartbody(model, data_list):
    print('----start send to smart body')
    start = time.time()
    predictions2 = None
    last_prediction = None
    stompClient = StompClient()
    end = time.time()
    print('intialization took  {}'.format(end - start))
    # time.sleep(0.02)
    for data_sample in data_list:
        time.sleep(0.085)#0.09

        prediction = model.predict(np.asarray([data_sample]))
        pred_str = PhonesSet.from_vector_to_label(prediction)

        if pred_str != last_prediction:
            last_prediction = pred_str
            stompClient.send_viseme_command(pred_str)

    print('test')


if config.load_model_and_validate:
    X,Y = Utils.load_data_for_validation(validation_path, input_vec_size)
    inputs_transpose, labels_many_to_one, input_len, num_of_features = Utils.convert_to_cnn_inputs_and_labels(X, Y)
    if model is None:
        model = MyModel(input_len=input_len, num_features=num_of_features, num_labels=num_of_labels)
        model_file = config.file_name
        model.build_model("CONV1", [], [],weights_path=model_file)

    predictions = model.predict(inputs_transpose)
    Utils.print_compare(labels_many_to_one,predictions)

    thread = threading.Thread(target=send_visemes_to_smartbody, args=(model,inputs_transpose))
    thread.daemon = True  # Daemonize thread
    thread.start()

    file_path = 'C:\\Users\\user\\Desktop\\timit-code\\data\\dr1\\si648_new.wav'
    #thread2= threading.Thread(target=play_wav,args=(file_path,))
    #thread2.daemon = True  # Daemonize thread
    play_wav(file_path)

    #thread2.start()







