from keras.models import load_model
model_file_name = "model"
import h5py
from PhonesSet import PhonesSet
from DataSet import DataSet

input_vec_size = 9
num_of_features = 39
num_of_labels = PhonesSet.get_num_labels()
LOAD_FROM_FILE = True
labels_file_name = "labels_vecs_to_orig.h5"
inputs_file_name = "inputs_vecs_to_orig.h5"
inputs_dataset_name = "inputs"
labels_dataset_name = "labels"
if not LOAD_FROM_FILE:
    buckeye_dataset = DataSet("C:\Users\user\Documents\quick",input_vec_size)#/home/bokobzt/Datasets/buckeye/short",#""/home/bokobzt/Datasets/buckeye/onespeaker",input_vec_size)#"C:\Users\user\Documents\quick","C:\Users\user\Documents\\test_buckeye"///home/bokobzt/Datasets/buckeye/speech
    #C:\Users\user\Documents\10speakers
    xVal,yFit = buckeye_dataset.get_speakers_data(num_of_speakers=41)

    with h5py.File(inputs_file_name, 'w') as hf:
        hf.create_dataset(inputs_dataset_name,  data=inputs)

    with h5py.File(labels_file_name, 'w') as hf:
        hf.create_dataset(labels_dataset_name, data=labels)
else:
    with h5py.File(inputs_file_name, 'r') as hf:
        inputs = hf[inputs_dataset_name][:]

    with h5py.File(labels_file_name, 'r') as hf:
        labels = hf[labels_dataset_name][:]



model = load_model(model_file_name)
yFit = model.predict(xVal, batch_size=10, verbose=1)
print()
print(yFit)