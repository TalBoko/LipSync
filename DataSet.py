import glob
import buckeye_client
import timit_client
from PhonesSet import *



class DataSet:
    def __init__(self, base_dir_path,input_size,overlap=0):
        self.base_dir_path = base_dir_path
        self.dirs_dirs_list = glob.glob(self.base_dir_path + "/*/")
        print self.dirs_dirs_list
        self.next_dir_index= 0
        self.bulk_index = 0
        self.num_of_features = 39
        self.input_size = input_size
        self.overlap = overlap

    def get_dirs_data(self,num_of_dirs):
        X,Y = self.load_next_dir_data()
        num_of_dirs = min(num_of_dirs,len(self.dirs_dirs_list))
        for i in range(num_of_dirs-1):
            try:
                print("continue to next dir, dir num "+ str(self.next_dir_index))
                X1,Y1 = self.load_next_dir_data()
                X = np.concatenate((X,X1))
                Y = np.concatenate((Y,Y1))
            except Exception as ex:
                print("failed to load data from dir num "+ str(self.next_dir_index))
                print(str(ex))

        return X,Y


    def load_next_dir_data(self):
        '''
        :param input_size: how many mfcc in each input vector
        :param overlap: overlap size between inputs
        :return: x_inputs,y_labels
        '''
        inputs = []
        outputs = []
        print("len(self.dirs_dirs_list) = "+str(len(self.dirs_dirs_list)))
        if self.next_dir_index < len(self.dirs_dirs_list) :
            if config.dataset == 'buckeye':
                print 'working on buckeye dataset'
                client = buckeye_client.Speaker(self.dirs_dirs_list[self.next_dir_index])
            elif config.dataset == 'timit':
                print 'working on timit dataset'
                client = timit_client.Timit(self.dirs_dirs_list[self.next_dir_index])

            features, labels = client.get_corpus()
            self.next_dir_index += 1
            print("frames " + str(len(features)))
            num_of_features = len(features)
            for i in range(0,num_of_features-self.input_size+1,self.input_size-self.overlap):
                inputs += [features[i:i + self.input_size]]
                outputs += [self.get_output_vector(labels[i:i + self.input_size])]

        return np.asarray(inputs),np.asarray(outputs)

    def get_output_vector(self, labels_list):
        assert len(labels_list) == self.input_size
        labels_vectors_list = []
        num_label_options = PhonesSet.get_num_labels()
        for label in labels_list:
            label_int_value = PhonesSet.get_label_index(label)
            label_vector = np.zeros(num_label_options)
            label_vector[label_int_value] = 1
            labels_vectors_list += [label_vector]
        try:
            for vec in labels_vectors_list:
                assert sum(vec) == 1 ,"invalid label"
        except Exception as e:
            print(labels_list)
            print(labels_vectors_list)
        return labels_vectors_list

