from SBStompClient import StompClient
import Utils
from model import MyModel
import model_config
from PhonesSet import PhonesSet
import numpy as np
import time


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
