# LipSync
lip sync with SmartBody

# python version
2.7.13 using Anaconda2

# code detailes:

### timit_client.py
class for handeling the reading of all the files in timit dataset.

### TimitDataSet.py 
class for loading learning data from timit dataset, in the required structure for learning.

### PhonesSet.py - 
handle all the optional learning phoneme sets, and the convertion to Smart body\'s visemes set.

### WavReader.py
Read the information from the 'שה file and extract the desired properties,
according to the given configuration

### SBStompClient.py
ActiveMQ client, support 2 type of messaging to SmartBody - visemes & bml(face animation) commands.

### model_stomp_client.py
get model and inputes, 
convert inputes to visemes by the given model and send the visemes to ActiveMQ.

### WaveStreamer.py
play a given wav file

### lip_synchronization.py
get a wav file, create model inputes.
play the audio while sending the appropriate visemes to the SmartBody Avatar.

### SpeechFromMicrophone.py
code example - recording speech from microphone.

### validate_model.py
class for model validation 

### model.py
build and train NN models

### Utils.py
global util
