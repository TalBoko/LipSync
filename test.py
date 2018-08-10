import os
import fnmatch
pathes = []
for root, dirnames, filenames in os.walk('/home/jkeshet/tal/timit/train'):
        for filename in fnmatch.filter(filenames, '*.wav'):
                new_name = filename.split('.')[0]+'new.wav'
                new_path = os.path.join(root,new_name)
                old_path = os.path.join(root,filename)
                os.system('sox {} {} '.format(old_path,new_path))


for root, dirnames, filenames in os.walk('/home/jkeshet/tal/timit'):
	for filename in fnmatch.filter(filenames, '*new.wav'):
		new_path = os.path.join(root,filename)
		print(new_path)