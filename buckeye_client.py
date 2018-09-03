import io
import glob
import os
from WavReader import *
from PhonesSet import *
import logging
import model_config

FRAME_SIZE=400
SHIFT_SIZE=160
FILES_RATE = 16000


class Phone(object):
    """A phone entry in the Buckeye Corpus.

    Parameters
    ----------
    seg : str
        Label for the phone (pseudo-ARPABET in the Buckeye Corpus).

    beg : float, optional
        Timestamp where the phone begins. Default is None.

    end : float, optional
        Timestamp where the phone ends. Default is None.

    Attributes
    ----------
    beg
    end
    dur

    """

    def __init__(self, seg, beg=None, end=None):
        self._seg = seg
        self._beg = beg
        self._end = end

    def __repr__(self):
        return 'Phone({}, {}, {})'.format(repr(self._seg), self._beg, self._end)

    def __str__(self):
        return '<Phone [{}] at {}>'.format(self._seg, self._beg)

    @property
    def seg(self):
        """Label for this phone (e.g., using ARPABET transcription)."""
        return self._seg

    @property
    def beg(self):
        """Timestamp where the phone begins."""
        return self._beg

    @property
    def end(self):
        """Timestamp where the phone ends."""
        return self._end

    @property
    def dur(self):
        """Duration of the phone."""
        try:
            return self._end - self._beg

        except TypeError:
            raise AttributeError('Duration is not available if beg and end '
                                 'are not numeric types')


class Speaker:

    def __init__(self,dir_path):
        self.dir_path = dir_path
        self.tracks= []
        self.num_skipped_files = 0
        self.config_logger()

    def config_logger(self):
        # create logger
        logging.basicConfig(filename='logger.log', level=logging.DEBUG)

    def get_corpus(self):
        logging.info("start loading speaker data :  {} ".format(self.dir_path))
        file_paths = glob.glob(self.dir_path+'*.wav')
        self.tracks_names_list =[os.path.basename(wave_file).split(".")[0] for wave_file in file_paths]
        #for track_name in self.tracks_names_list:
        #    self.tracks += [Track(self.dir_path,track_name)]
        self.mfcc_inputs= []
        self.labels = []
        for track_name in self.tracks_names_list:
            try:
                self.generate_labeled_data_from_track(self.dir_path,track_name)
            except Exception as e:
                logging.warning("skipping file of speaker " + self.dir_path)
                logging.error("exception : " + str(e))
                self.num_skipped_files += 1

        logging.info("skipped {} files of current speaker".format(str(self.num_skipped_files)))
        self.normalize_mfcc_data()
        return self.mfcc_inputs,self.labels

    def generate_labeled_data_from_track(self,dir_path, name):
        unknown_phonemes = []
        wave_file_path = os.path.join(dir_path, name + ".wav")
        phoneme_file_path = os.path.join(dir_path, name + ".phones")

        phones_file = io.open(phoneme_file_path, encoding='latin-1')
        phones = list(self.process_phones(phones_file))
        phones_file.close()

        # load wave frames and extract mfcc features
        ##################

        shift_len = 0.01
        feature_type_str = model_config.feature_type
        mfcc_type = Feature_Type[feature_type_str]
        if mfcc_type == Feature_Type.REAL_TIME:
            frame_len = 0.025
        else:
            frame_len=0.032

        wav_reader = WavReader(16000, frame_len=frame_len, shift_len=shift_len,mfcc_type=mfcc_type)
        track_mfccs = wav_reader.get_all_frame_features(wave_file_path, model_config.num_mels)
        ##################

        is_first_data=True
        phones_index = 0

        for frame_index in range(len(track_mfccs)):
            frame_start_time = frame_index * shift_len
            frame_end_time = frame_start_time + frame_len

            if frame_start_time > phones[phones_index].end:

                phones_index += 1
                if phones_index + 1 == len(phones):
                    print("break ignore last frames with label " + phones[phones_index].seg.encode('utf-8'))
                    print ("ignored " + str(len(track_mfccs)-frame_index) + " frames")
                    break

            if frame_end_time < phones[phones_index].end:
                phoneme = phones[phones_index].seg.encode('utf-8')

            else:  # frame_start_time >= self.phones[phones_index].beg and frame_end_time > phone.end:

                # more in current phoneme
                if phones[phones_index].end - frame_start_time >= frame_end_time - phones[phones_index].end:
                    phoneme = phones[phones_index].seg.encode('utf-8')
                    #self.labels.append()
                else:  # shoud labeled as next phone
                    #if phones_index + 1 < len(phones):
                    try:
                        phoneme = phones[phones_index + 1].seg.encode('utf-8')
                    except:
                        print("phoneme_index " + str(phones_index))
                        #print (phones[0:phones_index+1])
                    '''else:
                        print ("last skip")
                        print (frame_index)
                        print (len(track_mfccs))
                        phoneme = phones[phones_index].seg.encode('utf-8')'''

            phoneme_label = PhonesSet.get_mapping(phoneme)
            if is_first_data and (phoneme_label == PhonesSet.DEFAULT or phoneme_label == PhonesSet.get_mapping('SIL')):
                #skip first frames with unknown phone label or silence from beginning of recording
                continue
            #otherwise
            if is_first_data:
                is_first_data = False

            if phoneme_label == PhonesSet.DEFAULT or phoneme_label == PhonesSet.get_mapping('SIL'):
                unknown_phonemes += [(track_mfccs[frame_index],phoneme_label,phoneme)]
            else:
                if len(unknown_phonemes) > 4 :
                    #ignore all sequences of more then 3 frames of unnown or silence labels
                    print("-------ignored label {}-{} of {} frames len ".format(unknown_phonemes[0][1],unknown_phonemes[0][2],str(len(unknown_phonemes))))
                    unknown_phonemes = []

                elif len(unknown_phonemes) <= 4 :
                    if len(unknown_phonemes) > 0:
                        print("--------dont ignore label {}-{} of {} frames len ".format(unknown_phonemes[0][1],unknown_phonemes[0][2],str(len(unknown_phonemes))))
                    #if unknown phoneme is of lenth of less then 4 frames, don`t ignore it.
                    for frame in unknown_phonemes:
                        self.mfcc_inputs.append(frame[0])
                        self.labels.append(frame[1])

                    unknown_phonemes = []

                self.labels.append(phoneme_label)
                self.mfcc_inputs.append(track_mfccs[frame_index])

        assert(len(self.mfcc_inputs) == len(self.labels))

    def calc_normalization_properties(self):
        window_size = 10000
        sum_features_list = np.sum(self.mfcc_inputs[:window_size], axis=0)
        self.features_average = sum_features_list / window_size
        self.std_list = np.std(self.mfcc_inputs[:window_size], axis=0)

    def normalize_mfcc_data(self):
        self.calc_normalization_properties()
        normalized_mach_inputs = self.mfcc_inputs-self.features_average
        normalized_mach_inputs = normalized_mach_inputs/self.std_list
        self.mfcc_inputs = normalized_mach_inputs

    def process_phones(self, phones):
        """Yield Phone instances from a .phones file in the Buckeye Corpus.

        Parameters
        ----------
        phones : file-like
            Open file-like object created from a .phones file in the Buckeye
            Corpus.

        Yields
        ------
        Phone
            One Phone instance for each entry in the .phones file, in
            chronological order.

        """

        # skip the header
        line = phones.readline()

        while not line.startswith('#'):
            if line == '':
                raise EOFError

            line = phones.readline()

        line = phones.readline()

        # iterate over entries
        previous = 0.0
        while line != '':
            try:
                time, color, phone = line.split(None, 2)

                if '+1' in phone:
                    phone = phone.replace('+1', '')

                if ';' in phone:
                    phone = phone.split(';')[0]

                phone = phone.strip()

            except ValueError:
                if line == '\n':
                    line = phones.readline()
                    continue

                time, color = line.split()
                phone = None
                print("skipped unknown phoneme ...")



            time = float(time)
            if phone is not None:
                yield Phone(phone, previous, time)

            previous = time
            line = phones.readline()


