import model_config
import numpy as np
from enum import Enum

class Lables_type(Enum):
    TO_ORIG = 1
    TO_SHORTEN = 2
    TO_VISEMES = 3


class PhonesSet(object):
    mapping_type = Lables_type[model_config.mapping_type]

    DATASET = model_config.dataset
    DEFAULT = 'unknown'
    PHONES_MAPPING = {'ae': 'ah', 'ah': 'ah', 'ax': 'ah', #'ahn':'ah', 'ihn': 'ih',
                      'aa': 'aa', 'ao': 'ao', 'ey': 'eh', 'eh': 'eh',
                      'er': 'er', 'ih': 'ih', 'iy': 'ih', 'w': 'w',
                      'uw': 'w', 'uh': 'w', 'ow': 'ow', 'aw': 'aw', 'oy': 'oy',
                      'ay': 'ay', 'hh': 'h', 'h': 'h', 'r': 'r', 'l': 'l', 'el': 'l', 's': 'z', 'z': 'z',
                      'sh': 'sh', 'ch': 'sh', 'jh': 'sh', 'zh': 'sh', 'y': 'sh',
                      'th': 'th', 'dh': 'th', 'f': 'f', 'v': 'f', 'd': 'd', 'tq': 'd',
                      't': 'd', 'nx': 'd', 'en': 'd', 'n': 'd', 'ng': 'd', 'k': 'kg', 'g': 'kg',
                      'p': 'bmp', 'b': 'bmp', 'm': 'bmp', 'em': 'bmp', 'SIL': 'SIL'}

    PHONES_INDEX = {'ah': 0, 'aa': 1, 'ao': 2, 'eh': 3, 'er': 4, 'ih': 5, 'w': 6,
                    'ow': 7, 'aw': 8, 'oy': 9, 'ay': 10, 'h': 11, 'r': 12, 'l': 13, 'z': 14,
                    'sh': 15, 'th': 16, 'f': 17, 'd': 18, 'kg': 19, 'bmp': 20, 'SIL': 21, DEFAULT: 22}

    PHONES_MAPPING_TO_ORIG = {'ae': 'ae', 'ah': 'ah', 'ax': 'ax',
                              'aa': 'aa', 'ao': 'ao', 'ey': 'ey', 'eh': 'eh',
                              'er': 'er', 'ih': 'ih', 'iy': 'iy', 'w': 'w',
                              'uw': 'uw', 'uh': 'uh', 'ow': 'ow', 'aw': 'aw', 'oy': 'oy',
                              'ay': 'ay', 'h': 'h', 'r': 'r', 'l': 'l', 's': 's', 'z': 'z',
                              'sh': 'sh', 'ch': 'ch', 'jh': 'jh', 'zh': 'zh', 'y': 'y',
                              'th': 'th', 'dh': 'dh', 'f': 'f', 'v': 'v', 'd': 'd',
                              't': 't', 'n': 'n', 'ng': 'ng', 'k': 'k', 'g': 'g',
                              'p': 'p', 'b': 'b', 'm': 'm', 'en': 'en', 'el': 'el', 'hh': 'hh', 'nx': 'nx', 'tq': 'tq',
                              'em': 'em', 'SIL': 'SIL'}

    PHONES_INDEX_TO_ORIG = {'ae': 0, 'ah': 1, 'ax': 2,
                              'aa': 3, 'ao': 4, 'ey': 5, 'eh': 6,
                              'er': 7, 'ih': 8, 'iy': 9, 'w': 10,
                              'uw': 11, 'uh': 12, 'ow': 13, 'aw': 14, 'oy': 15,
                              'ay': 16, 'h': 17, 'r': 18, 'l': 19, 's': 20, 'z': 21,
                              'sh': 22, 'ch': 23, 'jh': 24, 'zh': 25, 'y': 26,
                              'th': 27, 'dh': 28, 'f': 29, 'v': 30, 'd': 31,
                              't': 32, 'n': 33, 'ng': 34, 'k': 35, 'g': 36,
                              'p': 37, 'b': 38, 'm': 39, 'en': 40, 'el': 41, 'hh': 42, 'nx': 43, 'tq': 44,
                              'em': 45, 'SIL': 46,DEFAULT:47}
    #need to fix buckeye because sil and default shouldent have phone index (not part of classification options)

    TIMIT_MAPPING_ORIG = {"aa":"aa","ae":"ae","ah":"ah","ao":"ao","aw":"aw","ax":"ax","axh":"axh","axr":"axr","ay":"ay","b":"b","bcl":"bcl","ch":"ch","d":"d","dcl":"dcl","dh":"dh","dx":"dx","eh":"eh","el":"el","em":"em","en":"en","eng":"eng","epi":"epi","er":"er","ey":"ey","f":"f","g":"g","gcl":"gcl","h#":"h#","hh":"hh","hv":"hv","ih":"ih","ix":"ix","iy":"iy","jh":"jh","k":"k","kcl":"kcl","l":"l","m":"m","n":"n","ng":"ng","nx":"nx","ow":"ow","oy":"oy","p":"p","pau":"pau","pcl":"pcl","q":"q","r":"r","s":"s","sh":"sh","t":"t","tcl":"tcl","th":"th","uh":"uh","uw":"uw","ux":"ux","v":"v","w":"w","y":"y","z":"z","zh":"zh"}

    TIMTIT_INDEX_ORIG = {"aa":0,"ae":1,"ah":2,"ao":3,"aw":4,"ax":5,"axh":6,"axr":7,"ay":8,"b":9,"bcl":10,"ch":11,"d":12,"dcl":13,"dh":14,"dx":15,"eh":16,"el":17,"em":18,"en":19,"eng":20,"epi":21,"er":22,"ey":23,"f":24,"g":25,"gcl":26,"h#":27,"hh":28,"hv":29,"ih":30,"ix":31,"iy":32,"jh":33,"k":34,"kcl":35,"l":36,"m":37,"n":38,"ng":39,"nx":40,"ow":41,"oy":42,"p":43,"pau":44,"pcl":45,"q":46,"r":47,"s":48,"sh":49,"t":50,"tcl":51,"th":52,"uh":53,"uw":54,"ux":55,"v":56,"w":57,"y":58,"z":59,"zh":60}

    TIMIT_MAPPING_SHORTEN = {'em': 'm', 'ch': 'ch', 'ix': 'ih', 'tcl': 'pau', 'ae': 'ae', 'iy': 'iy', 'th': 'th', 'axr': 'er', 'pcl': 'pau', 'dh': 'dh', 'kcl': 'pau', 'hv': 'hh', 'hh': 'hh', 'dx': 'dx', 'd': 'd', 'b': 'b', 'ux': 'uw', 'f': 'f', 'uw': 'uw', 'l': 'l', 'n': 'n', 'p': 'p', 'r': 'r', 'uh': 'uh', 'v': 'v', 'z': 'z', 'aa': 'aa', 'el': 'l', 'en': 'n', 'zh': 'sh', 'eh': 'eh', 'ah': 'ah', 'ao': 'aa', 'ih': 'ih', 'ey': 'ey', 'aw': 'aw', 'h#': 'pau', 'ay': 'ay', 'ax': 'ah', 'er': 'er', 'pau': 'pau', 'eng': 'ng', 'gcl': 'pau', 'ng': 'ng', 'nx': 'n', 't': 't', 'dcl': 'pau', 'oy': 'oy', 'ow': 'ow', 'jh': 'jh', 'bcl': 'pau', 'g': 'g', 'k': 'k', 'm': 'm', 'q': 'pau', 's': 's', 'sh': 'sh', 'w': 'w', 'epi': 'pau', 'y': 'y', 'axh': 'ah', 'ax-h': 'ah'}

    TIMIT_INDEX_SHORTEN = {'aa': 0, 'iy': 1, 'ch': 2, 'ae': 3, 'eh': 4, 'ah': 5, 'ih': 6, 'ey': 7, 'aw': 8, 'ay': 9, 'er': 10, 'pau': 11, 'ng': 12, 'r': 32, 'th': 14, 'uh': 33, 'w': 34, 'dh': 17, 'y': 36, 'hh': 19, 'jh': 20, 'dx': 21, 'b': 22, 'd': 23, 'g': 24, 'f': 25, 'uw': 26, 'm': 27, 'l': 28, 'n': 29, 'p': 30, 's': 31, 'sh': 13, 't': 15, 'oy': 16, 'v': 35, 'ow': 18, 'z': 37, 'k': 38}

    PHONEME_TO_VISEME = {'aa': 'aa', 'iy': 'ih', 'ch': 'sh', 'ae': 'ah', 'eh': 'eh', 'ah': 'ah', 'ih': 'ih', 'ey': 'eh', 'aw': 'aw', 'ay': 'ay', 'er': 'er', 'pau': 'sil', 'ng': 'd', 'r': 'r', 'th': 'th', 't': 'd', 'w': 'w', 'dh': 'th', 'y': 'sh', 'hh': 'h', 'jh': 'sh', 'dx': 't', 'b': 'bmp', 'd': 'd', 'g': 'kg', 'f': 'f', 'uw': 'w', 'm': 'bmp', 'l': 'l', 'n': 'd', 'p': 'bmp', 's': 'z', 'sh': 'sh', 'uh': 'w', 'oy': 'oy', 'v': 'f', 'ow': 'ow', 'z': 'z', 'k': 'kg'}

    VISEMES_INDEX = {'aa': 0, 'eh': 1, 'ah': 2, 'ih': 3, 'aw': 4, 'ay': 5, 'er': 6, 'sh': 7, 'th': 8, 'sil': 9, 'w': 10, 'bmp': 11, 'kg': 12, 'd': 13, 'f': 14, 'h': 15, 'l': 16, 'r': 17, 't': 18, 'oy': 19, 'ow': 20, 'z': 21}
    #supose to ignore q phone in Lee & hon shorten mappping



    # i added ihn as ih  - because may be an mistake in some files, swane for ahn
    @staticmethod
    def get_mapping(label):
        if PhonesSet.DATASET == 'buckeye':
            if PhonesSet.mapping_type == Lables_type.TO_ORIG:
                return PhonesSet.PHONES_MAPPING_TO_ORIG.get(label, PhonesSet.DEFAULT)
            elif PhonesSet.mapping_type == Lables_type.TO_SHORTEN:
                return PhonesSet.PHONES_MAPPING.get(label,PhonesSet.DEFAULT)

        elif PhonesSet.DATASET == 'timit':
            if PhonesSet.mapping_type == Lables_type.TO_ORIG:
                return PhonesSet.TIMIT_MAPPING_ORIG.get(label, PhonesSet.DEFAULT)
            elif PhonesSet.mapping_type == Lables_type.TO_SHORTEN:
                return PhonesSet.TIMIT_MAPPING_SHORTEN.get(label,PhonesSet.DEFAULT)
            elif PhonesSet.mapping_type == Lables_type.TO_VISEMES:
                timit_mapping = PhonesSet.TIMIT_MAPPING_SHORTEN.get(label, PhonesSet.DEFAULT)
                return PhonesSet.PHONEME_TO_VISEME.get(timit_mapping,PhonesSet.DEFAULT)


    @staticmethod
    def get_num_labels():
        if PhonesSet.DATASET == 'buckeye':
            if PhonesSet.mapping_type == Lables_type.TO_ORIG:
                return len(PhonesSet.PHONES_INDEX_TO_ORIG.keys())
            elif PhonesSet.mapping_type == Lables_type.TO_SHORTEN:
                return len(PhonesSet.PHONES_INDEX.keys())

        elif PhonesSet.DATASET == 'timit':
            if PhonesSet.mapping_type == Lables_type.TO_ORIG:
                return len(PhonesSet.TIMTIT_INDEX_ORIG.keys())
            elif PhonesSet.mapping_type == Lables_type.TO_SHORTEN:
                return len(PhonesSet.TIMIT_INDEX_SHORTEN.keys())
            elif PhonesSet.mapping_type == Lables_type.TO_VISEMES:
                return len(PhonesSet.VISEMES_INDEX.keys())

    @staticmethod
    def get_label_index(label):
        if PhonesSet.DATASET == 'buckeye':
            if PhonesSet.mapping_type == Lables_type.TO_ORIG:
                return PhonesSet.PHONES_INDEX_TO_ORIG.get(label)
            elif PhonesSet.mapping_type == Lables_type.TO_SHORTEN:
                return PhonesSet.PHONES_INDEX.get(label)

        if PhonesSet.DATASET == 'timit':
            if PhonesSet.mapping_type == Lables_type.TO_ORIG:
                return PhonesSet.TIMTIT_INDEX_ORIG.get(label)
            elif PhonesSet.mapping_type == Lables_type.TO_SHORTEN:
                return PhonesSet.TIMIT_INDEX_SHORTEN.get(label)
            elif PhonesSet.mapping_type == Lables_type.TO_VISEMES:
                return PhonesSet.VISEMES_INDEX.get(label)


    @staticmethod
    def from_vector_to_label(vec):
        vec = np.asarray(vec)
        index = np.argmax(vec)
        if PhonesSet.DATASET == 'buckeye':
            if PhonesSet.mapping_type == Lables_type.TO_ORIG:
                return PhonesSet.PHONES_INDEX_TO_ORIG.keys()[PhonesSet.PHONES_INDEX_TO_ORIG.values().index(index)]
            elif PhonesSet.mapping_type == Lables_type.TO_SHORTEN:
                return PhonesSet.PHONES_INDEX.keys()[PhonesSet.PHONES_INDEX.values().index(index)]

        if PhonesSet.DATASET == 'timit':
            if PhonesSet.mapping_type == Lables_type.TO_ORIG:
                return PhonesSet.TIMTIT_INDEX_ORIG.keys()[PhonesSet.TIMTIT_INDEX_ORIG.values().index(index)]
            elif PhonesSet.mapping_type == Lables_type.TO_SHORTEN:
                return PhonesSet.TIMIT_INDEX_SHORTEN.keys()[PhonesSet.TIMIT_INDEX_SHORTEN.values().index(index)]
            elif PhonesSet.mapping_type == Lables_type.TO_VISEMES:
                return PhonesSet.VISEMES_INDEX.keys()[PhonesSet.VISEMES_INDEX.values().index(index)]
'''
TIMIT : 
SYMBOL    DESCRIPTION
                ------    -----------
                  pau     pause
                  epi     epenthetic silence
                  h#      begin/end marker (non-speech events)
                  1       primary stress marker
                  2       secondary stress marker
'''