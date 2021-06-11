from ECG_utils import transforms
from ECG_utils.datareader import DataReader
from ECG_utils.utils import load_weights


class Config:
    REMAP = False

    INPUT_SIZE = 12

    HASH_TABLE = DataReader.get_label_maps(path="tables/")
    SNOMED_TABLE = DataReader.read_table(path="tables/")
    SNOMED_24_ORDERD_LIST = list(HASH_TABLE[0].keys())

    output_sampling = 125
    std = 0.2

    TRANSFORM_DATA_TRAIN = transforms.Compose(
        [transforms.Resample(output_sampling=output_sampling), transforms.ZScore(mean=0, std=std),
         transforms.RandomAmplifier(p=0.8, max_multiplier=0.2), transforms.RandomStretch(p=0.8, max_stretch=0.1), ])

    TRANSFORM_DATA_VALID = transforms.Compose(
        [transforms.Resample(output_sampling=output_sampling), transforms.ZScore(mean=0, std=std), ])

    TRANSFORM_LBL = transforms.SnomedToOneHot()

    loaded_weigths = load_weights('weights.csv', SNOMED_24_ORDERD_LIST)
