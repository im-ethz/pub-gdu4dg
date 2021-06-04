import math
import os

import pandas
import scipy.io as io

__all__ = ["DataReader"]


class DataReader:
    """Data manipulation class wrapper"""

    # Remap snomed code duplicates
    snomed_mapping = {"59118001": "713427006", "63593006": "284470004", "17338001": "427172004", }

    snomed_mapping_remap = {"59118001": "713427006", "63593006": "284470004", "17338001": "427172004",
                            "195042002": "164947007", "251173003": "284470004", "251268003": "10370003",
                            "233917008": "164947007", "251170000": "284470004", "426749004": "164889003",
                            "204384007": "164947007", "82226007": "698252002", "54016002": "164947007",
                            "282825002": "164889003", "425419005": "39732003", "445211001": "47665007",
                            "89792004": "47665007", "251266004": "10370003", "164873001": "39732003",
                            "195080001": "164889003", "251182009": "427172004", "164865005": "164917005",
                            "11157007": "17338001", "164884008": "17338001", "251168009": "63593006", }

    snomed_conditional_mapping = {"270492004": "164947007"}

    # Remap sex categories description
    sex_mapping = {"f": "female", "female": "female", "m": "male", "male": "male", }

    @staticmethod
    def read_table(path="tables/"):
        """Function reads the table with ALL the diagnosis codes."""
        table = pandas.read_csv(f'{path}Dx_map.csv', usecols=[1, 2])
        table.columns = ["Code", "Label"]
        return dict(zip(table.Code, table.Label))

    @staticmethod
    def read_sample(file_name):
        """Reads mat data as np array"""
        if os.path.exists(file_name):
            return io.loadmat(os.path.join(file_name))["val"]
        else:
            return None

    @staticmethod
    def read_header(file_name, snomed_table, from_file=True, remap=False):
        """Function saves information about the patient from header file in this order:
        sampling frequency, length of the signal, voltage resolution, age, sex, list of diagnostic labels both in
        SNOMED and abbreviations (sepate lists)"""

        sampling_frequency, resolution, age, sex, snomed_codes = [], [], [], [], []

        def string_to_float(input_string):
            """Converts string to floating point number"""
            try:
                value = float(input_string)
            except ValueError:
                value = None

            if math.isnan(value):
                return None
            else:
                return value

        if from_file:
            lines = []
            with open(file_name, "r") as file:
                for line_idx, line in enumerate(file):
                    lines.append(line)
        else:
            lines = file_name

        # Read line 15 in header file and parse string with labels

        snomed_codes = []
        resolution = []
        age = None
        sex = None
        for line_idx, line in enumerate(lines):
            if line_idx == 0:
                sampling_frequency = float(line.split(" ")[2])
                continue
            if 1 <= line_idx <= 12:
                resolution.append(string_to_float(line.split(" ")[2].replace("/mV", "").replace("/mv", "")))
                continue
            if line.startswith('#Age'):
                age = string_to_float(line.replace("#Age:", "").replace("#Age", "").rstrip("\n").strip())
                continue
            if line.startswith('#Sex'):
                sex = line.replace("#Sex:", "").replace("#Sex", "").rstrip("\n").strip().lower()
                if sex not in DataReader.sex_mapping:
                    sex = None
                else:
                    sex = DataReader.sex_mapping[sex]
                continue
            if line.startswith('#Dx'):
                if from_file:
                    snomed_codes = line.replace("#Dx:", "").replace("#Dx", "").rstrip("\n").strip().split(",")
                    if remap:
                        snomed_codes = [DataReader.snomed_mapping_remap.get(item, item) for item in snomed_codes]
                    else:
                        snomed_codes = [DataReader.snomed_mapping.get(item, item) for item in snomed_codes]
                continue

        if remap:
            for code in DataReader.snomed_conditional_mapping:
                if code in snomed_codes:
                    snomed_codes.append(DataReader.snomed_conditional_mapping[code])

            # Remove duplicates
            snomed_codes = list(set(snomed_codes))

        return sampling_frequency, resolution, age, sex, snomed_codes

    @staticmethod
    def read_header_keep_snomed(file_name, snomed_table, from_file=True, remap=False):
        """Function saves information about the patient from header file in this order:
        sampling frequency, length of the signal, voltage resolution, age, sex, list of diagnostic labels both in
        SNOMED and abbreviations (sepate lists)"""

        sampling_frequency, resolution, age, sex, snomed_codes = [], [], [], [], []

        def string_to_float(input_string):
            """Converts string to floating point number"""
            try:
                value = float(input_string)
            except ValueError:
                value = None

            if math.isnan(value):
                return None
            else:
                return value

        if from_file:
            lines = []
            with open(file_name, "r") as file:
                for line_idx, line in enumerate(file):
                    lines.append(line)
        else:
            lines = file_name

        # Read line 15 in header file and parse string with labels

        snomed_codes = []
        resolution = []
        age = None
        sex = None
        for line_idx, line in enumerate(lines):
            if line_idx == 0:
                sampling_frequency = float(line.split(" ")[2])
                continue
            if 1 <= line_idx <= 12:
                resolution.append(string_to_float(line.split(" ")[2].replace("/mV", "").replace("/mv", "")))
                continue
            if line.startswith('#Age'):
                age = string_to_float(line.replace("#Age:", "").replace("#Age", "").rstrip("\n").strip())
                continue
            if line.startswith('#Sex'):
                sex = line.replace("#Sex:", "").replace("#Sex", "").rstrip("\n").strip().lower()
                if sex not in DataReader.sex_mapping:
                    sex = None
                else:
                    sex = DataReader.sex_mapping[sex]
                continue
            if line.startswith('#Dx'):
                if 1:
                    snomed_codes = line.replace("#Dx:", "").replace("#Dx", "").rstrip("\n").strip().split(",")
                    if remap:
                        snomed_codes = [DataReader.snomed_mapping_remap.get(item, item) for item in snomed_codes]
                    else:
                        snomed_codes = [DataReader.snomed_mapping.get(item, item) for item in snomed_codes]
                continue

        if remap:
            for code in DataReader.snomed_conditional_mapping:
                if code in snomed_codes:
                    snomed_codes.append(DataReader.snomed_conditional_mapping[code])

            # Remove duplicates
            snomed_codes = list(set(snomed_codes))

        return sampling_frequency, resolution, age, sex, snomed_codes

    @staticmethod
    def get_label_maps(path="tables/"):
        """Function reads the table with ALL the diagnosis codes."""

        reader = pandas.read_csv(f'{path}dx_mapping_scored.csv', usecols=[1, 2])
        reader.columns = ["Code", "Labels"]

        snomed_codes, labels = reader.Code, reader.Labels

        label_mapping = {str(code): label for code, label in zip(snomed_codes, labels) if
                         str(code) not in DataReader.snomed_mapping}

        idx_mapping = {key: idx for idx, key in enumerate(label_mapping)}

        return idx_mapping, label_mapping
