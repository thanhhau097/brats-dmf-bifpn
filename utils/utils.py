import argparse
import os
import json
import difflib
import zipfile


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

class JsonHandler(object):
    """

    """
    def default(self, o):
        if isinstance(o, np.int64): return int(o)
        raise TypeError

    def read_json_file(self, file):
        with open(file) as f:
            data = f.read()
        return json.loads(data)

    def dump_to_file(self, data, file):
        with open(file, 'w') as fp:
            json.dump(data, fp, default=self.default)

def write_log(file_path, content, mode='a'):
    with open(file_path, mode) as opt_file:
        opt_file.write(content + "\n")

def calculate_ac(str1, str2):
    """Calculate accuracy by char of 2 string"""

    total_letters = len(str1)
    ocr_letters = len(str2)
    if total_letters == 0 and ocr_letters == 0:
        acc_by_char = 1.0
        return acc_by_char
    diff = difflib.SequenceMatcher(None, str1, str2)
    correct_letters = 0
    for block in diff.get_matching_blocks():
        correct_letters = correct_letters + block[2]
    if ocr_letters == 0:
        acc_by_char = 0
    elif correct_letters == 0:
        acc_by_char = 0
    else:
        acc_1 = correct_letters / total_letters
        acc_2 = correct_letters / ocr_letters
        acc_by_char = 2 * (acc_1 * acc_2) / (acc_1 + acc_2)

    return float(acc_by_char)

def unzip(zip_file, to_folder):
    zip_ref = zipfile.ZipFile(zip_file, 'r')
    zip_ref.extractall(to_folder)
    zip_ref.close()

