import os
import sys
import glob
import shutil
import pickle as pkl

import numpy as np


def load_pickle(path, name):
    """Load pickle"""
    with open(path + name, 'rb') as f:
        return pkl.load(f, encoding='latin1')


class BGCFLogger:
    """log the output metrics"""

    def __init__(self, logname, now, foldername, copy):
        self.terminal = sys.stdout
        self.file = None

        path = os.path.join(foldername, logname, now)
        os.makedirs(path)

        if copy:
            filenames = glob.glob('*.py')
            for filename in filenames:
                shutil.copy(filename, path)

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=True, is_file=True):
        """Write log"""
        if '\r' in message:
            is_file = False

        if is_terminal:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file:
            self.file.write(message)
            self.file.flush()


def convert_item_id(item_list, num_user):
    """Convert the graph node id into item id"""
    return np.array(item_list) - num_user
