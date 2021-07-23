import csv
import numpy as np


class Logger:
    """ Class that allows logging to an csv file. """
    def __init__(self, log_file_path, columns=None):
        self.log_path = log_file_path
        if columns is None:
            self.columns = ['file name', 'orig pred', 'mod pred', 'conv']
        else:
            self.columns = columns

        with open(self.log_path, 'w') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerow(columns)

    def append(self, value_list):
        with open(self.log_path, 'a') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerow(value_list)


def snr(x, x_hat):
    """ SNR computation according to https://github.com/coreyker/dnn-mgr/blob/master/utils/comp_ave_snr.py. """
    ign = 2048
    lng = min(x.shape[-1], x_hat.shape[-1])
    ratio = 20 * np.log10(np.linalg.norm(x[..., ign:lng - ign - 1]) /
                          np.linalg.norm(np.abs(x[..., ign:lng - ign - 1] - x_hat[..., ign:lng - ign - 1]) + 1e-12))
    return ratio
