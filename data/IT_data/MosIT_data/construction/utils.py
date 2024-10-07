import json
import os
import errno
import torch
import random
import numpy as np


class Logger(object):

    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(str(msg) + '\n')
        self.log_file.flush()
        # print(msg)


def load_json(path):
    with open(path, 'r') as f:
        x = json.load(f)
    return x


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU random seed
    torch.cuda.manual_seed(seed)  # GPU random seed
    torch.backends.cudnn.benchmark = True


def get_demonstration(file_path):
    demonstrations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        _temp = ''
        for line in lines:
            if line.startswith('-----------------'):
                demonstrations.append(_temp)
                _temp = ''
            else:
                _temp += line
        demonstrations.append(_temp)
    return demonstrations


def get_existing_states(directory_path):
    subdirectories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    if not subdirectories:
        return [], None
    else:
        subdirectories.sort(key=lambda x: os.path.getctime(os.path.join(directory_path, x)), reverse=True)

        newest_folder = subdirectories[0]
        with open(os.path.join(directory_path, newest_folder, 'log.txt'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
    return [l.strip() for l in lines], newest_folder


def load_topics(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        res = []
        for line in f.readlines():
            res.append(line.strip())
    return res


if __name__ == '__main__':
    file_path = './checkpoints'
    lines, n_dir = get_existing_states(file_path)
    print(lines)
    print(n_dir)