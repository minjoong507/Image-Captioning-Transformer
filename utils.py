import pickle
import os
import time


def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)


def start_time():
    return time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))


def write_log(filename, content):
    with open(filename, 'a') as f:
        f.write(content)


