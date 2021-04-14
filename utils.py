import pickle
import os

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