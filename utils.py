import pickle
import os
import time
import logging
import json


def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)


def start_time():
    return time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))


def write_log(filename, content):
    with open(filename, 'a') as f:
        f.write(content)


def get_logger():
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    return logger