# plain.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy as np


__all__ = ["data_length", "convert_data"]


def data_length(line):
    return len(line.split())

def add_noise(line):
    line = line.split()
    if len(line) < 3:
        return line
    else:
        length = len(line)
        index = [np.random.randint(0,length-1) for i in range(length/2)]
        for idx in index:
            line[idx], line[idx+1] = line[idx+1], line[idx]
        return line

def convert_data(data, voc, unk="UNK", eos="<eos>", time_major=False):
    # tokenize
    noise_data = [add_noise(line) + [eos] for line in data]
    data = [line.split() + [eos] for line in data]

    unkid = voc[unk]

    newdata = []
    new_noise_data = []

    for d in data:
        idlist = [voc[w] if w in voc else unkid for w in d]
        newdata.append(idlist)
    for d in noise_data:
        idlist = [voc[w] if w in voc else unkid for w in d]
        new_noise_data.append(idlist)

    data = newdata
    noise_data = new_noise_data

    lens = [len(tokens) for tokens in data]

    n = len(lens)
    maxlen = np.max(lens)

    batch_data = np.zeros((n, maxlen), "int32")
    batch_noise_data = np.zeros((n, maxlen), "int32")

    data_length = np.array(lens)

    for idx, item in enumerate(data):
        batch_data[idx, :lens[idx]] = item
    
    for idx, item in enumerate(noise_data): 
        batch_noise_data[idx, :lens[idx]] = item

    if time_major:
        batch_data = batch_data.transpose()
        batch_noise_data = batch_noise_data.transpose()

    return batch_data, batch_noise_data, data_length
