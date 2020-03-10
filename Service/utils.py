import numpy as np
from scipy.io.wavfile import read
import torch


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    #mask = (ids < lengths.unsqueeze(1)).bool()
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    #full_path = full_path.replace('wav','big')
    full_path = full_path.replace('DATA/VLSP/BigCorpus/wav','vn_tacotron2/BigCorpus/big')
    #print(full_path)
    #full_path = '/home/trinhan/AILAB/TTS/DNNmodel/vn_tacotron2/VAIS/' + full_path + '.wav'
    '''folder_name = full_path.split('_')[0]
    if 'VIVOSSPK' in folder_name:
            full_path = '/home/trinhan/AILAB/TTS/DNNmodel/vn_tacotron2/vivos/train/waves/' + folder_name + '/normalized/' + full_path + '.wav'
    else:
            full_path = '/home/trinhan/AILAB/TTS/DNNmodel/vn_tacotron2/vivos/test/waves/' + folder_name + '/normalized/' + full_path + '.wav'
    '''
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    #alter
    #split=" "
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split, 1) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
