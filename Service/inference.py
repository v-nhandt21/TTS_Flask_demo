#!/usr/bin/env python
# coding: utf-8

# ## Tacotron 2 inference code 
# Edit the variables **checkpoint_path** and **text** to match yours and run the entire code to generate plots of mel outputs, alignments and audio synthesis from the generated mel-spectrogram using Griffin-Lim.


import sys
print(sys.executable)


import matplotlib
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt

import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser
import soundfile as sd
from vinorm import TTSnorm
import os

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', interpolation='none')
def getAudio(text):
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    hparams = create_hparams()
    hparams.sampling_rate = 22050

    waveglow_path = os.path.join(__location__, 'waveglow_256channel.pt')
    waveglow = torch.load(waveglow_path,map_location='cpu')['model']
    waveglow.cpu().eval()
    for k in waveglow.convinv:
        k.float()
    #denoiser = Denoiser(waveglow)

    checkpoint_path = os.path.join(__location__, "checkpoint_9000")
    model = load_model(hparams)
    #print(model)
    state = torch.load(checkpoint_path,map_location='cpu')['state_dict']
    #print(state)
    model.load_state_dict(state)
    _ = model.cpu().eval()

    #text = "Bộ Y tế chỉ đạo Viện Vệ sinh dịch tễ và các địa phương điều tra dịch tễ các trường hợp có kết quả xét nghiệm dương tính, xác minh người tiếp xúc gần với bệnh nhân dương tính, khoanh vùng xử lý ổ dịch và cách ly theo dõi sức khỏe những người tiếp xúc."
    text = TTSnorm(text)
    sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()



    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    #plot_data((mel_outputs.float().data.cpu().numpy()[0], mel_outputs_postnet.float().data.cpu().numpy()[0], alignments.float().data.cpu().numpy()[0].T))


    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    #########
    #Thieu phan denoiser do phai chay tren gpu
    #########
        
    
    text_hashed=abs(hash(text)) % (10 ** 8)
    sd.write("static/audio/"+str(text_hashed)+'.wav',audio[0].data.cpu().numpy(), 22050)
    return text


#audio_denoised = denoiser(audio, strength=0.01)[:, 0]
#ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate) 





