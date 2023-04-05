from util import parse_args, load_config
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.dataio.preprocess import AudioNormalizer
from deepspeech import Model

import os
import scipy.io.wavfile as wav
from scipy.fft import fft, fftfreq, ifft
from scipy.io import wavfile
from scipy.signal import resample
import pandas as pd
import numpy as np
import torch
import sys, os
sys.path.append("deepspeech.pytorch")
from deepspeech_pytorch.utils import load_decoder, load_model
from deepspeech_pytorch.configs.inference_config import TranscribeConfig
from whitebox.stft import STFT, magphase
from whitebox.attack import torch_spectrogram
import torchaudio
import torch.nn.functional as F
from whitebox.attack import Attacker

def predict(model, torch_stft, sound, device):
    data = sound.to(device)
    
    # initial prediction
    spec = torch_spectrogram(data, torch_stft)
    input_sizes = torch.IntTensor([spec.size(3)]).int()
    out, output_sizes, hs = model(spec, input_sizes)
    return out


def similarity(config):
    device = 'cuda'
    sample_rate = 16000
    model = load_model(device=device, model_path=os.path.abspath(config['model_path']))
    n_fft = int(sample_rate * 0.02)
    hop_length = int(sample_rate * 0.01)
    win_length = int(sample_rate * 0.02)
    torch_stft = STFT(n_fft=n_fft , hop_length=hop_length, win_length=win_length ,  window='hamming', center=True, pad_mode='reflect', freeze_parameters=True, device=device)

    ## read all filename and filepath
    raw_path = config['audio_path'][:9]
    split_path = config['audio_path'][:5]
    audio_name = config['audio_path'][9:]
    wordfile_name = audio_name[:-8] + '.WRD'
    splitfile_name = audio_name + '_split.wav'
    newfile_name = audio_name + '_new.wav'

    ## read the original audio
    word_file = pd.read_csv(raw_path + wordfile_name, sep=" ", header=None)
    verb_range = range(word_file[0][config['split_index']],word_file[1][config['split_index']])
    verb = word_file[2][config['split_index']]
    gt = ' '.join(list(word_file[2].to_numpy()))
    print(gt)

    ## read the split audio
    # splitAudio = originalAudio[verb_range]
    # length = splitAudio.shape[0] / fs
    # time = np.linspace(0., length, splitAudio.shape[0])
    fs, audio = wav.read(raw_path + audio_name)
    sound, sample_rate = torchaudio.load(raw_path + audio_name)
    split_audio = sound[:, verb_range]
    length = split_audio.shape[1] / sample_rate

    template = predict(model, torch_stft, split_audio, device)
    template = template.flatten()
    template_length = template.size(0)
    # transform = fft(splitAudio)
    # len_transform = len(transform)

    """
    ###	compare all the audio file with all the words in frequency domain
    ### size of the word audio mast consistent with origional split audio
    """
    simi_max = 0
    simi_filename = ''
    simi_index = 0
    simi_transform = 0

    for root, directories, files in os.walk(config['dataset_path']):
        for file in files:
            if file[-4:] == ".wav" and file != audio_name:
                abs_filepath = os.path.join(root, file)
                pure_name = file[:-8]
                word = pd.read_csv(os.path.join(root, pure_name + '.WRD'), sep=" ", header=None)
                audio, _ = torchaudio.load(os.path.join(root, pure_name + '.WAV.wav'))

                for i in range(len(word)):
                    word_audio = audio[:, word.values[i][0] : word.values[i][1]]
                    tmp_result = predict(model, torch_stft, word_audio, device)
                    tmp_result = tmp_result.flatten()
                    tmp_result_length = tmp_result.size(0)

                    if tmp_result_length < template_length:
                        tmp_result = F.pad(tmp_result, pad = (0, template_length - tmp_result_length), mode='constant', value=0)
                    elif tmp_result_length > template_length:
                        tmp_result = tmp_result[:template_length]

                    cos = torch.nn.CosineSimilarity(dim=0)
                    diff = cos(tmp_result, template)
                    # if len(word_audio) != len_transform:
                    #     word_audio = resample(word_audio, len_transform)

                    # new_transform = fft(word_audio)
                    # diff = sum(abs(abs(new_transform[:len_transform//2]) - abs(transform[:len_transform//2])))
                    


                    if diff.item() > simi_max:
                        simi_max = diff.item()
                        simi_filename = file
                        simi_index = i
                        simi_word = word.values[i][2]
                        # simi_transform = new_transform


    # for file in os.listdir(raw_path):
    # 	if file[-4:] == ".wav" and file != audio_name:
    # 		# get pure_name 
    #         pure_name = file[:-8]

	# 		#  audio = pure_name + '.WAV.wav'
	# 		#  word = pure_name + '.WRD'
    #         # get current audio + word dataframe
    #         # fs, audio = wav.read(raw_path + pure_name + '.WAV.wav')
    #         word = pd.read_csv(raw_path + pure_name + '.WRD', sep=" ", header=None)

    #         audio, _ = torchaudio.load(raw_path + pure_name + '.WAV.wav')

    #         for i in range(len(word)):
    #             word_audio = audio[:, word.values[i][0] : word.values[i][1]]
    #             tmp_result = predict(model, torch_stft, word_audio, device)
    #             tmp_result = tmp_result.flatten()
    #             tmp_result_length = tmp_result.size(0)

    #             if tmp_result_length < template_length:
    #                 tmp_result = F.pad(tmp_result, pad = (0, template_length - tmp_result_length), mode='constant', value=0)
    #             elif tmp_result_length > template_length:
    #                 tmp_result = tmp_result[:template_length]

    #             cos = torch.nn.CosineSimilarity(dim=0)
    #             diff = cos(tmp_result, template)
    #             # if len(word_audio) != len_transform:
    #             #     word_audio = resample(word_audio, len_transform)

    #             # new_transform = fft(word_audio)
    #             # diff = sum(abs(abs(new_transform[:len_transform//2]) - abs(transform[:len_transform//2])))
                


    #             if diff.item() > simi_max:
    #                 simi_max = diff.item()
    #                 simi_filename = file
    #                 simi_index = i
    #                 simi_word = word.values[i][2]
    #                 # simi_transform = new_transform

    print(simi_max)
    print(simi_filename)
    print(simi_index)
    print(simi_word)

    target = gt.replace(verb, simi_word)
    print(target)


    cfg = TranscribeConfig
    decoder = load_decoder(labels=model.labels, cfg=cfg.lm)
    attacker = Attacker(model=model, sound=sound, target=target.upper(), decoder=decoder, device=device, save=config['output_file'], org_str=gt.upper())

    attacker.attack(epsilon = config['epsilon'], alpha=float(config['alpha']), attack_type=config['mode'], PGD_round=config['PGD_iter'])

def AttackScore(pred, gt):
    same = 0
    diff = 0
    if len(pred) != len(gt):
        return 1
    for i, word  in enumerate(gt):
        if gt[i] == pred[i]:
            same += 1
        else: diff += 1
    return diff/same


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    similarity(config)