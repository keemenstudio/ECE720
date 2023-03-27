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


def similarity(config):
    ## read all filename and filepath
    raw_path = config['audio_path'][:9]
    split_path = config['audio_path'][:5]
    audio_name = config['audio_path'][9:]
    wordfile_name = audio_name[:-8] + '.WRD'
    splitfile_name = audio_name + '_split.wav'
    newfile_name = audio_name + '_new.wav'

    ## read the original audio
    fs, originalAudio = wav.read(raw_path + audio_name)
    word_file = pd.read_csv(raw_path + wordfile_name, sep=" ", header=None)
    verb_range = range(word_file[0][config['split_index']],word_file[1][config['split_index']])
    word_list = list(word_file[2].to_numpy())

    ## read the split audio
    splitAudio = originalAudio[verb_range]
    length = splitAudio.shape[0] / fs
    time = np.linspace(0., length, splitAudio.shape[0])
    transform = fft(splitAudio)
    len_transform = len(transform)

    """
    ###	compare all the audio file with all the words in frequency domain
    ### size of the word audio mast consistent with origional split audio
    """
    simi_min = 2147483647
    simi_filename = ''
    simi_index = 0
    simi_transform = 0
    for file in os.listdir(raw_path):
    	if file[-4:] == ".wav" and file != audio_name:
    		# get pure_name 
            pure_name = file[:-8]

			#  audio = pure_name + '.WAV.wav'
			#  word = pure_name + '.WRD'
            # get current audio + word dataframe
            fs, audio = wav.read(raw_path + pure_name + '.WAV.wav')
            word = pd.read_csv(raw_path + pure_name + '.WRD', sep=" ", header=None)

            for i in range(len(word)):
                word_audio = audio[word[0][i]:word[1][i]]

                if len(word_audio) != len_transform:
                    word_audio = resample(word_audio, len_transform)

                new_transform = fft(word_audio)
                diff = sum(abs(abs(new_transform[:len_transform//2]) - abs(transform[:len_transform//2])))
                
                if diff < simi_min:
                    simi_min = diff
                    simi_filename = file
                    simi_index = i
                    simi_transform = new_transform

    print(simi_min)
    print(simi_filename)
    print(simi_index)

    attack(split_path, originalAudio, verb_range, word_list, transform, simi_transform, newfile_name)

def attack(split_path, originalAudio, verb_range, word_list, transform, simi_transform, newfile_name):
    if config['model_type'] == 'speechbrain':
        savedir = config['model_save_dir'] + config['model']
        asr_model = EncoderDecoderASR.from_hparams(source=config['model'], savedir=savedir) 
        normalizer = AudioNormalizer(sample_rate=16000)

    theta = 0.8
    epoch = 100
    epoch_counter = 0

    while(True or epoch_counter < epoch):
        epoch_counter += 1

        word_audio = ifft(transform)
        simi_audio = ifft(simi_transform)

        new_audio = originalAudio
        new_audio[verb_range] = word_audio * (1-theta) + simi_audio * theta

        wavfile.write(split_path + "precessing.wav", 16000, new_audio.astype(np.int16))
        pred = asr_model.transcribe_file(split_path + "precessing.wav")
        AS = AttackScore(pred.lower().split(), word_list)

        if AS <= 0:
            return
        else:
            print("epoch: " + str(epoch_counter) + " attack score: " + str(AS) + " theta: " + str(theta))
            theta -= 0.01

        wavfile.write(split_path + newfile_name, 16000, new_audio.astype(np.int16))

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