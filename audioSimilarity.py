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
    newfile_name = audio_name + '_simi.wav'

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

    if config['model_type'] == 'speechbrain':
        savedir = config['model_save_dir'] + config['model']
        asr_model = EncoderDecoderASR.from_hparams(source=config['model'], savedir=savedir) 
        normalizer = AudioNormalizer(sample_rate=16000)
    if config['model_type'] == 'deepspeech':
        speech_model = Model(config['model_path'])
        speech_model.enableExternalScorer(config['model_score_path'])

    word_list = speech_model.stt(originalAudio)

    """
    ###	compare all the audio file with all the words in frequency domain
    ### size of the word audio mast consistent with origional split audio
    """
    simi_min = 9999999999
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
                
                ##
                ## Similarity calculation
                ## frequency similarity
                #diff = sum(abs(abs(new_transform[:len_transform//2]) - abs(transform[:len_transform//2])))

                # cos similarity
                diff = np.dot(splitAudio, word_audio)/(np.linalg.norm(splitAudio)*np.linalg.norm(word_audio))
                

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
    if config['model_type'] == 'deepspeech':
        speech_model = Model(config['model_path'])
        speech_model.enableExternalScorer(config['model_score_path'])

    theta = 1
    epoch = 100
    epoch_counter = 0
    stage = 0

    while(True or epoch_counter < epoch):
        epoch_counter += 1
        theta = np.round(theta, decimals=5)

        word_audio = ifft(transform)
        simi_audio = ifft(simi_transform)
        new_audio = np.copy(originalAudio)
        
        ##########
        # attack
        
        new_audio[verb_range] = word_audio * (1-theta) + simi_audio * theta

        '''index = 0 
        change = round(len(transform) * theta)
        for i in verb_range:
            if change == 0:
                new_audio[i] = word_audio[index]
            elif index % change == 0:
                new_audio[i] = word_audio[index]
            else:
                new_audio[i] = simi_audio[index]
            index += 1
        '''    
        ##########

        if config['model_type'] == 'speechbrain':
            wavfile.write(split_path + "precessing.wav", 16000, new_audio.astype(np.int16))
            pred = asr_model.transcribe_file(split_path + "precessing.wav")
            AS = compute_as(word_list.lower().split(), pred.lower().split())
        if config['model_type'] == 'deepspeech':
            pred = speech_model.stt(new_audio)
            AS = compute_as(word_list.lower().split(), pred.lower().split())

        ## evaluation
        print("epoch: " + str(epoch_counter) + 
            " " + newfile_name + 
            " theta: " + str(theta) +
            " AS: " + str(np.round(AS, decimals=3)) + 
            " SNR: " + str(compute_snr(originalAudio, new_audio)) +
            " CORR: " + str(compute_corr(originalAudio, new_audio)))
        if stage == 0 and AS > 0:
            theta -= 0.1
            wavfile.write(split_path + newfile_name, 16000, new_audio.astype(np.int16))
        elif stage == 0 and AS <= 0:
            theta += 0.09
            stage = 1
        elif stage == 1 and AS > 0:
            theta -= 0.01
            wavfile.write(split_path + newfile_name, 16000, new_audio.astype(np.int16))
        elif stage == 1 and AS <= 0:
            theta += 0.009
            stage = 2
        elif stage == 2 and AS > 0:
            theta -= 0.001
            wavfile.write(split_path + newfile_name, 16000, new_audio.astype(np.int16)) 
        elif stage == 2 and AS <= 0:
            theta += 0.0009
            stage = 3
        elif stage == 3 and AS > 0:
            theta -= 0.0001
            wavfile.write(split_path + newfile_name, 16000, new_audio.astype(np.int16))
        else:
            return


        

def compute_corr(originalAudio, new_audio):
    corr = np.correlate(originalAudio/np.linalg.norm(originalAudio), new_audio/np.linalg.norm(new_audio))
    return corr

def compute_snr(originalAudio, new_audio):
    signal_power = np.mean(np.square(new_audio))
    noise_power = np.mean(np.square(originalAudio - new_audio))

    # compute the SNR in decibels (dB)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def compute_as(truth, hypothesis):
    distances = np.zeros((len(truth) + 1) * (len(hypothesis) + 1), dtype=np.uint16)
    distances = distances.reshape((len(truth) + 1, len(hypothesis) + 1))
    
    for i in range(len(truth) + 1):
        distances[i][0] = i
    for j in range(len(hypothesis) + 1):
        distances[0][j] = j
    
    for i in range(1, len(truth) + 1):
        for j in range(1, len(hypothesis) + 1):
            if truth[i-1] == hypothesis[j-1]:
                distances[i][j] = distances[i-1][j-1]
            else:
                substitute_cost = distances[i-1][j-1] + 1
                insert_cost = distances[i][j-1] + 1
                delete_cost = distances[i-1][j] + 1
                distances[i][j] = min(substitute_cost, insert_cost, delete_cost)
    
    if distances[len(truth)][len(hypothesis)] == 0:
        return 0
    else:
        return 1 / distances[len(truth)][len(hypothesis)]


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    similarity(config)