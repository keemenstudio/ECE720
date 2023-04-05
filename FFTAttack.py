from util import parse_args, load_config
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.dataio.preprocess import AudioNormalizer
from deepspeech import Model

import scipy.io.wavfile as wav
from scipy.fft import fft, fftfreq, ifft
from scipy.io import wavfile
import pandas as pd
import numpy as np
import torch
import torchaudio

def attack(config):
    ## read all filename and filepath
    raw_path = config['audio_path'][:9]
    split_path = config['audio_path'][:5]
    audio_name = config['audio_path'][9:]
    wordfile_name = audio_name[:-8] + '.WRD'
    splitfile_name = audio_name + '_split.wav'
    #attackedfile_name = audio_name + '_attacked.wav'
    newfile_name = audio_name + '_new.wav'

    ## read the original audio
    fs, originalAudio = wav.read(raw_path + audio_name)
    word_file = pd.read_csv(raw_path + wordfile_name, sep=" ", header=None)
    verb_range = range(word_file[0][config['split_index']],word_file[1][config['split_index']])
    #verb_range = range(0,len(originalAudio))
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

    theta = 1
    epoch = 100
    epoch_counter = 0
    stage = 0

    while(True or epoch_counter < epoch):
        epoch_counter += 1


        ## cutoff frequency attack theta = 0.8 - 0.01
        #filtered = freqAttack(theta, transform, len_transform//2)

        ## cutoff amplitude attack theta = 0.3 = 0.01
        filtered = amplitudeAttack(theta, transform, len_transform//2)

        filtered = ifft(filtered)
        new_audio = np.copy(originalAudio)
        new_audio[verb_range] = filtered

        if config['model_type'] == 'speechbrain':
            wavfile.write(split_path + "precessing.wav", 16000, new_audio.astype(np.int16))
            pred = asr_model.transcribe_file(split_path + "precessing.wav")
            AS = AttackScore(pred.lower().split(), word_list)
        if config['model_type'] == 'deepspeech':
            pred = speech_model.stt(new_audio)
            AS = AttackScore(pred.lower().split(), word_list)

        ## evaluation
        print("epoch: " + str(epoch_counter) + " attack score: " + str(AS) + " theta: " + str(theta))
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
        else:
            return




def freqAttack(theta, transform, len_transform):
    threshold = round(len_transform * theta)
    filtered = np.copy(transform)
    filtered[len_transform-threshold:len_transform+threshold] = 0
    return filtered

def amplitudeAttack(theta, transform, len_transform):
    threshold = np.amax(transform)*theta
    filtered = np.copy(transform)
    filtered[abs(transform) < threshold] = 0
    return filtered

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
    attack(config)