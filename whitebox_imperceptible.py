import glob
import os

import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchaudio

import sys, os
# sys.path.append("adversarial-robustness-toolbox")
sys.path.insert(0, 'adversarial-robustness-toolbox')
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from art import config
from art.defences.preprocessor import Mp3Compression
from art.utils import get_file

from art.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech
from art.attacks.evasion.imperceptible_asr.imperceptible_asr_pytorch import ImperceptibleASRPyTorch

import scipy.io.wavfile as wav

import torchaudio


sys.path.append("deepspeech.pytorch")
from deepspeech_pytorch.utils import load_decoder, load_model
from deepspeech_pytorch.configs.inference_config import TranscribeConfig
import pandas as pd
import torch.nn.functional as F
import Levenshtein
def display_waveform(waveform, title="", sr=16000):
  """Display waveform plot and audio play UI."""
  plt.figure()
  plt.title(title)
  plt.plot(waveform)
  ipd.display(ipd.Audio(waveform, rate=sr))


def imperceptible_attack(audio_path, output_file, model_path, split_index, dataset_path, epsilon, alpha, mode, PGD_iter, attack_range = []):
    device = 'cuda'
    sample_rate = 16000
    use_amp = False
    # model = load_model(device=device, model_path=os.path.abspath(model_path))
    speech_recognizer = PyTorchDeepSpeech(
        pretrained_model="librispeech",
        device_type=device,
        use_amp = use_amp
        )
    

    ## read all filename and filepath
    raw_path = audio_path[:9]
    split_path = audio_path[:5]
    audio_name = audio_path[9:]
    wordfile_name = audio_name[:-8] + '.WRD'
    splitfile_name = audio_name + '_split.wav'
    newfile_name = audio_name + '_new.wav'

    ## read the original audio
    word_file = pd.read_csv(raw_path + wordfile_name, sep=" ", header=None)
    verb_range = range(word_file[0][split_index],word_file[1][split_index])
    verb = word_file[2][split_index]
    gt = ' '.join(list(word_file[2].to_numpy()))
    print(gt)
    asr_attack = ImperceptibleASRPyTorch(
        estimator=speech_recognizer,
        eps=0.001,
        max_iter_1=5,
        max_iter_2=5,
        learning_rate_1=0.00001,
        learning_rate_2=0.001,
        optimizer_1=torch.optim.Adam,
        optimizer_2=torch.optim.Adam,
        global_max_length=100000,
        initial_rescale=1.0,
        decrease_factor_eps=0.8,
        num_iter_decrease_eps=5,
        alpha=0.01,
        increase_factor_alpha=1.2,
        num_iter_increase_alpha=5,
        decrease_factor_alpha=0.8,
        num_iter_decrease_alpha=5,
        win_length=2048,
        hop_length=512,
        n_fft=2048,
        batch_size=2,
        use_amp=use_amp,
        opt_level="O1",
        attack_range = [verb_range.start, verb_range.stop]
    )
    ## read the split audio
    # splitAudio = originalAudio[verb_range]
    # length = splitAudio.shape[0] / fs
    # time = np.linspace(0., length, splitAudio.shape[0])
    fs, audio = wav.read(raw_path + audio_name)
    sound, sample_rate = torchaudio.load(raw_path + audio_name)
    split_audio = sound[:, verb_range]
    length = split_audio.shape[1] / sample_rate

    template,_ = speech_recognizer.predict(split_audio.numpy(), batch_size=1, transcription_output=False)
    template = template.flatten()
    template_length = template.size

    template_str = speech_recognizer.predict(sound.numpy(), batch_size=1, transcription_output=True)
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

    for root, directories, files in os.walk(dataset_path):
        for file in files:
            if file[-4:] == ".wav" and file != audio_name:
                abs_filepath = os.path.join(root, file)
                pure_name = file[:-8]
                word = pd.read_csv(os.path.join(root, pure_name + '.WRD'), sep=" ", header=None)
                audio, _ = torchaudio.load(os.path.join(root, pure_name + '.WAV.wav'))

                for i in range(len(word)):
                    word_audio = audio[:, word.values[i][0] : word.values[i][1]]
                    tmp_result, _ = speech_recognizer.predict(word_audio.numpy(), batch_size=1, transcription_output=False)
                    tmp_result = tmp_result.flatten()
                    tmp_result_length = tmp_result.size
                    if word.values[i][2] == verb:
                        continue
                    
                    tmp_result = torch.Tensor(tmp_result)
                    if tmp_result_length < template_length:
                        tmp_result = F.pad(tmp_result, pad = (0, template_length - tmp_result_length), mode='constant', value=0)
                    elif tmp_result_length > template_length:
                        tmp_result = tmp_result[:template_length]
                    tensor_template = torch.Tensor(template)
                    cos = torch.nn.CosineSimilarity(dim=0)
                    diff = cos(tmp_result, tensor_template)
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

    print(simi_max)
    print(simi_filename)
    print(simi_index)
    print(simi_word)
    print(f"range: {verb_range}")

    target = gt.replace(verb, simi_word)
    target = target.upper()
    print(target)

    y = np.array([target.upper()])

    sound_adv = asr_attack.generate(sound.numpy(), y)
    final_output = speech_recognizer.predict(sound_adv, batch_size=1, transcription_output=True)
    sound = sound.numpy()
    # abs_ori = 20*np.log10(np.sqrt(np.mean(np.absolute(sound)**2)))
    # abs_after = 20*np.log10(np.sqrt(np.mean(np.absolute(sound_adv)**2)))
    # db_difference = abs_after-abs_ori
    l_distance = Levenshtein.distance(target[0], final_output[0])
    return 0, l_distance, target, final_output[0], template_str[0], sound_adv, sound, sound_adv - sound