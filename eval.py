import torch
from jiwer import wer
import numpy as np
import os
from util import parse_args, load_config
import pandas as pd
import torchaudio
from whitebox_similarity import pgd_attack
import json

def snr(audio, perturbation):
    signal_power = np.mean(np.square(audio))
    noise_power = np.mean(np.square(perturbation))

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

def main(config):
    summary = []

    for root, directories, files in os.walk(config['dataset_path']):
        for file in files:
            if file[-4:] == ".wav":
                abs_filepath = os.path.join(root, file)
                output_file = config['output_path'] + file + '.out.wav'
                print(f"current processing: {abs_filepath}")
                db_difference, l_distance, target_string, final_output, original_output, perturbed_data, data_raw, perterbation = pgd_attack(abs_filepath, output_file, config['model_path'], config['split_index'], config['dataset_path'], config['epsilon'], config['alpha'], config['mode'], config['PGD_iter'])
                perturbed_data = perturbed_data.detach().cpu().numpy()
                data_raw = data_raw.detach().cpu().numpy()
                perterbation = perterbation.detach().cpu().numpy()
                result_snr = snr(perturbed_data, perterbation)
                print(f"snr {result_snr}")
                result_as = compute_as(original_output.split(' '), final_output.split(' '))
                print(f"as {result_as}")
                result_dict = {}
                result_dict['snr'] = result_snr
                result_dict['as'] = result_as
                corr = np.correlate(data_raw[0], perturbed_data[0], mode='full')
                print(f"cc {corr}")

                result_dict['cc'] = corr
                result_dict['file'] = abs_filepath
                result_dict['output_file'] = abs_filepath
                result_dict['original_output'] = original_output
                result_dict['target_output'] = target_string
                result_dict['final_output'] = final_output
                summary.append(result_dict)
                print(result_dict)
                # return
    print(f"final output: {json.dumps(result_dict)}")

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    main(config)