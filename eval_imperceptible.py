import torch
# from jiwer import wer
import numpy as np
import os
from util import parse_args, load_config
import pandas as pd
import torchaudio
from whitebox_imperceptible import imperceptible_attack
import json

def snr(audio, perturbation):
    signal_power = np.mean(np.square(audio))
    noise_power = np.mean(np.square(perturbation))

    # compute the SNR in decibels (dB)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def wer3(r, h):
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    if d[len(r)][len(h)] == 0:
        return 0
    else:
        return 1 / d[len(r)][len(h)]

def wer(recog, ref):
    n = len(recog)
    m = len(ref)
    dp = np.zeros((n + 1, m + 1))

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0:
                dp[i, j] = j
            elif j == 0:
                dp[i, j] = i
            elif recog[i-1] == ref[j-1]:
                dp[i, j] = dp[i-1, j-1]
            else:
                dp[i, j] = 1 + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])

    num_insertions = dp[n, m-1]
    num_deletions = dp[n-1, m]
    num_substitutions = dp[n-1, m-1]

    distance = (num_insertions + num_deletions + num_substitutions)
    wer = (num_insertions + num_deletions + num_substitutions) / m
    if distance == 0:
        return 0
    else:
        return 1 / distance

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
    cnt = 0
    for root, directories, files in os.walk(config['audio_files_path']):
        for file in files:
            if file[-4:] == ".wav":
                abs_filepath = os.path.join(root, file)
                output_file = config['output_path'] + file + '.imperceptible.out.wav'
                print(f"current processing: {abs_filepath}")
                db_difference, l_distance, target_string, final_output, original_output, perturbed_data, data_raw, perterbation = imperceptible_attack(abs_filepath, output_file, config['model_path'], config['split_index'], config['dataset_path'], config['epsilon'], config['alpha'], config['mode'], config['PGD_iter'])
                perturbed_data = perturbed_data
                data_raw = data_raw
                perterbation = perterbation
                result_snr = snr(perturbed_data, perterbation)
                print(f"snr {result_snr}")
                result_as = wer(original_output.split(" "), final_output.split(" "))
                print(f"as {result_as}")
                as2 = compute_as(original_output.split(" "), final_output.split(" "))
                print(f"as2 {as2}")
                as3 = wer3(original_output.split(" "), final_output.split(" "))
                print(f"as3 {as3}")
                result_dict = {}
                result_dict['snr'] = result_snr
                result_dict['as'] = as3
                corr = np.correlate(data_raw[0], perturbed_data[0], mode='same').max() / (np.std(data_raw[0]) * np.std(perturbed_data[0]) * len(data_raw[0]))
                print(f"cc {corr}")

                result_dict['cc'] = corr
                result_dict['file'] = abs_filepath
                result_dict['output_file'] = abs_filepath
                result_dict['original_output'] = original_output
                result_dict['target_output'] = target_string
                result_dict['final_output'] = final_output
                summary.append(result_dict)
                print(result_dict)
                cnt += 1
                if cnt > 20:
                    print(f"final output: {json.dumps(summary)}")
                    return
                # return
    print(f"final output: {json.dumps(summary)}")

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    main(config)