import torch
from jiwer import wer
import numpy as np
import os
from util import parse_args, load_config
import pandas as pd
import torchaudio
from whitebox_similarity import pgd_attack

def snr(audio, perturbation, rel_length=torch.tensor([1.0])):
    """
    Signal to Noise Ratio computation
    Arguments
    ---------
    audio : torch.tensor
        the original padded audio
    perturbation : torch.tensor
        the padded perturbation
    rel_length : torch.tensor
        the relative length of the wavs in the batch
    """
    num = torch.max(audio, dim=1)
    den = torch.max(perturbation, dim=1)
    ratio = num[0]/den[0]

    snr = 20. * torch.log10(ratio)
    return torch.round(snr).long()

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
    for root, directories, files in os.walk(config['dataset_path']):
        for file in files:
            if file[-4:] == ".wav":
                abs_filepath = os.path.join(root, file)
                output_file = config['output_path'] + file + '.out.wav'
                print(f"current processing: {abs_filepath}")
                db_difference, l_distance, target_string, final_output, original_output, perturbed_data, data_raw, perterbation = pgd_attack(abs_filepath, output_file, config['model_path'], config['split_index'], config['dataset_path'], config['epsilon'], config['alpha'], config['mode'], config['PGD_iter'])
                result_snr = snr(data_raw, perturbed_data)
                print(f"snr {result_snr}")
                result_as = compute_as(original_output.split(' '), final_output.split(' '))
                print(f"as {result_as}")
                return


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    main(config)