from util import parse_args, load_config
import scipy.io.wavfile as wav
import numpy as np
import pandas as pd


def spliter(split_path, split_filename, split_index):

	fs, audio = wav.read(split_path + split_filename)

	word = pd.read_csv(split_path + split_filename[:-8] + ".WRD", sep=" ", header=None)

	verb = audio[word[0][split_index]:word[1][split_index]]

	wav.write( split_path[:-4] + split_filename + "_split.wav", fs, verb.astype(np.int16))

if __name__ == '__main__':
	args = parse_args()
	spliter(args.split_path, args.split_filename, args.split_index)