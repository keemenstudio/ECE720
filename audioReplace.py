from util import parse_args, load_config
import scipy.io.wavfile as wav
import numpy as np
import pandas as pd

def replacer(split_path, split_filename, split_index):
	fs, originalAudio = wav.read(split_path + split_filename)
	fs, wordAudio = wav.read(split_path[:-4] + split_filename + "_attacked.wav")

	word = pd.read_csv(split_path + split_filename[:-8] + ".WRD", sep=" ", header=None)

	if len(originalAudio[word[0][split_index]:word[1][split_index]]) == len(wordAudio):
		originalAudio[word[0][split_index]:word[1][split_index]] = wordAudio

		wav.write( split_path[:-4] + split_filename + "_new.wav", fs, originalAudio.astype(np.int16))
	else:
		print("Length of the attacked audio and original split audio not match") 
		return

if __name__ == '__main__':
	args = parse_args()
	replacer(args.split_path, args.split_filename, args.split_index)