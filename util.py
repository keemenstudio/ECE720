import argparse
import os
import pathlib
import yaml

def load_config(path=None):
    if path is None:
        path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        'configs/config.yaml')
        
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/speechbrain_config.yaml')

    ## for audio split
    parser.add_argument('--split_path', type=str, default='test/raw/')
    parser.add_argument('--split_filename', type=str, default='')
    parser.add_argument('--split_index', type=int, default= 0)
    args = parser.parse_args()
    return args