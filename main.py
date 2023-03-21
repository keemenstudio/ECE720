
from util import parse_args, load_config

from speechbrain.pretrained import EncoderDecoderASR

def recognize(config):
    savedir = config['model_save_dir'] + config['model']
    asr_model = EncoderDecoderASR.from_hparams(source=config['model'], 
    savedir=savedir) 
    return asr_model.transcribe_file(config['audio_path'])

def attack(config):
    pass


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    print(recognize(config))