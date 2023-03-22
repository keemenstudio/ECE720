from util import parse_args, load_config
from speechbrain.pretrained import EncoderDecoderASR
from deepspeech import Model
import scipy.io.wavfile as wav

def recognize(config):
    if config['model_type'] == 'speechbrain':
        savedir = config['model_save_dir'] + config['model']
        asr_model = EncoderDecoderASR.from_hparams(source=config['model'], 
        savedir=savedir) 
        return asr_model.transcribe_file(config['audio_path'])
    if config['model_type'] == 'deepspeech':
        speech_model = Model(config['model_path'])
        speech_model.enableExternalScorer(config['model_score_path'])
        fs, audio = wav.read(config['audio_path'])
        return speech_model.stt(audio)


def attack(config):
    pass


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    print(recognize(config))