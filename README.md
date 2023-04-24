# ECE720

### requirements
```
conda create env --name ece720
pip install -r requirements.txt
```

### White-box evaluation
#### PGD
python eval.py --config configs/eval_config.yaml
#### Imperceptible
python eval_imperceptible.py --config configs/eval_config.yaml


### Black-box evaluation
#### Semi-attack
python audioSimilarity.py --config deepspeech_config_win.yaml
#### FFT attack
python FFTAttack.py --config deepspeech_config_win.yaml