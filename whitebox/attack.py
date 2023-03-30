from .stft import STFT, magphase
import torch.nn as nn
import torch
import Levenshtein
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import speechbrain as sb
import torch.optim as optim
def target_sentence_to_label(sentence, labels="_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "):
    out = []
    for word in sentence:
        out.append(labels.index(word))
    return torch.IntTensor(out)

def torch_spectrogram(sound, torch_stft):
    real, imag = torch_stft(sound)
    mag, cos, sin = magphase(real, imag)
    mag = torch.log1p(mag)
    mean = mag.mean()
    std = mag.std()
    mag = mag - mean
    mag = mag / std
    mag = mag.permute(0,1,3,2)
    return mag


class Attacker:
    def __init__(self, model, sound, target, decoder, sample_rate=16000, device="cpu", save=None):
        """
        model: deepspeech model
        sound: raw sound data [-1 to +1] (read from torchaudio.load)
        label: string
        """
        self.sound = sound
        self.sample_rate = sample_rate
        self.target_string = target
        self.target = target
        self.__init_target()
        
        self.model = model
        self.model.to(device)
        self.model.train()
        self.decoder = decoder
        self.criterion = nn.CTCLoss()
        self.device = device
        n_fft = int(self.sample_rate * 0.02)
        hop_length = int(self.sample_rate * 0.01)
        win_length = int(self.sample_rate * 0.02)
        self.torch_stft = STFT(n_fft=n_fft , hop_length=hop_length, win_length=win_length ,  window='hamming', center=True, pad_mode='reflect', freeze_parameters=True, device=self.device)
        self.save = save
    
    def get_ori_spec(self, save=None):
        spec = torch_spectrogram(self.sound.to(self.device), self.torch_stft)
        plt.imshow(spec.cpu().numpy()[0][0])
        if save:
            plt.savefig(save)
            plt.clf()
        else:
            plt.show()

    def get_adv_spec(self, save=None):
        spec = torch_spectrogram(self.perturbed_data.to(self.device), self.torch_stft)
        plt.imshow(spec.cpu().numpy()[0][0])
        if save:
            plt.savefig(save)
            plt.clf()
        else:
            plt.show()
    
    # prepare
    def __init_target(self):
        self.target = target_sentence_to_label(self.target)
        self.target = self.target.view(1,-1)
        self.target_lengths = torch.IntTensor([self.target.shape[1]]).view(1,-1)

    # FGSM
    def fgsm_attack(self, sound, epsilon, data_grad):
        
        # find direction of gradient
        sign_data_grad = data_grad.sign()
        
        # add noise "epilon * direction" to the ori sound
        perturbed_sound = sound - epsilon * sign_data_grad
        
        return perturbed_sound
    
    # PGD
    def pgd_attack(self, sound, ori_sound, eps, alpha, data_grad) :
        grad_sign = data_grad.sign()
        # grad_sign[:, :3720] *= alpha * 0.1
        # grad_sign[:, 3720:8840] *= alpha
        # grad_sign[:, 8840:] *= alpha * 0.1
        adv_sound = sound - alpha * grad_sign
        # adv_sound = sound - alpha * data_grad.sign() # + -> - !!!
        eta = torch.clamp(adv_sound - ori_sound.data, min=-eps, max=eps)
        # eta[:, :3720] *= 0.1
        # eta[:, 3720:8840] *= alpha
        # eta[:, 8840:] *= 0.1
        sound = ori_sound + eta

        return sound
    
    def cw_attack(self, sound, ori_sound, target, max_iters, learning_rate, kappa):
        def f(x):
            spec = torch_spectrogram(x, self.torch_stft)
            input_sizes = torch.IntTensor([spec.size(3)]).int()
            out, output_sizes, hs = self.model(spec, input_sizes)
            out = out.transpose(0, 1)  # TxNxH
            out = out.log_softmax(2)
            return out
        
        def loss_fn(x, target, kappa):
            logits = f(x)
            batch_size = logits.size(0)
            target = target.view(-1, 1).expand(batch_size, logits.size(1))
            real_logits = torch.gather(logits, 1, target)
            max_except_real = torch.where(target == logits, torch.full_like(logits, -1e9), logits)
            wrong_logits, _ = torch.max(max_except_real, dim=1)
            loss = torch.clamp(wrong_logits - real_logits.squeeze(1) + kappa, min=0)
            return torch.sum(loss)

        # def loss_fn(x, target, kappa):
        #     logits = f(x)
        #     real_logits = logits.gather(1, target.type(torch.int64).view(-1, 1)).squeeze(1)
        #     wrong_logits = torch.max(logits, dim=1)[0]
        #     loss = torch.clamp(wrong_logits - real_logits + kappa, min=0)
        #     return torch.sum(loss)

        delta = torch.zeros_like(sound, requires_grad=True)
        optimizer = optim.Adam([delta], lr=learning_rate)

        for i in range(max_iters):
            adv_sound = torch.clamp(sound + delta, 0, 1)
            loss = loss_fn(adv_sound, target, kappa)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return torch.clamp(sound + delta, 0, 1).detach()
    
    def attack(self, epsilon, alpha, attack_type = "FGSM", PGD_round=40):
        print("Start attack")
        
        data, target = self.sound.to(self.device), self.target.to(self.device)
        data_raw = data.clone().detach()
        
        # initial prediction
        spec = torch_spectrogram(data, self.torch_stft)
        input_sizes = torch.IntTensor([spec.size(3)]).int()
        out, output_sizes, hs = self.model(spec, input_sizes)
        decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
        original_output = decoded_output[0][0]
        print(f"Original prediction: {decoded_output[0][0]}")
        
        # ATTACK
        ############ ATTACK GENERATION ##############
        if attack_type == "FGSM":
            data.requires_grad = True
            
            spec = torch_spectrogram(data, self.torch_stft)
            input_sizes = torch.IntTensor([spec.size(3)]).int()
            out, output_sizes, hs = self.model(spec, input_sizes)
            out = out.transpose(0, 1)  # TxNxH
            out = out.log_softmax(2)
            loss = self.criterion(out, self.target, output_sizes, self.target_lengths)
            
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

        elif attack_type == "PGD":
            for i in range(PGD_round):
                print(f"PGD processing ...  {i+1} / {PGD_round}", end="\r")
                data.requires_grad = True
                
                spec = torch_spectrogram(data, self.torch_stft)
                input_sizes = torch.IntTensor([spec.size(3)]).int()
                out, output_sizes, hs = self.model(spec, input_sizes)
                out = out.transpose(0, 1)  # TxNxH
                out = out.log_softmax(2)
                loss = self.criterion(out, self.target, output_sizes, self.target_lengths)
                
                self.model.zero_grad()
                loss.backward()
                data_grad = data.grad.data

                data = self.pgd_attack(data, data_raw, epsilon, alpha, data_grad).detach_()
            perturbed_data = data
        elif attack_type == "C&W":
            max_iters = 1000
            learning_rate = 0.01
            kappa = 0.0  # Adjust this value for the desired level of misclassification confidence
            perturbed_data = self.cw_attack(data, data_raw, target, max_iters, learning_rate, kappa)
        ############ ATTACK GENERATION ##############

        # prediction of adversarial sound
        spec = torch_spectrogram(perturbed_data, self.torch_stft)
        input_sizes = torch.IntTensor([spec.size(3)]).int()
        out, output_sizes, hs = self.model(spec, input_sizes)
        decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
        final_output = decoded_output[0][0]
        
        perturbed_data = perturbed_data.detach()
        abs_ori = 20*np.log10(np.sqrt(np.mean(np.absolute(data_raw.cpu().numpy())**2)))
        abs_after = 20*np.log10(np.sqrt(np.mean(np.absolute(perturbed_data.cpu().numpy())**2)))
        db_difference = abs_after-abs_ori
        l_distance = Levenshtein.distance(self.target_string, final_output)
        print(f"Max Decibel Difference: {db_difference:.4f}")
        print(f"Adversarial prediction: {decoded_output[0][0]}")
        print(f"Levenshtein Distance {l_distance}")
        if self.save:
            torchaudio.save(self.save, src=perturbed_data.cpu(), sample_rate=self.sample_rate)
        self.perturbed_data = perturbed_data
        return db_difference, l_distance, self.target_string, final_output
    

class SpeechBrainAttacker(Attacker):
    def __init__(self, model, sound, target, decoder, sample_rate=16000, device="cpu", save=None):
        """
        model: deepspeech model
        sound: raw sound data [-1 to +1] (read from torchaudio.load)
        label: string
        """
        self.sound = sound
        self.sample_rate = sample_rate
        self.target_string = target
        self.target = target
        self.__init_target()
        
        self.model = model
        # self.model.to(device)
        # self.model.train()
        self.decoder = decoder
        self.criterion = nn.CTCLoss()
        self.device = device
        n_fft = int(self.sample_rate * 0.02)
        hop_length = int(self.sample_rate * 0.01)
        win_length = int(self.sample_rate * 0.02)
        self.torch_stft = STFT(n_fft=n_fft , hop_length=hop_length, win_length=win_length ,  window='hamming', center=True, pad_mode='reflect', freeze_parameters=True, device=self.device)
        self.save = save

    def get_ori_spec(self, save=None):
        spec = torch_spectrogram(self.sound.to(self.device), self.torch_stft)
        plt.imshow(spec.cpu().numpy()[0][0])
        if save:
            plt.savefig(save)
            plt.clf()
        else:
            plt.show()

    def get_adv_spec(self, save=None):
        spec = torch_spectrogram(self.perturbed_data.to(self.device), self.torch_stft)
        plt.imshow(spec.cpu().numpy()[0][0])
        if save:
            plt.savefig(save)
            plt.clf()
        else:
            plt.show()
    
    # prepare
    def __init_target(self):
        self.target = target_sentence_to_label(self.target)
        self.target = self.target.view(1,-1)
        self.target_lengths = torch.IntTensor([self.target.shape[1]]).view(1,-1)

    # FGSM
    def fgsm_attack(self, sound, epsilon, data_grad):
        
        # find direction of gradient
        sign_data_grad = data_grad.sign()
        
        # add noise "epilon * direction" to the ori sound
        perturbed_sound = sound - epsilon * sign_data_grad
        
        return perturbed_sound
    
    # PGD
    def pgd_attack(self, sound, ori_sound, eps, alpha, data_grad) :
        
        adv_sound = sound - alpha * data_grad.sign() # + -> - !!!
        eta = torch.clamp(adv_sound - ori_sound.data, min=-eps, max=eps)
        sound = ori_sound + eta

        return sound
    
    def attack(self, epsilon, alpha, attack_type = "FGSM", PGD_round=40):
        print("Start attack")
        
        data, target = self.sound.to(self.device), self.target.to(self.device)
        data_raw = data.clone().detach()
        
        # initial prediction
        spec = torch_spectrogram(data, self.torch_stft)
        input_sizes = torch.IntTensor([spec.size(3)]).int()
        p_seq, wav_lens, p_tokens = self.model.compute_forward(spec, sb.Stage.VALID)
        # decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
        # original_output = decoded_output[0][0]
        print(f"Original prediction: {p_seq}")
        
        # ATTACK
        ############ ATTACK GENERATION ##############
        if attack_type == "FGSM":
            data.requires_grad = True
            
            spec = torch_spectrogram(data, self.torch_stft)
            input_sizes = torch.IntTensor([spec.size(3)]).int()
            # p_seq, wav_lens, p_tokens = self.model.compute_forward(spec, sb.Stage.TRAIN)
            # out = out.transpose(0, 1)  # TxNxH
            # out = out.log_softmax(2)
            # loss = self.criterion(out, self.target, output_sizes, self.target_lengths)
            
            # self.model.zero_grad()
            # loss.backward()
            self.model.fit_batch(spec)

            data_grad = data.grad.data

            perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

        elif attack_type == "PGD":
            for i in range(PGD_round):
                print(f"PGD processing ...  {i+1} / {PGD_round}", end="\r")
                data.requires_grad = True
                
                spec = torch_spectrogram(data, self.torch_stft)
                input_sizes = torch.IntTensor([spec.size(3)]).int()
                out, output_sizes, hs = self.model(spec, input_sizes)
                out = out.transpose(0, 1)  # TxNxH
                out = out.log_softmax(2)
                loss = self.criterion(out, self.target, output_sizes, self.target_lengths)
                
                self.model.zero_grad()
                loss.backward()


                data_grad = data.grad.data

                data = self.pgd_attack(data, data_raw, epsilon, alpha, data_grad).detach_()
            perturbed_data = data
        ############ ATTACK GENERATION ##############

        # prediction of adversarial sound
        spec = torch_spectrogram(perturbed_data, self.torch_stft)
        input_sizes = torch.IntTensor([spec.size(3)]).int()
        # out, output_sizes, hs = self.model(spec, input_sizes)
        # decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
        # final_output = decoded_output[0][0]

        p_seq, wav_lens, p_tokens = self.model.compute_forward(spec, sb.Stage.VALID)
        final_output = p_seq
        
        perturbed_data = perturbed_data.detach()
        abs_ori = 20*np.log10(np.sqrt(np.mean(np.absolute(data_raw.cpu().numpy())**2)))
        abs_after = 20*np.log10(np.sqrt(np.mean(np.absolute(perturbed_data.cpu().numpy())**2)))
        db_difference = abs_after-abs_ori
        l_distance = Levenshtein.distance(self.target_string, final_output)
        print(f"Max Decibel Difference: {db_difference:.4f}")
        # print(f"Adversarial prediction: {decoded_output[0][0]}")
        print(f"Adversarial prediction: {final_output}")
        print(f"Levenshtein Distance {l_distance}")
        if self.save:
            torchaudio.save(self.save, src=perturbed_data.cpu(), sample_rate=self.sample_rate)
        self.perturbed_data = perturbed_data
        return db_difference, l_distance, self.target_string, final_output