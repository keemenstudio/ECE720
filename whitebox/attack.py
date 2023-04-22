from .stft import STFT, magphase
import torch.nn as nn
import torch
import Levenshtein
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import speechbrain as sb
import torch.optim as optim
from scipy.signal import gaussian
from scipy.signal import convolve

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
    def pgd_attack(self, sound, ori_sound, eps, alpha, data_grad, attack_range) :
        guassian_filter = gaussian(sound.shape[1], (attack_range[1] - attack_range[0]) // 2)
        guassian_filter /= np.sum(guassian_filter)
        guassian_filter *= 5000
        guassian_filter = np.roll(guassian_filter, (attack_range[1] + attack_range[0]) // 2 - sound.shape[1] // 2)
        guassian_filter_mean = np.mean(guassian_filter[attack_range[0] : attack_range[1]])
        guassian_filter[attack_range[0] : attack_range[1]].fill(guassian_filter_mean)
        grad_sign = data_grad.sign()
        # plt.plot(guassian_filter)
        # plt.savefig("guassian_filter.png")
        # grad_sign[:, :3720] *= alpha * 0.1
        # grad_sign[:, 3720:8840] *= alpha
        # grad_sign[:, 8840:] *= alpha * 0.1
        adv_sound = sound - alpha * grad_sign
        # adv_sound = sound - alpha * data_grad.sign() # + -> - !!!
        eta = torch.clamp(adv_sound - ori_sound.data, min=-eps, max=eps)
        # plt.plot(eta.detach().cpu().numpy()[0,:])
        # plt.savefig("eta.png")
        # filter_eta = convolve(eta.detach().cpu().numpy()[0,:], guassian_filter, mode="same")
        filter_eta = eta.detach().cpu().numpy()[0,:] * guassian_filter
        # plt.plot(filter_eta)
        # plt.savefig("filter_eta.png")
        # eta[:, :3720] *= 0.1
        # eta[:, 3720:8840] *= alpha
        # eta[:, 8840:] *= 0.1
        sound = ori_sound + torch.Tensor(filter_eta).to("cuda")

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
    
    def wer3(self, r, h):
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

    def attack(self, epsilon, alpha, attack_type = "FGSM", PGD_round=40, attack_range = []):
        print("Start attack")
        
        data, target = self.sound.to(self.device), self.target.to(self.device)
        data_raw = data.clone().detach()
        
        # initial prediction
        spec = torch_spectrogram(data, self.torch_stft)
        input_sizes = torch.IntTensor([spec.size(3)]).int()
        out, output_sizes = self.model(spec, input_sizes)
        decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
        original_output = decoded_output[0][0]
        print(f"Original prediction: {decoded_output[0][0]}")
        print(f"attack range {attack_range}")
        
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
                out, output_sizes = self.model(spec, input_sizes)

                decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
                # if len(self.target_string) > 0 and decoded_output[0][0] != self.target_string:
                AS = self.wer3(decoded_output[0][0], original_output)
                if AS == 1:
                    print(f"early stop! iteration {i}")
                    break
                out = out.transpose(0, 1)  # TxNxH
                out = out.log_softmax(2)
                loss = self.criterion(out, self.target, output_sizes, self.target_lengths)
                
                self.model.zero_grad()
                loss.backward()
                data_grad = data.grad.data

                data = self.pgd_attack(data, data_raw, epsilon, alpha, data_grad, attack_range).detach_()
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
        out, output_sizes = self.model(spec, input_sizes)
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
        return db_difference, l_distance, self.target_string, final_output, original_output, perturbed_data, data_raw, perturbed_data - data_raw
    

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