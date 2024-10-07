import os
import torch
import torchaudio
import numpy as np
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


class VaeAudioProcessor():
    def __init__(
        self,
        sampling_rate: int = 16000,
        max_wav_value: float = 32768.0,
        n_mel_channels: int = 64,
        hop_length: int = 160,
        duration: float = 10.24,
        waveform_only: bool = False,
        mel_fmax: int = 8000,
        mel_fmin: int = 0,
        filter_length: int = 1024,
        win_length: int = 1024,
        
    ):
        super().__init__()
        # audio
        self.sampling_rate = sampling_rate
        self.max_wav_value = max_wav_value
        self.duration = duration 
        
        # stft
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length

        # mel
        self.n_mel = n_mel_channels
        self.mel_fmax = mel_fmax
        self.mel_fmin = mel_fmin
        
        self.hopsize = hop_length
        self.target_length = int(self.duration * self.sampling_rate / self.hopsize)
        self.pad_wav_start_sample = 0  # If none, random choose
        self.waveform_only = waveform_only

        self.mel_basis = {}
        self.hann_window = {}

        self.trim_wav = False
        
    def pad_wav(self, waveform, target_length):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        if waveform_length == target_length:
            return waveform

        # Pad
        temp_wav = np.zeros((1, target_length), dtype=np.float32)
        if self.pad_wav_start_sample is None:
            rand_start = int(self.random_uniform(0, target_length - waveform_length))
        else:
            rand_start = 0

        temp_wav[:, rand_start : rand_start + waveform_length] = waveform
        return temp_wav

    def trim_wav(self, waveform):
        if np.max(np.abs(waveform)) < 0.0001:
            return waveform

        def detect_leading_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = 0
            while start + chunk_size < waveform_length:
                if np.max(np.abs(waveform[start : start + chunk_size])) < threshold:
                    start += chunk_size
                else:
                    break
            return start
    
        def detect_ending_silence(waveform, threshold=0.0001):
                chunk_size = 1000
                waveform_length = waveform.shape[0]
                start = waveform_length
                while start - chunk_size > 0:
                    if np.max(np.abs(waveform[start - chunk_size : start])) < threshold:
                        start -= chunk_size
                    else:
                        break
                if start == waveform_length:
                    return start
                else:
                    return start + chunk_size

        start = detect_leading_silence(waveform)
        end = detect_ending_silence(waveform)

        return waveform[start:end]

    def read_wav_file(self, filename):
        # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
        waveform, sr = torchaudio.load(filename)

        waveform, random_start = self.random_segment_wav(
            waveform, target_length=int(sr * self.duration)
        )

        waveform = self.resample(waveform, sr)
        # random_start = int(random_start * (self.sampling_rate / sr))

        waveform = waveform.numpy()[0, ...]

        waveform = self.normalize_wav(waveform)

        if self.trim_wav:
            waveform = self.trim_wav(waveform)

        waveform = waveform[None, ...]
        waveform = self.pad_wav(
            waveform, target_length=int(self.sampling_rate * self.duration)
        )
        return waveform, random_start

    def read_audio_file(self, filename):
        if os.path.exists(filename):
            waveform, random_start = self.read_wav_file(filename)
        else:
            print(
                'Non-fatal Warning [dataset.py]: The wav path "',
                filename,
                '" is not find in the metadata. Use empty waveform instead. This is normal in the inference process.',
            )
            target_length = int(self.sampling_rate * self.duration)
            waveform = torch.zeros((1, target_length))
            random_start = 0

        # log_mel_spec, stft = self.wav_feature_extraction_torchaudio(waveform) # this line is faster, but this implementation is not aligned with HiFi-GAN
        if not self.waveform_only:
            log_mel_spec, stft = self.wav_feature_extraction(waveform)
        else:
            # Load waveform data only
            # Use zero array to keep the format unified
            log_mel_spec, stft = None, None

        return log_mel_spec, stft, waveform, random_start

    def mel_spectrogram_train(self, y):
        if torch.min(y) < -1.0:
            print("train min value is ", torch.min(y))
        if torch.max(y) > 1.0:
            print("train max value is ", torch.max(y))

        if self.mel_fmax not in self.mel_basis:
            mel = librosa_mel_fn(
                sr=self.sampling_rate,
                n_fft=self.filter_length,
                n_mels=self.n_mel,
                fmin=self.mel_fmin,
                fmax=self.mel_fmax,
            )
            self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)] = (
                torch.from_numpy(mel).float().to(y.device)
            )
            self.hann_window[str(y.device)] = torch.hann_window(self.win_length).to(
                y.device
            )

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.filter_length - self.hop_length) / 2),
                int((self.filter_length - self.hop_length) / 2),
            ),
            mode="reflect",
        )

        y = y.squeeze(1)

        stft_spec = torch.stft(
            y,
            self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[str(y.device)],
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        stft_spec = torch.abs(stft_spec)

        mel = spectral_normalize_torch(
            torch.matmul(
                self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)], stft_spec
            )
        )

        return mel[0], stft_spec[0]
    
    def wav_feature_extraction(self, waveform):
        waveform = waveform[0, ...]
        waveform = torch.FloatTensor(waveform)

        # log_mel_spec, stft, energy = Audio.tools.get_mel_from_wav(waveform, self.STFT)[0]
        log_mel_spec, stft = self.mel_spectrogram_train(waveform.unsqueeze(0))

        log_mel_spec = torch.FloatTensor(log_mel_spec.T)
        stft = torch.FloatTensor(stft.T)

        log_mel_spec, stft = self.pad_spec(log_mel_spec), self.pad_spec(stft)
        return log_mel_spec, stft

    def random_uniform(self, start, end):
        val = torch.rand(1).item()
        return start + (end - start) * val
    
    def resample(self, waveform, sr):
        waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
        return waveform

    def normalize_wav(self, waveform):
        waveform = waveform - np.mean(waveform)
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        return waveform * 0.5  # Manually limit the maximum amplitude into 0.5
    
    def random_segment_wav(self, waveform, target_length):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        # Too short
        if (waveform_length - target_length) <= 0:
            return waveform, 0

        for i in range(10):
            random_start = int(self.random_uniform(0, waveform_length - target_length))
            if torch.max(
                torch.abs(waveform[:, random_start : random_start + target_length])
                > 1e-4
            ):
                break

        return waveform[:, random_start : random_start + target_length], random_start
    
    def pad_spec(self, log_mel_spec):
        n_frames = log_mel_spec.shape[0]
        p = self.target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            log_mel_spec = m(log_mel_spec)
        elif p < 0:
            log_mel_spec = log_mel_spec[0 : self.target_length, :]

        if log_mel_spec.size(-1) % 2 != 0:
            log_mel_spec = log_mel_spec[..., :-1]

        return log_mel_spec
    
    def feature_extraction(self, audio_path):

        # Read wave file and extract feature
        while True:
            try:
                (
                    log_mel_spec,
                    stft,
                    waveform,
                    random_start,
                ) = self.read_audio_file(audio_path)
                break
            except Exception as e:
                print(
                    "Error encounter during audio feature extraction: ", e, audio_path
                )
                continue

        # The filename of the wav file
        fname = audio_path
        waveform = torch.FloatTensor(waveform)

        return (
            fname,
            waveform,
            stft,
            log_mel_spec,
            random_start,
        )
    
    def preprocess(self, audio):
        (
            fname,
            waveform,
            stft,
            log_mel_spec,
            # the metadata of the sampled audio file and the mixup audio file (if exist)
            random_start,
        ) = self.feature_extraction(audio)

        data = {
            # tensor, [batchsize, 1, samples_num]
            "fname": fname,  # list
            "waveform": "" if (waveform is None) else waveform.float(),
            # tensor, [batchsize, t-steps, f-bins]
            "stft": "" if (stft is None) else stft.float(),
            # tensor, [batchsize, t-steps, mel-bins]
            "log_mel_spec": "" if (log_mel_spec is None) else log_mel_spec.float(),
            "duration": self.duration,
            "sampling_rate": self.sampling_rate,
            "random_start_sample_in_original_audio_file": random_start,
        }
        return data
    



if __name__ == "__main__":
    import torch
    from tqdm import tqdm

    audio_processor = VaeAudioProcessor()
    audio_path = 'assets/test.wav'
    data = audio_processor.preprocess(audio_path)
    print(data.keys())
    print(data['waveform'].shape)
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(k, ": ", v.shape)
        else:
            print(k, ": ", v)
    
    # dict_keys(['fname', 'waveform', 'stft', 'log_mel_spec', 'duration', 'sampling_rate', 'random_start_sample_in_original_audio_file'])
    # torch.Size([1, 163840])
    # fname :  assets/bird_audio.wav
    # waveform :  torch.Size([1, 163840])
    # stft :  torch.Size([1024, 512])
    # log_mel_spec :  torch.Size([1024, 64])
    # duration :  10.24
    # sampling_rate :  16000
    # random_start_sample_in_original_audio_file :  0