""""
Dataset stores all the information about a dataset
Dataloader is used to load and manage the data for training
The loader wraps the dataset
"""
import os
import torch
from torch import nn
from torch.utils.data import Dataset
import random
import torchaudio
import pandas as pd
import numpy as np
from scipy.io import wavfile


class EigenScapeDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_directory,
                 transformation=None,
                 target_sample_rate=None,
                 target_num_samples=None,
                 target_num_channels=None,
                 device='cpu'):
        """

        Args:
            annotations_file:
            audio_directory:
            transformation:
            target_sample_rate:
            target_num_samples:
            target_num_channels:
            device:
        """
        self.device = device
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_directory
        if transformation is not None:
            self.transformation = transformation.to(self.device)
        else:
            self.transformation = None
        self.target_sample_rate = target_sample_rate
        self.target_num_samples = target_num_samples
        self.target_num_channels = target_num_channels
        self.min = 0
        self.max = 1

    def __len__(self):  # len(my_dataset_object)
        return len(self.annotations)  # self.num_files

    def __getitem__(self, index):  # a_list[1] -> a_list.__getitem__(1)
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path,
                                     normalize=True)
        # Send signal to device
        signal = signal.to(self.device)
        # signal -> (channels, samples) -> (1, 16000)
        signal = self._resample_if_needed(signal, sr)
        # signal -> (channels, samples) -> (2, 16000)
        signal = self._mix_down_if_needed(signal)
        signal = self._cut_if_needed(signal)
        signal = self._right_pad_if_needed(signal)
        if self.transformation is not None:
            signal = self.transformation(signal)
        return signal, label

    def _get_audio_sample_path(self, index):
        file = self.annotations.iloc[index, 0]
        path = os.path.join(self.audio_dir, file)
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 1]

    def _resample_if_needed(self, signal, sr):
        if self.target_sample_rate is None:
            return signal
        if sr == self.target_sample_rate:
            return signal
        resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
        resampler.to(self.device)
        signal = resampler(signal)
        return signal

    def _mix_down_if_needed(self, signal):
        if self.target_num_channels == None:
            return signal
        if signal.shape[0] == self.target_num_channels:
            return signal
        if signal.shape[0] == 1:
            return signal
        if self.target_num_channels == 1:
            return signal[0,:]
        signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_needed(self, signal):
        # signal -> Tensor -> (channels, samples)
        if self.target_num_samples is None:
            return signal
        if signal.shape[1] > self.target_num_samples:
            signal = signal[:, :self.target_num_samples]
        return signal

    def _right_pad_if_needed(self, signal):
        if self.target_num_samples is None:
            return signal
        signal_length = signal.shape[1]
        if signal_length < self.target_num_samples:
            pad_size = self.target_num_samples - signal_length
            last_dim_padding = (0, pad_size)  # (pre_pad_num, post_pad_num) paired over dims
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _normalise(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def _denormalise(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array


def partition(input, frame_size):
    num_partitions = int(np.floor(input.size(-1) / frame_size))
    partitions = torch.hsplit(input, num_partitions)
    return partitions


def get_next_file_name(file_path):
    file_name = f'{torch.randint(16 ** 6, (1,)).numpy()[0]:x}.wav'
    check_path = os.path.join(file_path, file_name)
    if os.path.exists(check_path) is True:
        file_name = get_next_file_name(file_path)
    return file_name


def apply_fade(signal, fade_length):
    n_chans = signal.size(0)
    ramp = torch.Tensor(np.tile(10**(np.arange(-90, 0, 90/fade_length, dtype=float) / float(20)), (n_chans, 1)))
    signal[:, :fade_length] *= ramp
    signal[:, -fade_length:] *= ramp.fliplr()
    return signal


def prepare_dataset(target_dir, target_annotations, source_dir, source_annotations, sub_sample_size, sub_sample_rate, n_folds, n_channels, device):
    torch.manual_seed(42)  # Repeatability
    if os.path.isdir(target_dir) is False:
        os.mkdir(target_dir)
    elif len(os.listdir(target_dir)) != 0:
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                os.remove(os.path.join(root, file))

    data_set = EigenScapeDataset(source_annotations,
                                 source_dir,
                                 None,
                                 sub_sample_rate,
                                 None,
                                 n_channels,
                                 device)
    labels = []
    file_names = []
    recording_numbers = []
    clip_numbers = []
    for i, (data, label) in enumerate(data_set):

        partitions = partition(data, sub_sample_size)
        for j, part in enumerate(partitions):
            labels.append(label)
            recording_numbers.append(i % 8)
            clip_numbers.append(j)
            file_name = get_next_file_name(target_dir)
            file_names.append(file_name)
            file_path = os.path.join(target_dir, file_name)
            part = torchaudio.functional.highpass_biquad(waveform=part,
                                                         sample_rate=48000,
                                                         cutoff_freq=100.,
                                                         Q=0.707)
            part = torchaudio.functional.highpass_biquad(waveform=part,
                                                         sample_rate=48000,
                                                         cutoff_freq=100.,
                                                         Q=0.707)
            part = apply_fade(part, 1*48000)
            wavfile.write(file_path, sub_sample_rate, part.detach().to('cpu').numpy().transpose())
    folds = torch.FloatTensor(1, len(labels)).uniform_(0, n_folds).numpy().astype('int').tolist()[0]
    zipped = list(zip(file_names, labels, folds, recording_numbers, clip_numbers))
    annotations = pd.DataFrame(zipped, columns=['file_name', 'label', 'fold', 'recording', 'clip'])
    annotations.to_csv(target_annotations, index=False)
    return True


if __name__ == "__main__":
    AUDIO_DIR = "D:\\Data\\lib_EigenScape\\raw_audio\\first_order"
    ANNOTATIONS_FILE = "D:\\Data\\lib_EigenScape\\raw_audio\\annotations.csv"
    TARGET_DIR = "D:\\Data\\lib_EigenScape\\split"
    TARGET_ANNOTATIONS = "D:\\Data\\lib_EigenScape\\split\\annotations.csv"
    SAMPLE_RATE = 48000
    T = 60
    num_chans = 4
    NUM_SAMPLES = SAMPLE_RATE * T  # 22050
    print(os.environ['PATH'])
    # if torch.cuda.is_available():
    #     device = "cuda"
    # else:
    #     device = "cpu"
    device = "cpu"
    print(f"Using device: {device}")
    prepare_dataset(TARGET_DIR, TARGET_ANNOTATIONS,
                    AUDIO_DIR, ANNOTATIONS_FILE,
                    NUM_SAMPLES, SAMPLE_RATE,
                    0, num_chans, device)
    # mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
    #                                                        n_fft=1024,
    #                                                        hop_length=512,
    #                                                        n_mels=64
    #                                                        )
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=1024,
                                                    hop_length=512,
                                                    normalized=True
                                                    )
    transforms = nn.Sequential(spectrogram)
    data_set = EigenScapeDataset(TARGET_ANNOTATIONS,
                                 TARGET_DIR,
                                 transformation=transforms,
                                 device=device)
    print(f"There are {len(data_set)} samples in the dataset.")
    signal, label = data_set[1]
    print(signal.size())