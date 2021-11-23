""""
Dataset stores all the information about a dataset
Dataloader is used to load and manage the data for training
The loader wraps the dataset
"""
import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd


class EigenScapeVideoDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 video_directory,
                 transformation=None,
                 device='cpu'):
        self.device = device
        self.annotations = pd.read_csv(annotations_file)
        self.video_dir = video_directory
        self.transformation = transformation

    def __len__(self):  # len(my_dataset_object)
        return len(self.annotations)  # self.num_files

    def __getitem__(self, index):  # a_list[1] -> a_list.__getitem__(1)
        video_path = self._get_video_path(index)
        if video_path == -1:
            return torch.Tensor(1), -1
        label = self._get_video_label(index)
        signal = torchvision.io.read_video(video_path)
        signal = signal.to(self.device)
        if self.transformation is None:
            return signal, label
        transformed_video = self.transformation(signal)
        return transformed_video, label

    def _get_video_path(self, index):
        file = self.annotations.iloc[index, 1]
        path = os.path.join(self.video_dir, file)
        if os.path.isfile(path) is False:
            if path[-1] == "\n":
                path = path[:-1]
            if path[-1] == "\r":
                path = path[:-1]
        if os.path.isfile(path) is True:
            return path
        else:
            return -1

    def _get_video_label(self, index):
        return self.annotations.iloc[index, 2]


if __name__ == "__main__":
    VIDEO_DIR = "/media/sedur/data/datasets/lib_EigenScape/images/pool"
    VIDEO_DEST_DIR = "/media/sedur/data/datasets/lib_EigenScape/images/result"
    ANNOTATIONS_FILE = os.path.join(os.getcwd(), "EigenScape.csv")
    NUM_FRAMES = 1
    print(os.environ['PATH'])
    # if torch.cuda.is_available():
    #     device = "cuda"
    #     torch.multiprocessing.set_start_method("spawn")
    # else:
    #     device = "cpu"
    device = "cpu"

    print(f"Using device: {device}")

    # mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
    #                                                        n_fft=1024,
    #                                                        hop_length=512,
    #                                                        n_mels=64
    #                                                        )

    esid = EigenScapeImageDataset(ANNOTATIONS_FILE,
                                 VIDEO_DIR,
                                 None,
                                 NUM_FRAMES,
                                 device)
    print(f"There are {len(esid)} samples in the dataset.")
    fps = 30.0
    targ_length = 30
    # torchvision.io.write_video("testvideo.MP4",
    #                            new_signal,
    #                            fps)
    data_loader = DataLoader(esid,
                            batch_size=1,
                            num_workers=0)
    loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    for i, (data, label) in loop:
        # data = torch.unsqueeze(data, dim=0)
        video_name = label[0] + str(i) + ".MP4"
        video_string = os.path.join(VIDEO_DEST_DIR, video_name)
        if os.path.isfile(video_string) is True:
            loop.set_description(f"Epoch: [{i}/{len(esid)}]")
            loop.set_postfix_str(s=f"File {video_string} exists // bypassing", refresh=True)
            continue
        new_data = data.repeat(int(fps) * targ_length, 1, 1, 1)
        torchvision.io.write_video(video_string,
                                   new_data,
                                   fps)
        del new_data
        del data
        loop.set_description(f"Epoch: [{i}/{len(esid)}]")
        loop.set_postfix_str(s=f"File {video_string} exists // bypassing", refresh=True)

    # torchvision.io.write_video(
    #     filename: str,
    #     video_array: torch.Tensor,
    #     fps: float,
    #     video_codec: str = 'libx264',
    #     options: Optional[ Dict[str, Any]] = None,
    #     audio_array: Optional[torch.Tensor] = None,
    #     audio_fps: Optional[float] = None,
    #     audio_codec: Optional[str] = None,
    #     audio_options: Optional[Dict[str, Any]] = None