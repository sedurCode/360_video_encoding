# coding=utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# import spatialmedia
import os
from EigenScapeDataset import EigenScapeDataset
from EigenScapeVideoDataset import EigenScapeVideoDataset
import pandas as pd


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print("Hi, {0}".format(name))  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    video_dir = 'D:\\Data\\lib_EigenScape\\blurred'
    video_annotations_file = 'D:\\Data\\lib_EigenScape\\blurred\\annotations.csv'
    audio_dir = 'D:\\Data\\lib_EigenScape\\split'
    audio_annotations_file = 'D:\\Data\\lib_EigenScape\\split\\annotations.csv'
    dest_dir = 'D:\\Data\\lib_EigenScape\\rendered_video'
    # audio_data_set = EigenScapeDataset(audio_annotations_file,
    #                                    audio_dir,
    #                                    transformation=None,
    #                                    device='cpu')
    # video_data_set = EigenScapeVideoDataset(video_annotations_file,
    #                                         video_dir,
    #                                         transformation=None,
    #                                         device='cpu')
    video_annotations = pd.read_csv(video_annotations_file)
    audio_annotations = pd.read_csv(audio_annotations_file)
    for i in range(len(video_annotations)):
        source = video_annotations.loc[i, 'Filename']
        loc_name = source.replace(".wav", "").split("-")[0]
        loc_num = int(source.replace(".wav", "").split("-")[1])-1
        sample_name = f"{loc_name}{loc_num}"
        for j, row in audio_annotations.iterrows():
            this_sample = f"{row['label']}{row['recording']}"
            if this_sample == sample_name and row['clip'] == 0:
                output_file_name = os.path.join(dest_dir, f"{sample_name}.mov")
                if os.path.isfile(output_file_name):
                    a = 1
                    continue
                video_file = os.path.join(video_dir, f"{video_annotations.loc[i, 'Class']}{i}.mp4")
                if os.path.isfile(video_file) is False:
                    a = 1
                audio_file = os.path.join(audio_dir, row["file_name"])
                if os.path.isfile(audio_file) is False:
                    a = 1
                command_string = f"ffmpeg -i {video_file} -i {audio_file} -loop 1 -channel_layout 4.0 -map 1:a -map 0:v -c:a copy -c:v libx264 -framerate 30 {output_file_name}"
                # command_string = f"ffmpeg -i {video_file} -i {audio_file} -map 1:a -map 0:v -c:a copy -channel_layout 4.0 -c:v libx264 -b:v 40000k -bufsize 40000k -shortest PSandAmbiX.mov"
                os.system(command_string)
                inject_string = f"C:\\Python27\\python.exe spatialmedia  {output_file_name} {os.path.join(dest_dir, f'{sample_name}_injected.mov')} -a --i"
                os.system(inject_string)
                os.remove(output_file_name)
                print("woohoo")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
