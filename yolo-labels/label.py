# input is a video
# divide video into all frames
# process frames one at a time, with labeling
# click for top left, bottom right
# space for next label
# enter for next frame
# r for redo

from __future__ import annotations
from dataclasses import dataclass
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backend_bases import MouseButton

@dataclass
class Label:
    b_class: int # will always be 0 bc only one class "sample"
    x_center: float
    y_center: float
    width: float
    height: float

    def __str__(self):
        return f"{self.b_class} {np.round(self.x_center, 6)} {np.round(self.y_center, 6)} {np.round(self.width, 6)} {np.round(self.height, 6)}"

    def __repr__(self):
        return f"{self.b_class} {np.round(self.x_center, 6)} {np.round(self.y_center, 6)} {np.round(self.width, 6)} {np.round(self.height, 6)}"

    def save_labels(labels: list[Label], output: Path):
        with open(output, 'w') as f:
            for label in labels:
                f.write(str(label) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', help='video file')
    parser.add_argument('frames', help='frame directory')
    parser.add_argument('labels', help='labels directory')
    parser.add_argument("--save_frames", help="whether to save frames", action="store_true")
    parser.add_argument("--start", help="start frame", type=int, default=0)
    args = parser.parse_args()

    video_path = Path(args.video)
    frames_path = Path(args.frames)
    labels_path = Path(args.labels)

    if args.save_frames:
        save_frames(video_path, frames_path)

    print(f"Video: {video_path}, Frame Path: {frames_path}")

    num_frames = len(list(frames_path.iterdir()))

    for i in range(args.start, num_frames, 7):
        frame = frames_path / f"{i}.jpg"
        print(f"Processing frame {frame}")
        try:
            labels = label_frame(frame)
        except Exception as e:
            print(f"Error labeling frame {frame}: {e}")
            continue
        if labels != []:
            label_path = labels_path / f"{frame.stem}.txt"
            Label.save_labels(labels, label_path)

def save_frames(video: Path, output: Path):
    cap = cv2.VideoCapture(str(video))
    frame_count = 0
    
    output.mkdir(parents=True, exist_ok=True)

    frames_tqdm = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_path = output / f"{frame_count}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frames_tqdm.update(1)
    cap.release()
    print(f"Saved {frame_count} frames to {output}")

def label_frame(frame: Path) -> list[Label]:
    frame = cv2.imread(str(frame))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    labels: list[Label] = []
    rects: list[patches.Rectangle] = []

    while True:
        fig, ax = plt.subplots(1, figsize=(10, 6))

        for rect in rects:
            ax.add_patch(patches.Rectangle((rect[0], rect[1]), 
                                          rect[2], 
                                          rect[3], 
                                          linewidth=1, 
                                          edgecolor='r', 
                                          facecolor='none'))
        ax.imshow(frame)

        coords: list[tuple] = plt.ginput(2)

        # Enter key breaks ginput and returns whatever was saved
        if len(coords) == 0:
            plt.close()
            break

        x = min(coords[0][0], coords[1][0])
        y = min(coords[0][1], coords[1][1])

        width = abs(coords[0][0] - coords[1][0])
        height = abs(coords[0][1] - coords[1][1])

        x_center = (x + width / 2) / frame.shape[1]
        y_center = (y + height / 2) / frame.shape[0]
        width_norm = width / frame.shape[1]
        height_norm = height / frame.shape[0]

        label = Label(0, x_center, y_center, width_norm, height_norm)
        labels.append(label)
        rects.append([x, y, width, height])

        for rect in rects:
            ax.add_patch(patches.Rectangle((rect[0], rect[1]), 
                                          rect[2], 
                                          rect[3], 
                                          linewidth=1, 
                                          edgecolor='r', 
                                          facecolor='none'))

        ax.imshow(frame)
        plt.show()

    return labels


if __name__ == "__main__":
    main()
