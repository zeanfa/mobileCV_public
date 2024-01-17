#!/bin/env python3

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from supervision.draw.color import ColorPalette
from supervision.video.source import get_video_frames_generator
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.video.sink import VideoSink
from supervision.video.dataclasses import VideoInfo
import os
from tqdm import tqdm

best_model = torch.load('best-large.pth')
# best_model = torch.load('best-small.pth')
best_model.eval()


def check_predict(model, filename):
    # Загрузка и обработка фото
    X = []
    src = cv2.imread(filename, cv2.IMREAD_COLOR)
    dst = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    X.append(cv2.resize(dst, dsize=(200, 200), interpolation=cv2.INTER_AREA))
    X = np.array(X)
    X = X.astype('float32')
    X = X / 255.0
    X = X.reshape(-1, 3, 200, 200)
    X = torch.from_numpy(X).float()
    print(model(X).argmax(-1).item())


def predict_special(src):
    X = []
    dst = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    X.append(cv2.resize(dst, dsize=(200, 200), interpolation=cv2.INTER_AREA))
    X = np.array(X)
    X = X.astype('float32')
    X = X / 255.0
    X = X.reshape(-1, 3, 200, 200)
    X = torch.from_numpy(X).float()

    return 'Special car' if best_model(X).argmax(-1).item() > 0 else ' '


def run_small():
    best_model = torch.load('best-large.pth')
    best_model.eval()
    for r in os.listdir('../Dataset/Special_car_small/'):
        check_predict(best_model, '../Dataset/Special_car_small/' + r)

    for r in os.listdir('../Dataset/Rest_small/'):
        check_predict(best_model, '../Dataset/Rest_small/' + r)


def run():

    model = YOLO('yolov8x.pt')
    model.fuse()

    # 2: 'car'
    # 5: 'bus'
    # 7: 'truck'

    TARGET_CLASS_IDS = [2, 5, 7]
    for i in [4]:
        SOURCE_VIDEO_PATH = f"./{i}.mp4"
        # TARGET_VIDEO_PATH = f"./{i}-result-small.mp4"
        TARGET_VIDEO_PATH = f"./{i}-result-large.mp4"

        video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
        generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
        box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=1)

        # Файл для записи видео
        with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
            for frame in tqdm(generator, total=video_info.total_frames):
                results = model(frame)
                detections = Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int)
                )
                # оставляем только выбранные ID
                mask = np.array([class_id in TARGET_CLASS_IDS for class_id in detections.class_id], dtype=bool)
                detections.filter(mask=mask, inplace=True)

                labels = []
                for d in detections:
                    cords = d[0].astype(int)
                    detect_img = frame[cords[1]:cords[3], cords[0]:cords[2]]
                    labels.append(predict_special(detect_img))

                # запись фрейма в видео
                frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
                sink.write_frame(frame)


if __name__ == '__main__':
    run()
    # run_small()
