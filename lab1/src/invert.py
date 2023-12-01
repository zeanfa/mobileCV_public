# ITMO University
# Mobile Computer Vision course
# 2020
# by Aleksei Denisov
# denisov@itmo.ru

import cv2
import numpy as np
import time

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def my_adaptive_thresh_mean(img, region, C=5):
    max_r = img.shape[0]
    max_c = img.shape[1]
    half_region = (region - 1) / 2
    res_img = []
    for r in range(max_r):
        new_line = []
        start_r = int(0 if (r - half_region) < 0 else r - half_region)
        end_r = int(r + half_region if (r + half_region) < max_r else max_r)
        for c in range(max_c):
            start_c = int(0 if (c - half_region) < 0 else c - half_region)
            end_c = int(c + half_region if (c + half_region) < max_c else max_c)
            region = img[start_r:end_r, start_c:end_c]
            treshold = region.mean() + C
            adaptive = 255 if img[r, c] < treshold else 0
            new_line.append(np.uint8(adaptive))
        res_img.append(new_line)
    return np.array(res_img)


def show_video():
    cap = cv2.VideoCapture('lab1_10s.mp4')
    need_save = True

    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = time.perf_counter()
    width = cap.get(3)
    height = cap.get(4)

    while cap.isOpened():
        try:
            ret_val, frame = cap.read()

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Show video
            cv2.imshow('Original', frame)

            numpy_thresh9 = my_adaptive_thresh_mean(img, 9, 5)
            cv2.imshow('Numpy Adaptiv lab 9', numpy_thresh9)

            numpy_thresh299 = my_adaptive_thresh_mean(img, 299, 5)
            cv2.imshow('Numpy Adaptiv lab 299', numpy_thresh299)

            if need_save:
                cv2.imwrite('numpy_adaptive_9.jpg', numpy_thresh9)
                cv2.imwrite('numpy_adaptive_299.jpg', numpy_thresh299)
                need_save = False

            keyCode = cv2.waitKey(1) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        except:
            cap.release()
            # cv2.destroyAllWindows()

    end_time = time.perf_counter()
    result = f"""
    Frame resolution: {width} х {height}
    Execution time: {end_time - start_time:.4f}s
    Number of frames: {number_of_frames:.4f}
    Frames per second: {number_of_frames / (end_time - start_time):.4f}
    Second for one frame: {(end_time - start_time) / number_of_frames:.4f}"""

    with open('numpy_lib_results.txt', 'w') as f:
        f.write(result)

    print(result)


if __name__ == "__main__":
    show_video()
