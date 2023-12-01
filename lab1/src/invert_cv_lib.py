# ITMO University
# Mobile Computer Vision course
# 2020
# by Aleksei Denisov
# denisov@itmo.ru

import cv2
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


def show_video():
    cap = cv2.VideoCapture('lab1_10s.mp4')
    need_save = True

    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) * 2
    start_time = time.perf_counter()
    width = cap.get(3)
    height = cap.get(4)

    while cap.isOpened():
        try:
            ret_val, frame = cap.read()

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            thresh9 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
            thresh299 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 299, 5)

            # Show video
            cv2.imshow('Original', frame)
            cv2.imshow('Adaptive mean 9', thresh9)
            cv2.imshow('Adaptive mean 299', thresh299)

            if need_save:
                cv2.imwrite('cv2_original.jpg', img)
                cv2.imwrite('cv2_adaptive_9.jpg', thresh9)
                cv2.imwrite('cv2_adaptive_299.jpg', thresh299)
                need_save = False

            # cv2.imshow('Adaptive mean 9', thresh3)
            # cv2.imshow('Adaptive mean 399', thresh4)
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

    with open('cv_lib_results.txt', 'w') as f:
        f.write(result)

    print(result)

if __name__ == "__main__":
    show_video()
