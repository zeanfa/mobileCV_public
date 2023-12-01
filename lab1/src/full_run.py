from invert import show_video as numpy_video
from invert_cv_lib import show_video as cv2_video
from invert_njit import show_video as njit_video


if __name__ == "__main__":
    cv2_video()
    numpy_video()
    njit_video()
