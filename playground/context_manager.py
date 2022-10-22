import contextlib
import cv2 as cv


class VideoCaptureWrapper(cv.VideoCapture):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


@contextlib.contextmanager
def video_capture_wrapper(*args, **kwargs):
    vid_stream = cv.VideoCapture(*args, **kwargs)
    try:
        yield vid_stream
    finally:
        vid_stream.release()
