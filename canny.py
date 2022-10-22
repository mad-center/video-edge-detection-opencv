import logging
import os
import winsound
from pathlib import Path

import cv2 as cv
import moviepy.editor as mpy
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(f'cv version: {cv.__version__}')


def convert_video(input_video_path: str, output_video_path: str, overwrite: bool = False) -> None:
    if overwrite:
        _convert_video(input_video_path, output_video_path)
    else:
        output_video_exists = os.path.exists(output_video_path)
        if not output_video_exists:
            _convert_video(input_video_path, output_video_path)


def edge_detect(image: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # gray.shape: (1080, 1920)

    # 高斯平滑去噪: 为何高斯平滑后效果反而更差了？轮廓断开现象加剧。
    blurred = cv.GaussianBlur(gray, gaussian_ksize, 0, 0)
    # blurred.shape: (1080, 1920)

    xgrad = cv.Sobel(blurred, cv.CV_16SC1, 1, 0)
    ygrad = cv.Sobel(blurred, cv.CV_16SC1, 0, 1)

    # - apertureSize 表示 Sobel 算子的孔径大小。
    # - L2gradient 为计算图像梯度幅度（gradient magnitude）的标识。其默认值为 False。
    # 如果为 True，使用更精确的 L2 范数（即两个方向的导数的平方和再开方），否则使用 L1 范数（直接将两个方向导数的绝对值相加）。

    canny = cv.Canny(xgrad, ygrad, canny_low_threshold, canny_high_threshold)
    return canny


def recolor(image: np.ndarray) -> np.ndarray:
    w, h = image.shape
    r = np.ones((w, h, 3), dtype=np.uint8) * 255
    mask_0 = (image[:, :] == 0)
    mask_255 = (image[:, :] == 255)
    r[mask_0] = bg_color_rgb
    r[mask_255] = line_color_rgb
    return r


def morphological_dilation(image):
    # kernel more bigger, line more thicker.
    kernel = np.ones(dilation_ksize, np.uint8)
    dilation = cv.dilate(image, kernel, iterations=1)
    return dilation


def _convert_video(input_video_path: str, output_video_path: str) -> None:
    _make_sure_file_exists(input_video_path)
    video = cv.VideoCapture(input_video_path)

    # video meta info
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv.CAP_PROP_FPS)
    total_frames = video.get(cv.CAP_PROP_FRAME_COUNT)
    print(f'width: {width}; height: {height}; fps: {fps}; total_frames: {total_frames}.')

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    video_writer = cv.VideoWriter(output_video_path, fourcc, fps, (width, height), True)

    frame_start_index = 0
    video.set(cv.CAP_PROP_POS_FRAMES, frame_start_index)

    frame_current_index = 0

    pbar = tqdm(total=int(total_frames), desc="Processing frame...")

    while video.isOpened():
        result, frame = video.read()
        # ! 注意这里 frame 的形状中 w,h 顺序会交换。即：w,h,z => h,w,z
        # 例如 video shape: (1920,1080,3) => frame shape: (1080, 1920, 3)

        if result:
            pbar.update(1)
            frame_current_index += 1

            canny_frame = edge_detect(frame)
            dilation_frame = morphological_dilation(canny_frame)
            recolor_frame = recolor(dilation_frame)
            video_writer.write(recolor_frame)
        else:
            break

    # cleanup
    pbar.close()
    video_writer.release()
    video.release()


def extract_audio(video_path: str, output_audio_path: str, overwrite: bool = False) -> None:
    if overwrite:
        _extract_audio(output_audio_path, video_path)
    else:
        audio_exists = os.path.exists(output_audio_path)
        if not audio_exists:
            _extract_audio(output_audio_path, video_path)


def _extract_audio(output_audio_path: str, video_path: str) -> None:
    audio_track = mpy.AudioFileClip(video_path)
    audio_track.write_audiofile(output_audio_path)


def play(video_path: str, audio_path: str) -> None:
    _make_sure_file_exists(video_path) and _make_sure_file_exists(audio_path)

    video = cv.VideoCapture(video_path)
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv.CAP_PROP_FPS)

    cv.namedWindow('play', 0)
    cv.resizeWindow('play', int(width / 2), int(height / 2))

    # audio here must be async
    winsound.PlaySound(audio_path, flags=winsound.SND_ASYNC)
    while video.isOpened():
        result, frame = video.read()
        if result:
            cv.imshow('play', frame)
        else:
            logger.info('play video and audio done.')
            break

        # cv.wait time(ms) * fps =1s => wait time= 1000ms/fps=1000/30=33.33ms
        # but cv.waitKey(delay=None) only accepts integer type as delay ???
        # so the audio doesn't synchronize with video when playing
        wait_time = int(1000 * (1 / fps))

        # 27 is "esc" key
        # for more detail, check https://stackoverflow.com/a/39201163
        if cv.waitKey(wait_time) & 0xFF == 27:
            # https://docs.python.org/3/library/winsound.html#winsound.SND_PURGE
            # windows platform doesn't support SND_PURGE flag
            winsound.PlaySound(None, flags=winsound.SND_ASYNC)
            break

    video.release()
    cv.destroyAllWindows()


def _make_sure_file_exists(file_path: str) -> None:
    file_path_exists = os.path.exists(file_path)
    if not file_path_exists:
        raise FileNotFoundError(f'The file {file_path} does not exist.')


def write_video(video_path: str, audio_path: str) -> None:
    _make_sure_file_exists(video_path) and _make_sure_file_exists(audio_path)

    # define output video name for duplicate processing checking.
    video_filename = video_path.rsplit(".", maxsplit=1)[0]
    video_file_ext = video_path.rsplit(".", maxsplit=1)[1]
    output_video_path = f'{video_filename}-[add-audio].{video_file_ext}'

    output_video_exists = os.path.exists(output_video_path)
    if not output_video_exists:
        video_clip = mpy.VideoFileClip(video_path)
        audio_clip = mpy.AudioFileClip(audio_path)
        video_add_audio_clip = video_clip.set_audio(audio_clip)
        video = mpy.CompositeVideoClip([video_add_audio_clip])

        # DON'T specify any audio_codec like audio_codec="pcm_s16le"
        video.write_videofile(output_video_path)

        audio_clip.close()
        video_clip.close()
        video.close()


if __name__ == '__main__':
    # =======================API params(用户参数)=======================
    # 高斯模糊的核大小，必须为 (3,3) 或(5,5)或(7,7)
    gaussian_ksize = (3, 3)
    # 输出视频的背景颜色，RGB 格式，必须为数组格式
    bg_color_rgb = [237, 244, 247]
    # 输出视频的线条颜色，RGB 格式，必须为数组格式
    line_color_rgb = [173, 155, 236]
    # canny 低阈值
    canny_low_threshold = 30
    # canny 高阈值，一般为低阈值的 2 或 3 倍
    canny_high_threshold = 90
    # 形态学膨胀核，修复 canny 边缘检测后的线条太细的问题, 核越大，线条越粗。
    dilation_ksize = (2, 2)
    # 输入视频的路径
    input_video_path = "materials/videos/2007-autumn-anime-spot-op.mp4"
    # 输出视频的文件夹
    output_folder = "output"
    # ================================================================

    # Derived variables
    filename_noext = Path(input_video_path).stem
    converted_video_path = f"{output_folder}/{filename_noext}-[canny].mp4"
    output_audio_path = f"{output_folder}/{filename_noext}.wav"  # must be wav format

    # Actions
    convert_video(input_video_path, converted_video_path, overwrite=False)
    extract_audio(input_video_path, output_audio_path, overwrite=False)

    # preview video after saving video edge detection.
    # play(converted_video_path, output_audio_path)

    # write new video
    write_video(converted_video_path, output_audio_path)
