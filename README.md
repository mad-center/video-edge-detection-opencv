# video edge detection

Use opencv-python library to detect the edge of anime video.

## environment
- Python 3.7+

## setup
install dependencies:
```
pip install -r requirement_dev.txt
```

## usage

edit `canny.py` options as follows:
```python
# =======================API params(用户参数)=======================
# 高斯模糊的核大小，必须为 (3,3) 或 (5,5) 或(7,7)
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
# 输出视频的文件夹，手动创建保证存在
output_folder = "output"
# ================================================================
```
Then run command:
```bash
python canny.py
```
If all goes well without errors, you can see three files are generated in `output_folder`, similar to the following:
```
- 2007-autumn-anime-spot-op-[canny]-[add-audio].mp4
- 2007-autumn-anime-spot-op-[canny].mp4
- 2007-autumn-anime-spot-op.wav
```
The `2007-autumn-anime-spot-op-[canny]-[add-audio].mp4` is the final output result.

## documents

Some related documents are located in `docs` folder:

- [edge-detect-quality](docs/edge-detect-quality.md): how to increase canny detect quality.
- [opencv-get-started-notes.md](docs/opencv-get-started-notes.md): some basic opencv knowledge.

## LICENSE

Mixed LICENSE

- All codes except (`reimplement_canny.py`) are under MIT LICENSE.
- `reimplement_canny.py` is under `No License` components due to the absence of LICENSE in [scatter-animation-for-ikun](https://github.com/GBL-123/scatter-animation-for-ikun).