# README.md

This project provides the code and results for 'MAGNet: Multi-scale Awareness and Global Fusion Network for RGB-D Salient Object Detection'<br>

# Environments

```bash
conda create -n magnet python=3.9.18
conda activate magnet
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -c conda-forge opencv-python==4.7.0
pip install timm==0.6.5
conda install -c conda-forge tqdm
conda install yacs
```

# Data Preparation

- Download the RGB-D raw data from [baidu](https://pan.baidu.com/s/1T5RjFeoxWJZNJUuspn5YuA?pwd=7ttk) / [Google drive](https://drive.google.com/file/d/1MHCI_8UI_A1qKIlagl2Z9hC9iXUa1ZwC/view?usp=sharing) <br>
- Download the RGB-T raw data from [baidu](https://pan.baidu.com/s/1eexJSI4a2EGoaYcDkt1B9Q?pwd=i7a2) / [Google drive](https://drive.google.com/file/d/1hLhn5WV6xh-Q41upXF-bzyVpbszF9hUc/view?usp=sharing) <br>

Note that the depth maps of the raw data above are foreground is white.

# Training & Testing

- Train the MAGNet:
    1. download the pretrained SMT pth from [baidu](https://pan.baidu.com/s/11bNtCS7HyjnB7Lf3RIbpFg?pwd=bxiw) / [Google drive](https://drive.google.com/file/d/1eNhQwUHmfjR-vVGY38D_dFYUOqD_pw-H/view?usp=sharing), and put it under  `ckps/smt/`.
    2. modify the `rgb_root` `depth_root` `gt_root` in `train_Net.py` according to your own data path.
    3. run `python train_Net.py`
- Test the MAGNet:
    1. modify the `test_path` `pth_path` in `test_Net.py` according to your own data path.
    2. run `python test_Net.py`

# Evaluate tools

- You can select one of toolboxes to get the metrics
[CODToolbox](https://github.com/DengPingFan/CODToolbox) / [SOD_Evaluation_Metrics](https://github.com/zyjwuyan/SOD_Evaluation_Metrics)

# Saliency Maps

We provide the saliency maps of DUT, LFSD, NJU2K, NLPR, SIP, SSD, STERE datasets.

- RGB-D [baidu](https://pan.baidu.com/s/1FK8jcDb61QdFvZF1qKMV6g?pwd=c3a6) / [Google drive](https://drive.google.com/file/d/1uoeNZPzsj2RAr0ofM8fPD6N0JJ8HCyn9/view?usp=sharing)<br>

We provide the saliency maps of VT821, VT1000, VT5000 datasets.

- RGB-T [baidu](https://pan.baidu.com/s/1IQIkZS9GzmBT0PHflHqMNw?pwd=ebuw) / [Google drive](https://drive.google.com/file/d/198k-3R-yDF_y0Br7MoeSBP5XQOPuXPnL/view?usp=sharing)<br>

# Trained Models

- RGB-D [baidu](https://pan.baidu.com/s/1RPMA5Z3liMoUlG0AWuGeRA?pwd=5aqf) / [Google drive](https://drive.google.com/file/d/1vb2Vcbz9bCjvaSwoRZjIi39ae5Ei1GVs/view?usp=sharing) <br>

# Acknowledgement

The implement of this project is based on the codebases bellow. <br>

- [SeaNet](https://github.com/MathLee/SeaNet) <br>
- [LSNet](https://github.com/zyrant/LSNet) <br>
- Fps/speed test [MobileSal](https://github.com/yuhuan-wu/MobileSal/blob/master/speed_test.py)
- Evaluate tools [CODToolbox](https://github.com/DengPingFan/CODToolbox) / [SOD_Evaluation_Metrics](https://github.com/zyjwuyan/SOD_Evaluation_Metrics)<br>

# Contact

Feel free to contact me if you have any questions: (mingyu6346 at 163 dot com)
