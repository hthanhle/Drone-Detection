# Drone Detection
## Descriptions
This project develops a drone detection system consisting of two stages: 
1. Drone object detector using YOLOv3 [[1]](https://pjreddie.com/media/files/papers/YOLOv3.pdf).
2. Payload classification using VGG16 [[2]](https://www.robots.ox.ac.uk/~vgg/publications/2015/Simonyan15/simonyan15.pdf), and Inception-v3 [[3]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf).

[![Watch the video](/output/test.jpg)](https://www.youtube.com/watch?v=tX19s4a37RI)


## Quick start
### Installation
1. Install Tensorflow=1.13.1 and Keras=2.2.4 following [the official instructions](https://pytorch.org/)
2. git clone https://github.com/hthanhle/Drone-Detection
3. Install dependencies: `pip install -r requirements.txt`

### Test
Please run the following commands: 

1. Test the full pipeline: `python test_detector_with_payload.py`
2. Test the drone detector: `python test_detector.py`
3. Test the drone detector: `python test_payload.py`
