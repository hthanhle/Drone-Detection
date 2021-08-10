# Drone Detection
## Descriptions
This project develops a drone detection system consisting of two stages: 

1. Drone object detector using YOLOv3 [[1]](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
2. Payload classification using VGG16 [[2]] (https://www.robots.ox.ac.uk/~vgg/publications/2015/Simonyan15/simonyan15.pdf), and Inception-v3 [[3]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)

## Quick start
### Installation
1. Install PyTorch=1.7.0 following [the official instructions](https://pytorch.org/)
2. git clone https://github.com/hthanhle/Age-Prediction
3. Install dependencies: `pip install -r requirements.txt`

### Test
Please run the following commands: 

1. Test on a single image: `python get_age.py --input test.jpg --output test_out.jpg --detector retinaface --estimator ssrnet`
2. Test on camera: 

**Example 1:** `python get_age_cam_coral.py`

**Example 2:** `python get_age_cam_basic.py`

4. Test on a single video: 
 
**Example 1:** `python get_age_video_ssrnet.py --input test.mp4 --output test_out.mp4`

**Example 2:** `python get_age_video_coral.py --input test.mp4 --output test_out.mp4`
