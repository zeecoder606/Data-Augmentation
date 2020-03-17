# Data Augmentation for Person Detection

 
## Installation

Clone this repository. Download the pre-trained Keras model for Human-Pose Detection(Open-pose) from [Open-pose](https://drive.google.com/file/d/13C2psaHPj0ooxyVUK85ub3EVpHkVtr4E/view?usp=sharing). Download the pre-trained VGG-19 PyTorch model from [VGG](https://drive.google.com/file/d/14A6RevoScEBoJtPfWAuhloaGXZoy82Sf/view?usp=sharing).

Put the VGG model in the main Data-Augmentation repository.
Put the Human-Pose Detection(Open-pose) model in the Data-Augmentation/tool.

# Requirement
    python 2
    pytorch(0.3.1)
    torchvision(0.2.0)
    numpy
    opencv
    scipy
    scikit-image
    pillow
    pandas
    tqdm
    dominate

# Testing
 place all the input images to transform in /seed_data/test_in
 
 Put all the target images, to which the input image is transformed in seed-data/test_out. 

Run the following python commands to generate 18-channel keypoint Posemaps of the input and target images. 

```python
python tool/compute_coordinates.py test_in
python tool/compute_coordinates.py test_out

```
Input Posemaps are stored in seed_data/test_inK. Target Posemaps are stored in seed_data/test_outK

The model will take 3 inputs, 1) Input image from test_in, 2) Input pose from test_inK and, 3) Output Pose from test_outK.
Every input person image from test_in will be transformed to a set of person images in all the target poses.  

```python
python test.py --dataroot ./seed_data/ --name final_honeywell --model PATN --phase test --dataset_mode keypoint --norm batch --batchSize 1 --resize_or_crop no --gpu_ids 0 --BP_input_nc 18 --no_flip --which_model_netG PATN --checkpoints_dir ./checkpoints --which_epoch latest --results_dir ./results --display_id 0

```
