# pl_inspection

pl_inspection is a package for power lines allignment and following missions using UAVs. It uses YOLOv8 to detect the power lines to be followed and a hybrid Type-2-fuzzy-PID controller to keep the drone over the cables throoughtout the mission. The methodology was tested using the [Parrot Bebop-2](https://www.parrot.com/assets/s3fs-public/2021-09/bebop-2_user-guide_uk_2.pdf) drone and can be adapted to other models using the ROS framework.

# Installation guide

First, install the bebop_autonomy and Parrot_arsdk to control the Bebop-2 UAV using ROS-noetic. Follow the [guide](https://github.com/antonellabarisic/parrot_arsdk/blob/noetic_dev/README.md) 

Install the additional packages :

```
pip install -U scikit-learn
pip3 install --upgrade pyit2fls

```

# PLDataset

<img src=https://github.com/guiaugustoga987/pl_inspection/assets/56890056/32118c5e-61c9-4b3f-a551-cde68936700c width=50% height=50%>



To use the PLdataset with a different object detection algorithm download it with the [link](https://universe.roboflow.com/pltdataset-cpx3u/power-lines-dataset) and select a proper annotation format. Some of the images used in this dataset was taken from the [TTPLA](https://github.com/R3ab/ttpla_dataset) and [PLD-UAV](https://github.com/SnorkerHeng/PLD-UAV) datasets.


# YOLOV8 training on PLdataset

To train the YOLOv8 on the Powerline dataset upload the [PLDataset_YOLOv8.ipynb](https://github.com/guiaugustoga987/pl_inspection/blob/main/training/PLDataset_YOLOv8.ipynb) in your google colab and follow the instructions in the notebook. 
