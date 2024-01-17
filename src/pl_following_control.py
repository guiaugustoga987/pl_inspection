#!/usr/bin/env python3

import rospy
import cv2
import torch
from ultralytics import YOLO
from cv_bridge import CvBridge,CvBridgeError
from rostopic import get_topic_type
from sensor_msgs.msg import Image,CompressedImage
import numpy as np
import cv_bridge
import matplotlib.pyplot as plt
from tello_driver.msg import TelloStatus
from std_msgs.msg import Empty
from ultralytics.yolo.utils.ops import scale_image
import random
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import subprocess
import time
import argparse
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.pipeline import Pipeline
from pyit2fls import IT2FLS, \
                     min_t_norm, product_t_norm, max_s_norm,IT2FS
from pyit2fls import trapezoid_mf, tri_mf,IT2Mamdani,T1FS,T1Mamdani
from tf.transformations import euler_from_quaternion
import os
import roslib; 
import pygame
import rospy
import sys, select, termios, tty
from std_msgs.msg import Bool
from bebop_msgs.msg import CommonCommonStateWifiSignalChanged
from bebop_msgs.msg import CommonCommonStateBatteryStateChanged
from bebop_msgs.msg import Ardrone3PilotingStateAirSpeedChanged
from bebop_msgs.msg import Ardrone3PilotingStateAltitudeChanged
from bebop_msgs.msg import Ardrone3PilotingStateSpeedChanged

class Yolov8Detector:

    def __init__(self):

        self.wifi_sub = rospy.Subscriber('/bebop/states/common/CommonState/WifiSignalChanged',CommonCommonStateWifiSignalChanged, self.wifi,queue_size=1)  
        self.battery_sub = rospy.Subscriber('/bebop/states/common/CommonState/BatteryStateChanged',CommonCommonStateBatteryStateChanged, self.battery,queue_size=1)    
        #self.air_speed_sub = rospy.Subscriber('/bebop/states/common/CommonState/WifiSignalChanged',CommonCommonStateWifiSignalChanged, self.wifi,queue_size=1)
        self.altitude_sub = rospy.Subscriber('bebop/states/ardrone3/PilotingState/AltitudeChanged',Ardrone3PilotingStateAltitudeChanged, self.altitude_callback,queue_size=1)
        self.speed_sub = rospy.Subscriber('/bebop/states/ardrone3/PilotingState/SpeedChanged',Ardrone3PilotingStateSpeedChanged, self.speed_callback,queue_size=1)
        

        self.bridge = cv_bridge.CvBridge()
        #self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/bebop/image_raw',Image, self.callback,queue_size=1)



        home_folder = os.getenv('HOME')
        self.model = YOLO(home_folder+"/catkin_ws/src/pl_inspection/weight/yolov8n_custom.pt")

        self.test = 0
        self.type = "/ze"
        self.number="/test"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self.video  = cv2.VideoWriter(home_folder+"/catkin_ws/src/pl_inspection/results"+self.type+self.number+"/clean.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 10, (856,480))
        self.video1  = cv2.VideoWriter(home_folder+"/catkin_ws/src/pl_inspection/results"+self.type+self.number+"/test.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 10, (856,480))

        self.pub = rospy.Publisher('image', Image, queue_size=10)
        self.sub_odom = rospy.Subscriber('bebop/odom', Odometry,self.odom_callback, queue_size=10)
        self.teste = rospy.Publisher('/manual_control', Bool, queue_size = 1)
        self.vel_pub = rospy.Publisher('/bebop/cmd_vel', Twist, queue_size = 1)
        #self.vel_pub = rospy.Publisher('/vservo/cmd_vel', Twist, queue_size = 1)
        self.camera_pub = rospy.Publisher('/bebop/camera_control', Twist, queue_size = 1)

        self.dt = 0.1
        self.last_ze_filtrado = 0
        self.derivative_ze_absolute = 0
        self.integral_ze = 0
        self.last_erro_ze = 0

        self.last_erro_phir = 0
        self.derivative_phir_absolute = 0
        self.integral_phir = 0

        self.no_frame = 0
        self.STOP_UAV = False
        self.start_control = False
        self.cont = 0

        self.integral_alt = 0
        self.last_erro_alt = 0
        self.derivative_alt = 0
        self.erro_alt_acc = 0
        self.last_alt_filtrado = 0


    def speed_callback(self,data):
        self.forward_speed = data.speedX
        self.lateral_speed = data.speedY


    def altitude_callback(self,data):
        self.altitude = data.altitude

    def battery(self,data):
        self.battery_percentage = data.percent


    def wifi(self,data):
        self.signal = data.rssi

    def odom_callback(self,data):
        
        quaternion = (
        data.pose.pose.orientation.x,
        data.pose.pose.orientation.y,
        data.pose.pose.orientation.z,
        data.pose.pose.orientation.w
        )
        euler = euler_from_quaternion(quaternion)

        self.roll = euler[0]
        self.pitch = euler[1]
        self.yaw = euler[2]  
        

    def callback(self, data):
        
        
        #np_arr = np.fromstring(data.data, np.uint8)
        #im = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        #print(im)
        #global im
        try:
            self.im = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
            #im = self.bridge.compressed_imgmsg_to_cv2(data)
            #img = cv2.imread('/home/ga/Documents/5.jpg')
        except CvBridgeError as e:
            print(e)
        


def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    
    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    # color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


def trajectory(imagebw,image_original):
    height,width = imagebw.shape
    #print(height,width)
    image_bw = cv2.rotate(imagebw, cv2.ROTATE_90_CLOCKWISE)
    image_bw = cv2.flip(image_bw,1)

    #linha_bw = cv2.cvtColor(image_bw, cv2.COLOR_BGR2GRAY)
    linha_bw = image_bw

    #print('c', linha_bw.shape)
    thresh = cv2.threshold(linha_bw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    per = []

    coords = np.column_stack(np.where(thresh > 0))

    gy,gx = np.array_split(coords,[-1],1)


    model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                    ('linear', LinearRegression(fit_intercept=False ))])

    '''
    ransac = RANSACRegressor(model, 
                            stop_probability=0.99,
                            max_trials=10,
                            min_samples=5,  
                            residual_threshold=250, 
                            random_state=42
    )
    ransac.fit(gx, gy)

    line_X = np.arange(0, height, 1)

    line_y_ransac = ransac.predict(line_X[:, np.newaxis])
    

    line_X1 = line_X.reshape(-1, 1)
    
    model.fit(line_X1,line_y_ransac)
    '''
    model.fit(gx,gy)
    coeficientes = model.named_steps['linear'].coef_

    a = coeficientes[0][2]
    b = coeficientes[0][1] 
    c = coeficientes[0][0]

    eixo_x = []
    eixo_y = []
    eixo_y = np.arange(0, height, 1)

    for i in range(height) :
        eixo_x.append((a*(i**2) +b*i + c)) 
        #eixo_x.append((b*i + c)) 


    y_pto_1 = (10)
    x_pto_1 = eixo_x[10]

    y_pto_2 = (100)
    x_pto_2 = eixo_x[100]

    y_pto_3 = (150)
    x_pto_3 = eixo_x[150]

    y_pto_4 = (220)
    x_pto_4 = eixo_x[220]

    y_pto_5 = (239)
    x_pto_5 = eixo_x[239]

    #y_pto_6 = (260)
    #x_pto_6 = eixo_x[260] 

    #y_pto_7 = (330)
    #x_pto_7 = eixo_x[330]

    #print(output)
    
    #cv2.circle(image_original, (int(x_pto_6),int(y_pto_6)), radius=3, color=(0, 0, 255), thickness=-1)
    #cv2.circle(image_original, (int(x_pto_5),int(y_pto_5)), radius=3, color=(0, 0, 255), thickness=-1)
    #cv2.circle(image_original, (int(x_pto_7),int(y_pto_7)), radius=3, color=(0, 0, 255), thickness=-1)
    #cv2.circle(image_original, (int(x_pto_1),int(y_pto_1)), radius=3, color=(0, 0, 255), thickness=-1)
    #cv2.circle(image_original, (int(x_pto_2),int(y_pto_2)), radius=3, color=(0, 0, 255), thickness=-1)
    #cv2.circle(image_original, (int(x_pto_3),int(y_pto_3)), radius=3, color=(0, 0, 255), thickness=-1)
    #cv2.circle(image_original, (int(x_pto_4),int(y_pto_4)), radius=3, color=(0, 0, 255), thickness=-1)
    x_pto_180 = eixo_x[180] #240
    y_pto_180 = 180
    #cv2.circle(image_original, (int(x_pto_180),int(y_pto_180)), radius=3, color=(0, 0, 255), thickness=-1)
    
    return image_original,eixo_x



def calc_erro(eixo_x,width,height,image_combined,altitude):

    erro_flag = True
    
    centro = 140
    p1 = centro-20 #220
    p2 = centro+20
    p3 = centro

    x_pto_180 = eixo_x[p3] #240
    y_pto_180 = p3
    x_pto_1 = eixo_x[p1]
    x_pto_2 = eixo_x[p2]

    #f = 692.8183639889837
    fx = 929.562627*0.5

    ang = 60 # Camera tilting related to y axis.
    ang = (ang*np.pi)/180

    #alt = 1.7
    x = altitude/np.cos(ang)
    x1 = (width*x)/fx
    conv1 = (1*x1)/width
    #print(width)

    erro = x_pto_180 - width/2
    erro_ze = erro*(conv1)

    delta_y = p2-p1
    delta_x = x_pto_2-x_pto_1

    erro_phir = (np.pi/2 - np.arctan2(delta_y,delta_x))


    h = 'Erro Lateral: {:.2f} m'.format(erro_ze)
    i = 'Erro Angular: {:.2f} rad'.format(erro_phir)

    cv2.circle(image_combined, (int(2*x_pto_180),int(2*y_pto_180)), radius=3, color=(0, 0, 255), thickness=-1)


    return erro_ze,image_combined,erro_phir,erro_flag,h,i


def fuzzySystem(algorithm, algorithm_params=[]):
    it2fls = IT2Mamdani(min_t_norm, max_s_norm, method="Centroid",algorithm=algorithm, algorithm_params=algorithm_params)


    #ku = 1 Tu= 5 bom resultado / 1 e 4 bom também - maior overshoot /

    KU_ze = 1 #0.4 #0.6 #0.65
    TU_ze = 5 # 4 #4.2   # 5 com 1/7

    kpmin_ = 0.32*KU_ze
    kpmax_ = 0.6*KU_ze
    kdmin_ = 0.08*KU_ze*TU_ze
    kdmax_ = 0.15*KU_ze*TU_ze

    kpmin = 0.32*KU_ze
    kpmax = 0.6*KU_ze
    kdmin = 0.08*KU_ze*TU_ze
    kdmax = 0.15*KU_ze*TU_ze

    #kpmax = kpmin_ + 1/7*kpmax_ 
    #kpmin = kpmin_ -1/7*kpmin_
    #kdmax = kdmin_ + 1/7*kdmax_
    #kdmin = kdmin_ -1/7*kdmin_
    
    kpmed = (kpmax+kpmin)/2
    kpmedtri = (kpmax-kpmin)/4
    kpmedinf = kpmed-kpmedtri
    kpmedsup = kpmed+kpmedtri

    kdmed = (kdmax+kdmin)/2
    kdmedtri = (kdmax-kdmin)/4
    kdmedinf = kdmed-kdmedtri
    kdmedsup = kdmed+kdmedtri

    domain1 = np.arange(0, 1.95, 0.01)
    domain2 = np.arange(-2, 2, 0.01)
    domain3 = np.arange(kpmin, kpmax, 0.01)
    domain4 = np.arange(kdmin, kdmax, 0.01)
    domain5 = np.arange(2, 4, 0.01)



    mf_inf = 1
    mf_inf_out = 1

    M = IT2FS(domain1,
                trapezoid_mf, [0.10, 0.15, 0.20, 0.25, 1.],
                trapezoid_mf, [0.12, 0.16, 0.19, 0.23,mf_inf])
    S = IT2FS(domain1,
                trapezoid_mf, [-0.01, 0.0, 0.10, 0.15, 1.],
                trapezoid_mf, [-0.01, 0.0, 0.07, 0.12,mf_inf])
    B = IT2FS(domain1,
                trapezoid_mf, [0.20, 0.25, 1.95, 1.96, 1.],
                trapezoid_mf, [0.23, 0.28, 1.95, 1.96,mf_inf])

    #IT2FS_plot(S,M,B)

    N = IT2FS(domain2,
                trapezoid_mf, [-3.1, -3, -0.8, 0, 1.],
                trapezoid_mf, [-3.1, -3, -1.2, -0.0,mf_inf])
    P = IT2FS(domain2,
                trapezoid_mf, [0, 0.8, 3, 3.1, 1.],
                trapezoid_mf, [0.0, 1.2, 3, 3.1,mf_inf])
    Z = IT2FS(domain2,
                tri_mf, [-1.2, 0, 1.2, 1, ],
                tri_mf, [-0.8, 0, 0.8,mf_inf])
    #IT2FS_plot(N,P,Z)

    S_P = IT2FS(domain3,
                trapezoid_mf, [kpmin-0.01,kpmin,kpmedinf,kpmed, 1.],
                trapezoid_mf, [kpmin-0.01,kpmin,0.98*kpmedinf,kpmed,mf_inf_out])
    B_P = IT2FS(domain3,
                trapezoid_mf, [kpmed,kpmedsup,kpmax,kpmax+0.01, 1.],
                trapezoid_mf, [kpmed,1.02*kpmedsup,kpmax,kpmax+0.01,mf_inf_out])
    M_P = IT2FS(domain3,
                tri_mf, [kpmedinf,kpmed,kpmedsup, 1, ],
                tri_mf, [1.02*kpmedinf,kpmed,0.98*kpmedsup,mf_inf_out])
    #IT2FS_plot(S_P,M_P,B_P)


    S_D = IT2FS(domain4,
                trapezoid_mf, [kdmin-0.01,kdmin,kdmedinf,kdmed, 1.],
                trapezoid_mf, [kdmin-0.01,kdmin,0.98*kdmedinf,kdmed,mf_inf_out])
    B_D = IT2FS(domain4,
                trapezoid_mf, [kdmed,kdmedsup,kdmax,kdmax+0.01, 1.],
                trapezoid_mf, [kdmed,1.02*kdmedsup,kdmax,kdmax+0.01,mf_inf_out])
    M_D = IT2FS(domain4,
                tri_mf, [kdmedinf,kdmed,kdmedsup, 1, ],
                tri_mf, [1.02*kdmedinf,kdmed,0.98*kdmedsup,mf_inf_out])
    #IT2FS_plot(S_D,M_D,B_D)

    S_I = IT2FS(domain5,
                trapezoid_mf, [1.9, 2, 2.5, 3, 1.],
                trapezoid_mf, [1.9, 2, 0.98*2.5, 3,mf_inf_out])
    B_I = IT2FS(domain5,
                trapezoid_mf, [3, 3.5, 4, 4.1, 1.],
                trapezoid_mf, [3, 1.02*3.5, 4, 4.1,mf_inf_out])
    M_I = IT2FS(domain5,
                tri_mf, [2.5, 3, 3.5, 1, ],
                tri_mf, [1.02*2.5, 3, 0.98*3.5,mf_inf_out])
    #IT2FS_plot(S_I,M_I,B_I)

    it2fls.add_input_variable("ze")  # E
    it2fls.add_input_variable("dot_ze")  # dot E
    it2fls.add_output_variable("P")
    it2fls.add_output_variable("D")
    it2fls.add_output_variable("alpha")
    '''
    #Original Approach
    it2fls.add_rule([("ze", S), ("dot_ze", N)], [("P", S_P),("D", B_D),("alpha", B_I)])
    it2fls.add_rule([("ze", M), ("dot_ze", N)], [("P", M_P),("D", M_D),("alpha", M_I)])
    it2fls.add_rule([("ze", B), ("dot_ze", N)], [("P", B_P),("D", S_D),("alpha", S_I)])

    it2fls.add_rule([("ze", S), ("dot_ze", Z)], [("P", B_P),("D", B_D),("alpha", M_I)])
    it2fls.add_rule([("ze", M), ("dot_ze", Z)], [("P", B_P),("D", M_D),("alpha", S_I)])
    it2fls.add_rule([("ze", B), ("dot_ze", Z)], [("P", B_P),("D", S_D),("alpha", S_I)])

    it2fls.add_rule([("ze", S), ("dot_ze", P)], [("P", S_P),("D", B_D),("alpha", B_I)])
    it2fls.add_rule([("ze", M), ("dot_ze", P)], [("P", M_P),("D", M_D),("alpha", M_I)])
    it2fls.add_rule([("ze", B), ("dot_ze", P)], [("P", B_P),("D", S_D),("alpha", S_I)])
    
    '''
    # Proposed Approach
    it2fls.add_rule([("ze", S), ("dot_ze", N)], [("P", S_P),("D", M_D),("alpha", B_I)]) 
    it2fls.add_rule([("ze", M), ("dot_ze", N)], [("P", M_P),("D", M_D),("alpha", M_I)]) 
    it2fls.add_rule([("ze", B), ("dot_ze", N)], [("P", B_P),("D", S_D),("alpha", S_I)])

    it2fls.add_rule([("ze", S), ("dot_ze", Z)], [("P", B_P),("D", B_D),("alpha", B_I)])
    it2fls.add_rule([("ze", M), ("dot_ze", Z)], [("P", M_P),("D", M_D),("alpha", S_I)])
    it2fls.add_rule([("ze", B), ("dot_ze", Z)], [("P", B_P),("D", S_D),("alpha", S_I)])

    it2fls.add_rule([("ze", S), ("dot_ze", P)], [("P", B_P),("D", M_D),("alpha", B_I)])
    it2fls.add_rule([("ze", M), ("dot_ze", P)], [("P", B_P),("D", M_D),("alpha", M_I)])
    it2fls.add_rule([("ze", B), ("dot_ze", P)], [("P", B_P),("D", S_D),("alpha", S_I)])
    
    
    return it2fls

def fuzzySystem_phir(algorithm, algorithm_params=[]):
    it2fls = IT2Mamdani(min_t_norm, max_s_norm, method="Centroid",algorithm=algorithm, algorithm_params=algorithm_params)

    KU_phir = 1.5 #2 
    TU_phir = 2.1

    kpmin_ = 0.32*KU_phir
    kpmax_ = 0.6*KU_phir
    kdmin_ = 0.08*KU_phir*TU_phir
    kdmax_ = 0.15*KU_phir*TU_phir

    kpmax = kpmin_ + 1/7*kpmax_
    kpmin = kpmin_ -1/7*kpmin_
    kdmax = kdmin_ + 1/7*kdmax_
    kdmin = kdmin_ -1/7*kdmin_

    #kpmin = 0.32*KU_phir
    #kpmax = 0.6*KU_phir
    #kdmin = 0.08*KU_phir*TU_phir
    #kdmax = 0.15*KU_phir*TU_phir

    kpmed = (kpmax+kpmin)/2
    kpmedtri = (kpmax-kpmin)/4
    kpmedinf = kpmed-kpmedtri
    kpmedsup = kpmed+kpmedtri

    kdmed = (kdmax+kdmin)/2
    kdmedtri = (kdmax-kdmin)/4
    kdmedinf = kdmed-kdmedtri
    kdmedsup = kdmed+kdmedtri

    domain1 = np.arange(0, 1.57, 0.01)
    domain2 = np.arange(-3, 3, 0.01)
    domain3 = np.arange(kpmin, kpmax, 0.01)
    domain4 = np.arange(kdmin, kdmax, 0.01)
    domain5 = np.arange(2, 4, 0.01)
    mf_inf = 1

    M = IT2FS(domain1,
                trapezoid_mf, [0.17,0.29,0.44,0.58, 1.],
                trapezoid_mf, [0.19,0.31,0.42,0.56,mf_inf])
    S = IT2FS(domain1,
                trapezoid_mf, [-0.03,0.0,0.19,0.22, 1.],
                trapezoid_mf, [-0.03,0.0,0.17,0.20,mf_inf])
    B = IT2FS(domain1,
                trapezoid_mf, [0.29,0.41,1.57,1.58, 1.],
                trapezoid_mf, [0.31,0.43,1.57,1.58,mf_inf])

    #IT2FS_plot(S,M,B)

    N = IT2FS(domain2,
                trapezoid_mf, [-3.1, -3, -0.8, 0, 1.],
                trapezoid_mf, [-3.1, -3, -1.2, -0.0,mf_inf])
    P = IT2FS(domain2,
                trapezoid_mf, [0, 0.8, 3, 3.1, 1.],
                trapezoid_mf, [0.0, 1.2, 3, 3.1,mf_inf])
    Z = IT2FS(domain2,
                tri_mf, [-1.2, 0, 1.2, 1, ],
                tri_mf, [-0.8, 0, 0.8,mf_inf])
    #IT2FS_plot(N,P,Z)

    S_P = IT2FS(domain3,
                trapezoid_mf, [kpmin-0.01,kpmin,kpmedinf,kpmed, 1.],
                trapezoid_mf, [kpmin-0.01,kpmin,0.98*kpmedinf,kpmed,mf_inf])
    B_P = IT2FS(domain3,
                trapezoid_mf, [kpmed,kpmedsup,kpmax,kpmax+0.01, 1.],
                trapezoid_mf, [kpmed,1.02*kpmedsup,kpmax,kpmax+0.01,mf_inf])
    M_P = IT2FS(domain3,
                tri_mf, [kpmedinf,kpmed,kpmedsup, 1, ],
                tri_mf, [1.02*kpmedinf,kpmed,0.98*kpmedsup,mf_inf])
    #IT2FS_plot(S_P,M_P,B_P)


    S_D = IT2FS(domain4,
                trapezoid_mf, [kdmin-0.01,kdmin,kdmedinf,kdmed, 1.],
                trapezoid_mf, [kdmin-0.01,kdmin,0.98*kdmedinf,kdmed,mf_inf])
    B_D = IT2FS(domain4,
                trapezoid_mf, [kdmed,kdmedsup,kdmax,kdmax+0.01, 1.],
                trapezoid_mf, [kdmed,1.02*kdmedsup,kdmax,kdmax+0.01,mf_inf])
    M_D = IT2FS(domain4,
                tri_mf, [kdmedinf,kdmed,kdmedsup, 1, ],
                tri_mf, [1.02*kdmedinf,kdmed,0.98*kdmedsup,mf_inf])
    #IT2FS_plot(S_D,M_D,B_D)

    S_I = IT2FS(domain5,
                trapezoid_mf, [1.9, 2, 2.5, 3, 1.],
                trapezoid_mf, [1.9, 2, 0.98*2.5, 3,mf_inf])
    B_I = IT2FS(domain5,
                trapezoid_mf, [3, 3.5, 4, 4.1, 1.],
                trapezoid_mf, [3, 1.02*3.5, 4, 4.1,mf_inf])
    M_I = IT2FS(domain5,
                tri_mf, [2.5, 3, 3.5, 1, ],
                tri_mf, [1.02*2.5, 3, 0.98*3.5,mf_inf])
    #IT2FS_plot(S_I,B_I,M_I)

    it2fls.add_input_variable("phir")  
    it2fls.add_input_variable("dot_phir")
    it2fls.add_output_variable("P")
    it2fls.add_output_variable("D")
    it2fls.add_output_variable("alpha")


    '''
    it2fls.add_rule([("phir", S), ("dot_phir", N)], [("P", S_P),("D", B_D),("alpha", B_I)])
    it2fls.add_rule([("phir", M), ("dot_phir", N)], [("P", M_P),("D", M_D),("alpha", M_I)])
    it2fls.add_rule([("phir", B), ("dot_phir", N)], [("P", B_P),("D", S_D),("alpha", S_I)])

    it2fls.add_rule([("phir", S), ("dot_phir", Z)], [("P", B_P),("D", B_D),("alpha", M_I)])
    it2fls.add_rule([("phir", M), ("dot_phir", Z)], [("P", B_P),("D", M_D),("alpha", S_I)])
    it2fls.add_rule([("phir", B), ("dot_phir", Z)], [("P", B_P),("D", S_D),("alpha", S_I)])

    it2fls.add_rule([("phir", S), ("dot_phir", P)], [("P", S_P),("D", B_D),("alpha", B_I)])
    it2fls.add_rule([("phir", M), ("dot_phir", P)], [("P", B_P),("D", M_D),("alpha", M_I)])
    it2fls.add_rule([("phir", B), ("dot_phir", P)], [("P", B_P),("D", S_D),("alpha", S_I)])
    '''
    it2fls.add_rule([("phir", S), ("dot_phir", N)], [("P", S_P),("D", M_D),("alpha", B_I)])
    it2fls.add_rule([("phir", M), ("dot_phir", N)], [("P", M_P),("D", M_D),("alpha", M_I)])
    it2fls.add_rule([("phir", B), ("dot_phir", N)], [("P", B_P),("D", S_D),("alpha", S_I)])

    it2fls.add_rule([("phir", S), ("dot_phir", Z)], [("P", B_P),("D", B_D),("alpha", B_I)])
    it2fls.add_rule([("phir", M), ("dot_phir", Z)], [("P", M_P),("D", M_D),("alpha", S_I)])
    it2fls.add_rule([("phir", B), ("dot_phir", Z)], [("P", B_P),("D", S_D),("alpha", S_I)])

    it2fls.add_rule([("phir", S), ("dot_phir", P)], [("P", B_P),("D", M_D),("alpha", B_I)])
    it2fls.add_rule([("phir", M), ("dot_phir", P)], [("P", B_P),("D", M_D),("alpha", M_I)])
    it2fls.add_rule([("phir", B), ("dot_phir", P)], [("P", B_P),("D", S_D),("alpha", S_I)])
    
    return it2fls


def sat(x,value) :

    if x > value :
        x = value
    if x < -value :
        x = -value

    return x

def clean_shutdown():      

    twist = Twist()
    ic.video.release()
    ic.video1.release()
    twist.angular.z = 0
    twist.linear.z = 0
    twist.linear.x = 0
    twist.linear.y = 0
    manual = True
    ic.teste.publish(manual)
    ic.vel_pub.publish(twist)
    rospy.loginfo("System is shutting down. Stopping robot...")



def process():

    ic = Yolov8Detector()
    pygame.init()
    speed = 0.5
    turn = 0.5
    j = pygame.joystick.Joystick(0)
    j.init()

    clock = pygame.time.Clock()
    #textPrint = TextPrint()

    manual_control = True
    moveBindings = {
        'fwd':(0,1,0,0),
        'bk':(0,-1,0,0),
        'lft':(-1,0,0,0),
        'rgt':(1,0,0,0),
        'UP':(0,0,1,0),
        'DOWN':(0,0,-1,0),
        'CLOCKWISE':(0,0,0,1),
        'COUNTER_CLOCKWISE':(0,0,0,-1),
        }


    joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
    for joystick in joysticks:
        print(joystick.get_name())
    
    stop_control = False
    ic.no_frame = 0
    manual = True
    camera_adjustment = True
    while not rospy.is_shutdown():
        rate = rospy.Rate(10) 
        sat_velocity = 1.5
        #ic.altitude = 1
        start = time.time()
        
        x = 0
        y = 0
        z = 0
        th = 0


        if camera_adjustment == True :
            print('Ajuste de camera')
            cam_twist = Twist()
            cam_twist.angular.y = -60
            ic.camera_pub.publish(cam_twist)
            rospy.sleep(2)
            camera_adjustment = False
            print('Camera Ajustada')


        events = pygame.event.get()
        for event in events:

            RB = j.get_button(5)
            select = j.get_button(6)
            start = j.get_button(7)

        if select == 1 : # The mission starts upon pressing the SELECT button on joystick.
            ic.start_control = True
            print('Start Autonomous Controller')
            
            manual = False
            

        if start == 1 : # The autonomous controller node is canceled pressing START button on joystick.
            reason = 'START BUTTON PRESSED : Shutting down the control node '
            rospy.signal_shutdown(reason) 

        frame_quantity = 5 # Quantidade de frames sem detecção para mudar de modo (Manual ou Autônomo)

        #print(control)

        #print(ic.no_frame)
        #print(control)
        #control = True
        fat = 0.5
        frame = ic.im
        h, w, _ = frame.shape
        frame = cv2.resize(frame, (int(fat*w), int(fat*h)))

        if ic.cont >= frame_quantity and ic.start_control == True : # CASO AS LINHAS NÃO SEJAM ENCONTRADAS EM frame_quantity FRAMES CONSECUTIVOS DEIXAR O DRONE EM HOVER (PAIRANDO) 

            image_combined = ic.im
            manual = True
            print('sem detecção - Drone em hovering mode')
            twist = Twist()
            twist.angular.z = 0
            twist.linear.z = 0
            twist.linear.x = 0
            twist.linear.y = 0 
            ic.vel_pub.publish(twist) 


        else :
            
            erro_flag = False
            size = 320+3*32 # TEST WITH 640.

            print('roll',ic.roll)
            print('yaw',ic.yaw)

            #ic.video.write(ic.im)

            fat = 0.5
            frame = cv2.resize(frame, (int(fat*w), int(fat*h))) # REDIMENSIONAR PARA CÁLCULO DA TRAJETÓRIA
            #frame = cv2.resize(frame, (int(640), int(480)))
            h, w, _ = frame.shape
            #print(frame.shape)
            #frame = cv2.rotate(frame, cv2.ROTATE_180)
            #frame = cv2.flip(frame,1)

            frame_original = frame
            #frame_retificado = image_rectification(h,w,frame,ic.roll)

            alt_stpt = 10

            #relative_alt = 1
            relative_alt = ic.altitude - alt_stpt
            #result = ic.model(ic.im, verbose=False,imgsz = 448)[0]
            results = ic.model.predict(frame, verbose=False,imgsz = size,conf = 0.5) #COLOCAR A IMAGEM ORIGINAL
            colors = [[0,255,0]]

            for r in results:
                boxes = r.boxes  # Boxes object for bbox outputs
                masks = r.masks  # Masks object for segment masks outputs
                probs = r.probs  # Class probabilities for classification outputs


            if ic.start_control == True : # Caso o controle autônomo seja iniciado -> Aplicar o controle de altitude


                #print(ic.im.shape)    
                
                KU_alt = 2.0 #2
                TU_alt = 4   #4
                beta = 1  #0.6

                kp_alt = 0.6*KU_alt
                ki_alt = 1.2*(KU_alt/TU_alt)
                kd_alt = 0.075*(KU_alt*TU_alt)

                altitude_setpoint = 13
                #ic.altitude = 1
                erro_alt = altitude_setpoint - ic.altitude

                alt_filtrado = (beta*erro_alt) + (1-beta)*ic.last_alt_filtrado
                ic.derivative_alt = (alt_filtrado - ic.last_alt_filtrado)/ic.dt  
                ic.last_alt_filtrado = alt_filtrado
                #ic.derivative_alt = (erro_alt - ic.last_erro_alt)/ic.dt

                ic.integral_alt += erro_alt*ic.dt
                ic.integral_alt = sat(ic.integral_alt,0.25) #Anti windup

                P_term = kp_alt * erro_alt
                I_term = ki_alt * ic.integral_alt
                D_term = kd_alt * ic.derivative_alt

                ic.last_erro_alt = erro_alt

                vel_alt = P_term + I_term + D_term
                vel_alt = sat(vel_alt,1)
                #vel_alt = 0
                '''
                if ic.cont < 251 :
                    print(erro_alt)
                    ic.erro_alt_acc += np.abs(erro_alt)
                if ic.cont == 251 :
                    media = ic.erro_alt_acc/(ic.cont-1)
                    print('media',media)

                '''
                '''
                if ic.altitude < 11.0 : # COMENTAR SE NÃO FOR FAZER O ALINHAMENTO - O NÓ DE CONTROLE É DESLIGADO CASO O DRONE ATINJA UMA ALTITUDE MENOR QUE 11 METROS.
                    reason = 'Low altitude threshold : Shutting down the control node '
                    twist = Twist()
                    twist.angular.z = 0
                    twist.linear.z = 0
                    twist.linear.x = 0
                    twist.linear.y = 0
                    ic.vel_pub.publish(twist)
                    rospy.signal_shutdown(reason) 
                '''
                #ic.cont += 1


            if masks is not None : # Caso as linhas sejam detectadas.

                output_bw = (results[0].cpu().masks.data[0].numpy() * 255).astype("uint8")
                output_bw = cv2.resize(output_bw, (w, h))
                masks = masks.data.cpu()

                #masks = masks[0].data.cpu() # Só a máscara de maior probabilidade

                #frame_original,eixo_x = trajectory(output_bw,frame_original)
                
                try :
                    frame_original,eixo_x = trajectory(output_bw,frame_original)
                except :
                    twist = Twist()
                    twist.angular.z = 0
                    twist.linear.z = 0
                    twist.linear.x = 0
                    twist.linear.y = 0
                    ic.vel_pub.publish(twist)
                    print('Erro no cálculo da trajetória')
                    ic.no_frame += 1
                    continue 

                ic.no_frame = 0 # Zera a variável de não-detecção.
                
                # rescale masks to original image
                for seg, box in zip(masks.data.cpu().numpy(), boxes):
                    seg = cv2.resize(seg, (856, 480))
                    #cv2.imshow("previewa", seg)
                    #cv2.waitKey(1)
                    #print('a',ic.im.shape)
                    #print('b',seg.shape)
                    #seg = image_rectification(h,w,seg,-ic.roll)
                    image_combined = overlay(ic.im, seg, colors[int(box.cls)], 0.4)
                



                #ic.altitude = 1
                #erro_ze,image_combined,erro_phir,erro_flag,h,i = calc_erro(eixo_x,w,h,image_combined,ic.altitude)

                #print(erro_flag)
                erro_ze,image_combined,erro_phir,erro_flag,h,i = calc_erro(eixo_x,w,h,image_combined,relative_alt)

                if ic.start_control == True :
                    ze = np.abs(erro_ze)
                    

                    #print(erro_ze)
                    beta = 0.98 # Low Pass filter to derivative term. See : https://ptolemy.berkeley.edu/projects/chess/tbd/wiki/C-code/LowPassFilterForDerivativeControl era 0.98
                    #print('last',ic.last_erro_ze)
                    derivative_ze_absolute = (np.abs(erro_ze) - np.abs(ic.last_erro_ze))/ic.dt
                    
                    ze_filtrado = (beta*erro_ze) + (1-beta)*ic.last_ze_filtrado
                    derivative_ze = (ze_filtrado - ic.last_ze_filtrado)/ic.dt  
                    
                    
                    derivative_ze = sat(derivative_ze,3)
                    ic.derivative_ze_absolute = sat(ic.derivative_ze_absolute,3)
                    
                    dot_ze = ic.derivative_ze_absolute      
                    #To avoid errors on Fuzzy controller
                    ze = sat(ze,1.95)

                    ic.last_ze_filtrado = ze_filtrado
                    ic.last_erro_ze = erro_ze

                    it2fpid_KM = fuzzySystem("NT")
                    c,TR = it2fpid_KM.evaluate({"ze":ze, "dot_ze":dot_ze})
                
                    P_ = TR["P"]
                    D_ = TR["D"]
                    alpha_ = TR["alpha"]

                    #P = (P_[0] + P_[1]) / 2
                    #D = (D_[0] + D_[1]) / 2
                    #alpha = (alpha_[0] + alpha_[1]) / 2
                    
                    kp_ze = P_
                    kd_ze = D_
                    ki_ze = (kp_ze*kp_ze)/(1*alpha_*kd_ze)

                    #kp_ze = 0.4
                    #kd_ze = 0
                    #ki_ze = 0

                    ic.integral_ze += erro_ze*ic.dt
                    ic.integral_ze = sat(ic.integral_ze,0.25) #Anti windup

                    P_term = kp_ze * erro_ze
                    I_term = ki_ze * ic.integral_ze
                    D_term = kd_ze * derivative_ze
                    
                    vel_ze = P_term + I_term + D_term
                    vel_ze = sat(vel_ze,1)

                    vel_ze = vel_ze/1

                    ic.derivative_phir = (erro_phir - ic.last_erro_phir)/ic.dt
                    ic.derivative_phir_absolute = (np.abs(erro_phir) - np.abs(ic.last_erro_phir))/ic.dt

                    ic.derivative_phir = sat(ic.derivative_phir,3)
                    ic.derivative_phir_absolute = sat(ic.derivative_phir_absolute,3)

                    ic.last_erro_phir = erro_phir


                    phir = np.abs(erro_phir)
                    #print(erro_phir)
                    dot_phir = ic.derivative_phir_absolute

                    it2fpid_KM = fuzzySystem_phir("NT")
                    c, TR = it2fpid_KM.evaluate({"phir":phir, "dot_phir": dot_phir})
                
                    P_ = TR["P"]
                    D_ = TR["D"]
                    alpha_ = TR["alpha"]

                    #P = (P_[0] + P_[1]) / 2
                    #D = (D_[0] + D_[1]) / 2
                    #alpha = (alpha_[0] + alpha_[1]) / 2

                    kp_phir = P_
                    kd_phir = D_
                    ki_phir = (kp_phir*kp_phir)/(1*alpha_*kd_phir)

                    #kp_phir = 2
                    #kd_phir = 0
                    #ki_phir = 0

                    ic.integral_phir += erro_phir*ic.dt
                    ic.integral_phir = sat(ic.integral_phir,0.25) #Anti windup

                    P_term = kp_phir * erro_phir
                    I_term = ic.integral_phir*ki_phir
                    D_term = kd_phir * ic.derivative_phir

                    vel_phir = P_term + I_term + D_term
                    vel_phir = sat(vel_phir,2)

                    #vel_phir = -vel_phir
                    #vel_phir = 0

                    #vel_ze = -0.35
                    #vel_ze = 0*vel_ze
                    vel_ze = 0.5*vel_ze

                    #altitude_setpoint = 1.0 
                    #erro_alt = altitude_setpoint - ic.altitude

                    #vel_alt = 0.92*erro_alt
                    #vel_alt = sat(vel_alt,0.2)

                    #print(erro_alt)

                    twist = Twist()
                    twist.angular.z = 1*vel_phir
                    twist.linear.z = 0
                    twist.linear.x = 0 # COLOCAR UM VALOR PARA MISSÃO DE SEGUIMENTO
                    twist.linear.y = 1*-vel_ze
                    ic.vel_pub.publish(twist)  

                    #GRAVAR OS DADOS EM ARQUIVOS TXT
                    '''
                    if ic.cont < 250 :

                        home_folder = os.getenv('HOME')
                        with open(home_folder+"/catkin_ws/src/pl_inspection/results"+ic.type+ic.number+"/ze.txt", "a") as output:
                            output.write("%s \n" % erro_ze)           
                        with open(home_folder+"/catkin_ws/src/pl_inspection/results"+ic.type+ic.number+"/phir.txt", "a") as output:
                            output.write("%s \n" % erro_phir)  
                        with open(home_folder+"/catkin_ws/src/pl_inspection/results"+ic.type+ic.number+"/vel_ze.txt", "a") as output:
                            output.write("%s \n" % vel_ze)           
                        with open(home_folder+"/catkin_ws/src/pl_inspection/results"+ic.type+ic.number+"/vel_phir.txt", "a") as output:
                            output.write("%s \n" % vel_phir) 
                        with open(home_folder+"/catkin_ws/src/pl_inspection/results"+ic.type+ic.number+"/kp_ze.txt", "a") as output:
                            output.write("%s \n" % kp_ze) 
                        with open(home_folder+"/catkin_ws/src/pl_inspection/results"+ic.type+ic.number+"/ki_ze.txt", "a") as output:
                            output.write("%s \n" % ki_ze) 
                        with open(home_folder+"/catkin_ws/src/pl_inspection/results"+c.type+ic.number+"/kd_ze.txt", "a") as output:
                            output.write("%s \n" % kd_ze) 
                        with open(home_folder+"/catkin_ws/src/pl_inspection/results"+ic.type+ic.number+"/kp_phir.txt", "a") as output:
                            output.write("%s \n" % kp_phir) 
                        with open(home_folder+"/catkin_ws/src/pl_inspection/results"+ic.type+ic.number+"/ki_phir.txt", "a") as output:
                            output.write("%s \n" % ki_phir) 
                        with open(home_folder+"/catkin_ws/src/pl_inspection/results"+ic.type+ic.number+"/kd_phir.txt", "a") as output:
                            output.write("%s \n" % kd_phir) 
                    '''
                    if ic.cont == 250 :
                        print('chegou')
                    ic.cont += 1
                    #print(ic.altitude)

            else : # Caso a linha não seja detectada

                ic.no_frame += 1 #variável incremental para ativar o controle manual caso não haja detecção das linhas em N iterações seguidas
                #print('nao detectado')
                
                image_combined = ic.im
                vel_ze = 0
                vel_phir = 0
                if ic.start_control == True :

                    twist = Twist()
                    twist.angular.z = 0
                    twist.linear.z = 0*vel_alt
                    twist.linear.x = 0
                    twist.linear.y = 0
                    ic.vel_pub.publish(twist)

        #if ic.cont > 100 :
        #    print(deu)
        cam_twist = Twist()
        cam_twist.angular.y = -60
        ic.camera_pub.publish(cam_twist)
        #ic.altitude = 1
        #a = 'Bateria: {:.2f} %'.format(ic.battery_percentage)
        b = 'Sinal Wifi: {:.2f}'.format(ic.signal)
        c = 'Altitude: {:.2f} m'.format(ic.altitude)
        d = 'Vel. Avanco: {:.2f} m/s'.format(ic.forward_speed)
        e = 'Vel. Lateral: {:.2f} m/s'.format(ic.lateral_speed)
        f = 'Relative Altitude: {:.2f} m'.format(relative_alt) 

        #height, width, _ = image_combined.shape

        alpha = 0.4
        overlay1 = image_combined.copy()
        output = image_combined.copy()
        cv2.rectangle(overlay1, (20, 0), (240, 160),(255, 255, 255), -1)
        cv2.addWeighted(overlay1, alpha, output, 1 - alpha,0, output)

        image_combined = output

        if ic.signal >= -30 :
            cv2.putText(image_combined,b, (20, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if ic.signal > -70 and ic.signal < -30 :
            cv2.putText(image_combined,b, (20, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 1, cv2.LINE_AA)
        if ic.signal < -70 :
            cv2.putText(image_combined,b, (20, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

        if erro_flag == True :
            cv2.putText(image_combined,h, (20, 140),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(image_combined,i, (20, 160),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


        #cv2.putText(image_combined,a, (20, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 1, cv2.LINE_AA)
        cv2.putText(image_combined,c, (20, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 1, cv2.LINE_AA)
        cv2.putText(image_combined,d, (20, 80),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 1, cv2.LINE_AA)
        cv2.putText(image_combined,e, (20, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 1, cv2.LINE_AA)
        cv2.putText(image_combined,f, (20, 120),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 1, cv2.LINE_AA)


        ros_image = ic.bridge.cv2_to_imgmsg(image_combined, "bgr8")
        ros_image.header.stamp = rospy.Time.now()
        ic.pub.publish(ros_image)

        ic.video.write(ic.im)

        ic.video1.write(image_combined)

        cv2.imshow("preview", image_combined)
        cv2.waitKey(1)
        
        ic.teste.publish(manual)
        rate.sleep() 
        #ic.cont += 1  
        #print(asd)
        passed = (time.time() - start)
        #print(passed)


if __name__ == "__main__":

    
    rospy.init_node("yolov8", anonymous=True)
    rospy.on_shutdown(clean_shutdown)
    try:
        ic = Yolov8Detector()
        process()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    '''    
    except :
        twist = Twist()
        twist.angular.z = 0
        twist.linear.z = 0
        twist.linear.x = 0
        twist.linear.y = 0
        ic.vel_pub.publish(twist)
        manual = True
        ic.teste.publish(manual)
    '''



    
