import rospy
import sys
config_path = rospy.get_param("config_path", "")
sys.path.append(config_path + "/../../../..")
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge
import time

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from MaskFormer.mask_former import add_mask_former_config

rospy.init_node("segmentation")

# Create a MaskFormer model
cfg = get_cfg()
add_deeplab_config(cfg)
add_mask_former_config(cfg)
cfg.merge_from_file(config_path + "/maskformer_swin_small_bs16_160k.yaml")
model = DefaultPredictor(cfg)
br = CvBridge()
time.sleep(1)
pub = rospy.Publisher('/semantic/colored_map', Image,queue_size=10)

def callback(msg):
    image = br.imgmsg_to_cv2(msg, "bgr8")
    outputs = model(image)
    empty = np.zeros_like(image)
    # v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    v = Visualizer(empty, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    v = v.draw_sem_seg(outputs['sem_seg'].argmax(dim=0).to('cpu'))  # Use sem_seg for semantic segmentation
    msg_out = br.cv2_to_imgmsg(v.get_image()[:, :, ::-1], "bgr8")
    msg_out.header = msg.header
    pub.publish(msg_out)


rospy.Subscriber("/camera/color/image_raw", Image, callback)

while True:
    rospy.spin()