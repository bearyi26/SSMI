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

# def create_colored_masks(image, results):
#     """
#     Overlays multiple masks on the image. Each mask will be assigned a different color.
#     """
#     # Define colors for each mask (you can randomize or set specific colors)
#     color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#     colored_mask = np.zeros_like(image.squeeze(0).permute(1,2,0).cpu().numpy(), dtype=np.uint8)

#     for i in range(len(results)):
#         # Create a color mask from the binary mask
#         mask = results[i].masks.data.cpu().numpy()[0]
#         colored_mask[mask == 1] = color[i]  # Apply the color to the mask

#     return colored_mask

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

# import rospy
# from sensor_msgs.msg import Image
# import torch
# from cv_bridge import CvBridge

# from ultralytics import FastSAM
# import numpy as np
# import time

# # Create a FastSAM model
# model = FastSAM("FastSAM-x.pt")  # or FastSAM-x.pt
# br = CvBridge()
# rospy.init_node("segmentation")
# time.sleep(1)
# pub = rospy.Publisher('/semantic/colored_map', Image,queue_size=10)

# def create_colored_masks(image, results):
#     """
#     Overlays multiple masks on the image. Each mask will be assigned a different color.
#     """
#     # Define colors for each mask (you can randomize or set specific colors)
#     color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#     colored_mask = np.zeros_like(image.squeeze(0).permute(1,2,0).cpu().numpy(), dtype=np.uint8)

#     for i in range(len(results)):
#         # Create a color mask from the binary mask
#         mask = results[i].masks.data.cpu().numpy()[0]
#         colored_mask[mask == 1] = color[i]  # Apply the color to the mask

#     return colored_mask

# def callback(msg):
#     image = br.imgmsg_to_cv2(msg, "bgr8")
#     image = torch.from_numpy(image).to("cuda").type(torch.float32)
#     image = torch.permute(image, (2,0,1)).unsqueeze(0)
#     results = model(image, texts=["door"])
#     # masks = results[0].masks.data.cpu().numpy()
#     # rospy.loginfo("Mask Shape: %s", str(masks.shape))
#     colored_masks = create_colored_masks(image, results)
#     msg_out = br.cv2_to_imgmsg(colored_masks, "bgr8")
#     msg_out.header = msg.header
#     pub.publish(msg_out)


# rospy.Subscriber("/camera/color/image_raw", Image, callback)

# while True:
#     rospy.spin()

# from ultralytics import YOLO
# import rospy
# import time
# import ros_numpy
# from sensor_msgs.msg import Image

# segmentation_model = YOLO("yolov8s-seg.pt")
# rospy.init_node("segmentation")
# time.sleep(1)

# seg_image_pub = rospy.Publisher("/semantic/colored_map", Image, queue_size=5)


# def callback(data):
#     """Callback function to process image and publish annotated images."""
#     array = ros_numpy.numpify(data)

#     seg_result = segmentation_model(array)
#     seg_annotated = seg_result[0].plot(show=False)
#     seg_image_msg = ros_numpy.msgify(Image, seg_annotated, encoding="rgb8")
#     seg_image_msg.header.stamp = data.header.stamp
#     seg_image_pub.publish(seg_image_msg)


# rospy.Subscriber("/camera/color/image_raw", Image, callback)

# while True:
#     rospy.spin()