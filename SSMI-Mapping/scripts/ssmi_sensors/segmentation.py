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
cfg.merge_from_file(config_path + "/maskformer_swin_large_IN21k_384_bs16_160k_res640.yaml")
model = DefaultPredictor(cfg)
br = CvBridge()
time.sleep(1)
pub = rospy.Publisher('/semantic/colored_map', Image, queue_size=10)

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

ADE20K_COLORMAP = np.asarray([
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
  ])

def apply_colormap(mask, colormap):
    """
    Converts a class ID mask into a colorized mask using the provided colormap.
    Args:
        mask (numpy.ndarray): 2D array of shape (H, W) with class IDs.
        colormap (list of tuples): List where index represents class ID and value is (R, G, B).
    Returns:
        numpy.ndarray: 3D array of shape (H, W, 3) with RGB values.
    """
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(colormap):
        colored_mask[mask == class_id] = color
    return colored_mask

def callback(msg):
    image = br.imgmsg_to_cv2(msg, "bgr8")
    outputs = model(image)
    colored_mask = apply_colormap(outputs['sem_seg'].argmax(dim=0).to('cpu'), ADE20K_COLORMAP)
    # empty = np.zeros_like(image)
    # # v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    # v = Visualizer(empty, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    # v = v.draw_sem_seg(outputs['sem_seg'].argmax(dim=0).to('cpu'))  # Use sem_seg for semantic segmentation
    # black_image = np.full((480, 640, 3), 0, dtype=np.uint8)
    # msg_out = br.cv2_to_imgmsg(black_image, "bgr8")
    # msg_out = br.cv2_to_imgmsg(v.get_image()[:, :, ::-1], "bgr8")
    msg_out = br.cv2_to_imgmsg(colored_mask[:, :, ::-1], "bgr8")
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