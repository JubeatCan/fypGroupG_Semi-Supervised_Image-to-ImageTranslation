import os
import sys
import time
import random
import math
import numpy as np
import skimage.io
from PIL import Image

class ShowProcess():

    i = 0 # 当前的处理进度
    max_steps = 0 # 总共需要处理的次数
    max_arrow = 50 #进度条的长度
    infoDone = 'done'

    def __init__(self, max_steps, infoDone = 'Done'):
        self.max_steps = max_steps
        self.i = 0
        self.infoDone = infoDone

    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps) #计算显示多少个'>'
        num_line = self.max_arrow - num_arrow #计算显示多少个'-'
        percent = self.i * 100.0 / self.max_steps #计算完成进度，格式为xx.xx%
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r' #带输出的字符串，'\r'表示不换行回到最左边
        sys.stdout.write(process_bar) #这两句打印字符到终端
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('')
        print(self.infoDone)
        self.i = 0
# Defined Root Path
ROOT_DIR = os.path.abspath('../')

sys.path.append(ROOT_DIR)
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

sys.path.append(os.path.join(ROOT_DIR, 'samples/coco/'))
import coco

MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory of images to make masks of
IMAGE_DIR = os.path.join(ROOT_DIR,'BITMAPMaskAO/')

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Load image from the images folder

#Modify your paths
folder = 'trainB'
resultFolder = 'trainB_m'

# main
import tensorflow as tf
from tensorflow.python.ops import gen_image_ops
tf.image.non_max_suppression = gen_image_ops.non_max_suppression_v2
from PIL import Image
file_names = os.walk(os.path.join(IMAGE_DIR,folder))
Names = next(os.walk(os.path.join(IMAGE_DIR,folder)))[2]
fileNames = []
results = []
for root, dirs, files in file_names:
    for name in files:
        fileNames.append(os.path.join(root,name))
print('Place:', IMAGE_DIR)
image = skimage.io.imread_collection(fileNames)
print('Count: ', len(image))
process_bar = ShowProcess(len(image), 'OK')
print('Detecting')
for i in image:
    results.append(model.detect([i], verbose=0)[0])
    process_bar.show_process()
    
process_bar = ShowProcess(len(results), 'OK')
print('Saving Masks')
for w in range(0, len(results)):
    y = image[w].shape[0]
    x = image[w].shape[1]
    mask = np.zeros((y,x),dtype=np.uint8)
    for i in range(0,y):
        for j in range(0,x):
            for z in range(0,results[w]['masks'].shape[2]):
                # Modify your desired object here.
                if(results[w]['masks'][i,j,z] and (results[w]['class_ids'][z] == 48 or results[w]['class_ids'][z] == 50 )):
                    #mask[i,j] = results[w]['class_ids'][z]
                    mask[i,j] = 255

    mask = Image.fromarray(mask)
    mask.save(os.path.join(IMAGE_DIR,resultFolder,Names[w]+'.bmp'),'bmp')
    process_bar.show_process()

print('Done')
