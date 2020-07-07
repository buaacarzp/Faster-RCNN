from frcnn_all import FasterRCNN
import torch
from PIL import Image, ImageFont, ImageDraw
import numpy as np 
import copy
from utils import DecodeBox
import os 
import colorsys
def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_width, resized_height
def image_process(image):
    image_shape = np.array(np.shape(image)[0:2])
    old_width = image_shape[1]
    old_height = image_shape[0]
    ori_image = copy.deepcopy(image)
    width,height = get_new_img_size(old_width,old_height)
    image = image.resize([width,height])
    photo = np.array(image,dtype = np.float32)/255
    photo = np.transpose(photo, (2, 0, 1))
    return ori_image,photo,width,height,old_width,old_height
class Propsals_Process(object):
    def __init__(self):
        self.classes_path = '../faster-rcnn-pytorch/model_data/voc_classes.txt'
        self.class_names = self._get_class()
        self.colors =self.color()
        self.num_classes =20
        self.mean = torch.Tensor([0,0,0,0]).cuda().repeat(self.num_classes+1)[None]
        self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).cuda().repeat(self.num_classes+1)[None]
        self.confidence =0.5
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    def color(self):
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
        return colors
    def Decode_Propsals(self,roi_cls_locs, roi_scores, rois,height,width):
        decodebox = DecodeBox(self.std, self.mean, self.num_classes)
        outputs = decodebox.forward(roi_cls_locs, roi_scores, rois, height=height, width=width, score_thresh = self.confidence)
        return outputs
    
    def Visiualize_Propsals(self,ori_image,outputs,old_width,old_height):
        if len(outputs)==0:
            return ori_image
        bbox = outputs[:,:4]
        conf = outputs[:, 4]
        label = outputs[:, 5]
        bbox[:, 0::2] = (bbox[:, 0::2])/width*old_width
        bbox[:, 1::2] = (bbox[:, 1::2])/height*old_height
        bbox = np.array(bbox,np.int32)
        image = ori_image
        thickness = (np.shape(ori_image)[0] + np.shape(ori_image)[1]) // old_width*2
        font = ImageFont.truetype(font='../faster-rcnn-pytorch/model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        for i, c in enumerate(label):
            predicted_class = self.class_names[int(c)]
            score = conf[i]

            left, top, right, bottom = bbox[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image
        

image_path = "imgs/street.jpg"
model_path ="model/logs/Epoch31-Total_Loss0.6042-Val_Loss0.6924.pth"#"../faster-rcnn-pytorch/model_data/voc_weights_resnet.pth"
faster_rcnn = FasterRCNN(20,"predict",backbone="resnet50").cuda()
faster_rcnn.load_state_dict(torch.load(model_path))

image = Image.open(image_path)
ori_image,image,width,height,old_width,old_height = image_process(image)
images = np.asarray([image])
images = torch.from_numpy(images).cuda()
#roi_cls_locs 为propsal的偏差,rois是rpn输出修正后的输出坐标
roi_cls_locs, roi_scores, rois_ori, roi_indices = faster_rcnn(images)
Propsals = Propsals_Process()
outputs = Propsals.Decode_Propsals(roi_cls_locs, roi_scores, rois_ori,height,width)
draw_image =Propsals.Visiualize_Propsals(ori_image,outputs,old_width,old_height)
draw_image.save("imgs/resultepoch31.jpg")