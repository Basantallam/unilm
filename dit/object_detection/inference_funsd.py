import argparse
import os
import cv2
import tarfile
import json
from ditod import add_vit_config

import torch

from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures.instances import Instances

def main():
    parser = argparse.ArgumentParser(description="Detectron2 inference script")
    #parser.add_argument(
    #    "--images_directory",
    #    help="Path to input image",
    #    type=str,
    #    required=True,
    #)
    parser.add_argument(
        "--output_directory",
        help="Name of the output directory",
        type=str,
    )
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # Step 1: instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    
    # Step 2: add model weights URL to config
    cfg.merge_from_list(args.opts)
    
    # Step 3: set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    # Step 4: define model
    predictor = DefaultPredictor(cfg)
    
    # Step 5: run inference
    x=0 #image counter
    bbox_pred=[]
    #with open('/netscratch/allam/name_to_id.json') as f:
    #    image_to_id = json.load(f)
    #print(len(image_to_id.keys()))
    #print(image_to_id)
    keys=['image_name','category_id','bbox','score']
    list_of_names=os.listdir('/content/unilm/dit/object_detection/FUNSD_dataset/testing_data/images/')
    print(len(list_of_names))

    for image_name in list_of_names:
    
      if not image_name.endswith('.jpg') and image_name.endswith('.png'):
        print("skipped ", image_name)
        continue
      
      img = cv2.imread(os.path.join('/content/unilm/dit/object_detection/FUNSD_dataset/testing_data/images/',image_name))
      if x%10==0:
        print(x)

      #print(image_name)
      md = MetadataCatalog.get(cfg.DATASETS.TEST[0])

      md.set(thing_classes=["dummy","question","answer","header","other"])
      
      output = predictor(img)["instances"]
      
      #print("fields ",Instances.get_fields(output))

      d=Instances.get_fields(output) 
      # dictionary of all predictions for this image
      #curr_list = list()
      # loop on all predictions for this image
      #for i in range(len(d['pred_classes'])):
        #curr_list.append(image_name)

        #print("pred class ", d['pred_classes'][i].item()," ", type(d['pred_classes'][i])," ",type(d['pred_classes'][i].item()))

        #curr_list.append(d['pred_classes'][i].item())
       
      #  pbox=d['pred_boxes'][i].tensor.cpu().numpy()
       # pbox_arr=list()
        #for n in pbox[0]:
          #print(n)
          #print()
         # pbox_arr.append(n)
        #print("bounding box ",pbox_arr," ", type(pbox_arr)," ", type(d['pred_boxes'][i]))

      #  curr_list.append(pbox_arr)
   
        #print("score ",d['scores'][i].item()," ", type(d['scores'][i].cpu().numpy())," ", type(d['scores'][i].item()))
       # curr_list.append(d['scores'][i].item())

        #print(curr_list)
        zipped=dict(zip(keys,curr_list))
        #print(zipped)
        #print(type(zipped))
        #curr_list = list()
        #bbox_pred.append(zipped)
        #print(type(bbox_pred))
      #print(Instances.get(output, 'pred_boxes'))
      #print(output.keys())
      v = Visualizer(img[:, :, ::-1],md,scale=1.0,instance_mode=ColorMode.SEGMENTATION)
      result = v.draw_instance_predictions(output.to("cpu"))
      result_image = result.get_image()[:, :, ::-1]
      #print(os.path.join(args.output_directory,os.path.splitext(image_name)[0]+"_output.jpg"))
      #step 6: save
      cv2.imwrite(os.path.join(args.output_directory,os.path.splitext(image_name)[0]+"_output.jpg"), result_image)
      x=x+1     

   # with open('netscratch/allam/OD_val_pred/bboxes/bbox_pred_val.json', 'w') as fp:
    #  json.dump(str(bbox_pred), fp)
    print("done all bbox files and preds!!! ")
    print(len(list_of_names))
if __name__ == '__main__':
    main()
