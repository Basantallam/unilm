# DiT for Object Detection

## Fine-Tuning On FUNSD

### If you woud like to fine-tune the DiT Graphical Object Detection Model on FUNSD, you can run the following commands:

### Clone Detectron2 Repository, which will be our detection Framework

```
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
```

## Install Some Required Modules

```
pip install shapely
pip install opencv-python-headless
```

### Clone Our Repository, which contains:

- FUNSD Dataset images and annotations modified to be in COCO-Style 
- Config file for the model
- The python code for fine-tuning on FUNSD, evaluating and inference

```
git clone https://github.com/Basantallam/unilm
```
 After Cloning Our Repository using the previous command, you will have a directory called unilm, which we'll be referencing in the next steps
 
## Run the Fine-Tning Code

You'll find a python file named train_net_funsd.py which you'll need to run to fine-tune the model.

### The arguments required are as follows:

- config-file: the path of the config file
- num-gpus: the number of gpus required 
- eval-only: a flag in case you want to just evaluate your model not train it
- MODEL.WEIGHTS: the initial model weights that you will finetune, which in this case I chose to be the weights of DiT Mask RCNN Base Model that was Trained on PubLayNet
- OUTPUT_DIR: which is the directory the checkpoints and final model will be saved at

```
python3 /unilm/dit/object_detection/train_net_funsd.py --config-file /unilm/dit/object_detection/funsd/maskrcnn_dit_base.yaml --num-gpus 8 MODEL.WEIGHTS https://layoutlm.blob.core.windows.net/dit/dit-fts/icdar19modern_dit-b_mrcnn.pth OUTPUT_DIR <path of directory you'll save the model in>
```

During Training, You will see in the terminal the loss, and the Average Precision during training, and the model has a CHECKPOINT_PERIOD of 2000, so it constantly saves checkpoints of the model.

## Citation

```
@misc{li2022dit,
    title={DiT: Self-supervised Pre-training for Document Image Transformer},
    author={Junlong Li and Yiheng Xu and Tengchao Lv and Lei Cui and Cha Zhang and Furu Wei},
    year={2022},
    eprint={2203.02378},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```



## Acknowledgment
Thanks to [Detectron2](https://github.com/facebookresearch/detectron2) for Mask R-CNN and Cascade Mask R-CNN implementation.
