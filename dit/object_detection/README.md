# DiT for Object Detection

## Fine-Tuning On FUNSD

If you woud like to fine-tune the Object Detection Model on FUNSD. You can run the following commands:

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
 
## Run the fine-tuning code in the python file named train_net_funsd.py 
The arguments required are as follows:

- the path of the config file
- the number of gpus required 
- eval only flag in case you want to just evaluate your model not train it
- initial model weights, which in this case I chose to be the weights of DiT Mask RCNN Base Model that was Trained on PubLayNet

```

python3 /unilm/dit/object_detection/train_net_funsd.py --config-file /unilm/dit/object_detection/funsd/maskrcnn_dit_base.yaml --num-gpus 8 MODEL.WEIGHTS https://layoutlm.blob.core.windows.net/dit/dit-fts/icdar19modern_dit-b_mrcnn.pth

```

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
