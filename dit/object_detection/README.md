# DiT for Object Detection

## Fine-Tuning Dit Graphical object detection model on FUNSD for Form Understanding of Noisy Scanned Documents

The FUNSD Dataset has annotations for semantic entity labeling where the entities are "Question", "Answer", "Header" and "Other"

Here is a sample image from the FUNSD Dataset, where you'll see

Questions in blue, Answers in green, Header in orange, Other in pink

![two_forms](https://user-images.githubusercontent.com/30272808/177758850-0189ec22-073c-4bd0-8646-04f18fe142a0.png)

## Setup
If you woud like to fine-tune the DiT Graphical Object Detection Model on FUNSD in order to train the model on the task of semantic entity labeling in Noisy Scanned Documents, you can run the following commands:

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
 
## Run the Fine-Tuning Code

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

During training, you will see in the terminal the loss, and the Average Precision during training, and the model has a checkpoint period of 2000, so it constantly saves checkpoints of the model.

An example of the output of the model on the test set is as follows

![aav31f00_0001435658_output](https://user-images.githubusercontent.com/30272808/177758213-9c83d935-0dc9-4b31-9c96-986f9b40bb48.jpg)

## Citations

```
@misc{https://doi.org/10.48550/arxiv.1905.13538,
  doi = {10.48550/ARXIV.1905.13538},
  url = {https://arxiv.org/abs/1905.13538},
  author = {Jaume, Guillaume and Ekenel, Hazim Kemal and Thiran, Jean-Philippe},
  keywords = {Information Retrieval (cs.IR), Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
  publisher = {arXiv},
  year = {2019},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

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
