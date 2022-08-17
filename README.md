# In this repository:
## We demonstrate the evaluation of 2 DiT pre-trained models.

1. Document Classification DiT model
2. Graphical Document Object Detection DiT model

Additionally, we fine-tuned the Graphical Document Object Detection DiT model to perform the task of Form Understanding on FUNSD Dataset. We test and evaluate its performance. 

# Training

The training code for each of the three models architecture can be found in [dit](https://github.com/Basantallam/unilm/tree/master/dit) which is composed of 2 directories:

1. [object detection](https://github.com/Basantallam/unilm/tree/master/dit/object_detection) contains the Graphical Document Object Detection DiT model configurations and training code as well as the FUNSD fine-tuned form understanding model configurations and training code.

2. [classification](https://github.com/Basantallam/unilm/tree/master/dit/classification) contains the Document Classification DiT model configurations and training code.


# Evaluation

In this section we demonstrate the code we used to evaluate each of the models. In order to run this code yourself, you'll need the prediction results of the model and the ground truth. However you can see the results we got without having to run the code yourself.

## Evaluation code of Document Classifier:

https://colab.research.google.com/drive/1rsPfpYvw8EWdrQPFybsS-bq3-UqN7Ndh?usp=sharing


## Evaluation code of Graphical Object Detection Validation Set:

https://colab.research.google.com/drive/17cIkVcm3DSeSHuW4RcG3gbtlykry1Vvs?usp=sharing


## Evaluation code of Graphical Object Detection Test Set:

https://colab.research.google.com/drive/1ugzysT9G3ULXUPB4RNe_ZE7z6CApS9n5?usp=sharing


## Evaluation code of Form Understanding (finetuned on FUNSD):

https://colab.research.google.com/drive/1OFTZ0WyplQRqgIbwYW84EbcInWT_25k1?usp=sharing
