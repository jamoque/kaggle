# The Nature Conservancy Fisheries Monitoring

### Description
The goal is to classify species of fish that occur in images. The full outline of the task can be found here:
https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring

### Instructions for running:

1. Download/clone this repo
2. Download/clone the repo at https://github.com/jamoque/tensorflow-vgg into `trainable_models/`
3. `$ mv src/trainable_models/tensorflow-vgg src/trainable_models/tensorflow_vgg`
4. To train:
	* `$ cd src`
	* `$ python train_vgg.py`
5. To test:
    * If you want to see the model's performance on the labeled test set, run:
        ```$ python eval.py src```
    * If you want to generate output for submission to the Kaggle competition, run:
        ```$ python eval.py src --submission > my_output_file.csv```

**NOTE**: you'll have to manually add the following to the top of `my_output_file.csv`: `image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT` before you submit. Sorry for the inconvenience!
