# The Nature Conservancy Fisheries Monitoring

### Description
The goal is to classify species of fish that occur in images. The full outline of the task can be found here:
https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring

### Instructions for running:

1. Download/clone this repo
2. Download/clone the repo at https://github.com/jamoque/tensorflow-vgg into `trainable_models/`
	* NOTE: If you want to train the model, you'll probably want to follow the instructions in the tensorflow-vgg repo to download the pre-trained weight file and put it in `trained_models/`
3. If you want to run the predictions, download [this](https://www.dropbox.com/s/uknryqgpdihf4c5/trained-step-16000.npy?dl=0) `.npy` file into `trained_models/`
4. `$ mv src/trainable_models/tensorflow-vgg src/trainable_models/tensorflow_vgg`
5. To train:
	* `$ cd src`
	* `$ python train_vgg.py`
6. To test:
    * If you want to see the model's performance on the labeled test set, run:
        ```$ python eval.py src```
    * If you want to generate output for submission to the Kaggle competition, run:
        ```$ python eval.py src --submission > submissions/my_output_file.csv```
        * **NOTE**: you'll have to manually add the following to the top of `my_output_file.csv`: `image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT` before you submit. Sorry for the inconvenience!
