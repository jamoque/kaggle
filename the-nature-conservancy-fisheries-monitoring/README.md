# The Nature Conservancy Fisheries Monitoring

### Description
The goal of this repo is to classify species of fish. The full outline of the task can be found here:
https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring

### Instructions for running:

1. Download/clone this repo
2. Download/clone the repo at https://github.com/jamoque/tensorflow-vgg into `trainable_models/`
3. `$ mv src/trainable_models/tensorflow-vgg src/trainable_models/tensorflow_vgg`
4. To train:
	* `$ cd src`
	* `$ python train_vgg.py`
5. To test:
    * `$ python eval.py src`