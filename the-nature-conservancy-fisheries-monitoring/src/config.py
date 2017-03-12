from collections import OrderedDict

#########################
# NETWORK PARAMETERS
#########################

# initial learning rate
learning_rate = 1e-4

# maximum number of steps to run during training
max_steps = 1000000000

# batch size (https://arxiv.org/pdf/1502.03167v3.pdf)
batch_size = 30

# small constant for stability of ADAM optimizer (https://arxiv.org/pdf/1412.6980v8.pdf)
epsilon = 1e-8

# dropout rate (https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
dropout = 0.5

# regularization constant
reg = 1e-3

#########################
# DATASET PARAMETERS
#########################

# the number of categories
num_labels = 109;

# relative path to the directory where data files are stored
data_path = '../data/'

# directory where checkpoints for trained models are saves
train_dir = 'trained_models'

# path to labeled training data
training_label_file_path = data_path + 'train_images.txt'

# path to labeled test data
test_labels_file_path = data_path + 'test_images.txt'

_mapping = OrderedDict({
	'ALB': 0,
	'BET': 1,
	'DOL': 2,
	'LAG': 3,
	'NoF': 4,
	'OTHER': 5,
	'SHARK': 6,
	'YFT': 7
})

num_classes = len(_mapping.keys())

def label_to_int(label):
	return _mapping[label]

def int_to_label(i):
	return _mapping.values()[i]

#########################
# PROCEDURAL PARAMETERS
#########################

# how often to output statistics for training
output_frequency = 100

# how often to test the trained model on test data and checkpoint the model
test_and_save_frequency = 1000
