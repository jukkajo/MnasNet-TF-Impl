from tensorflow.keras import optimizers, layers, models, callbacks, utils, preprocessing, regularizers
from tensorflow.keras  import backend as K
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

def mnas_net(n_classes=1, input_shape=(224, 224, 3), alpha=1):
	inputs = layers.Input(shape=input_shape)

	architecture = conv_bn(inputs, 32*alpha, 3,   strides=2)
	architecture = sep_conv_w_bn_noskip(architecture, 16*alpha, 3,  strides=1)
	
	# MBConv3 3x3
	architecture = mb_conv_idskip(architecture, filters=24, kernel_size=3,  strides=2, filters_multiplier=3, alpha=alpha)
	architecture = mb_conv_idskip(architecture, filters=24, kernel_size=3,  strides=1, filters_multiplier=3, alpha=alpha)
	architecture = mb_conv_idskip(architecture, filters=24, kernel_size=3,  strides=1, filters_multiplier=3, alpha=alpha)
	# MBConv3 5x5
	architecture = mb_conv_idskip(architecture, filters=40, kernel_size=5,  strides=2, filters_multiplier=3, alpha=alpha)
	architecture = mb_conv_idskip(architecture, filters=40, kernel_size=5,  strides=1, filters_multiplier=3, alpha=alpha)
	architecture = mb_conv_idskip(architecture, filters=40, kernel_size=5,  strides=1, filters_multiplier=3, alpha=alpha)
	# MBConv6 5x5
	architecture = mb_conv_idskip(architecture, filters=80, kernel_size=5,  strides=2, filters_multiplier=6, alpha=alpha)
	architecture = mb_conv_idskip(architecture, filters=80, kernel_size=5,  strides=1, filters_multiplier=6, alpha=alpha)
	architecture = mb_conv_idskip(architecture, filters=80, kernel_size=5,  strides=1, filters_multiplier=6, alpha=alpha)
	# MBConv6 3x3
	architecture = mb_conv_idskip(architecture, filters=96, kernel_size=3,  strides=1, filters_multiplier=6, alpha=alpha)
	architecture = mb_conv_idskip(architecture, filters=96, kernel_size=3,  strides=1, filters_multiplier=6, alpha=alpha)
	# MBConv6 5x5
	architecture = mb_conv_idskip(architecture, filters=192, kernel_size=5,  strides=2, filters_multiplier=6, alpha=alpha)
	architecture = mb_conv_idskip(architecture, filters=192, kernel_size=5,  strides=1, filters_multiplier=6, alpha=alpha)
	architecture = mb_conv_idskip(architecture, filters=192, kernel_size=5,  strides=1, filters_multiplier=6, alpha=alpha)
	architecture = mb_conv_idskip(architecture, filters=192, kernel_size=5,  strides=1, filters_multiplier=6, alpha=alpha)
	# MBConv6 3x3
	architecture = mb_conv_idskip(architecture, filters=320, kernel_size=3,  strides=1, filters_multiplier=6, alpha=alpha)
	# Flattening layer and custom 2 branch bbox and class detection head for binary problems
	flatten = Flatten()(architecture)

	bounding_box_head = Dense(128, activation="relu")(flatten)
	bounding_box_head = Dense(64, activation="relu")(bounding_box_head)
	bounding_box_head = Dense(32, activation="relu")(bounding_box_head)
	bounding_box_head = Dense(4, activation="sigmoid", name="bounding_box")(bounding_box_head)

	softmax_head = Dense(512, activation="relu")(flatten)
	softmax_head = Dropout(0.5)(softmax_head)
	softmax_head = Dense(512, activation="relu")(softmax_head)
	softmax_head = Dropout(0.5)(softmax_head)
	softmax_head = Dense(n_classes, activation="softmax", name="class_label")(softmax_head)

	model = Model(inputs=inputs, outputs=(bounding_box_head, softmax_head))
	
	# By default, this would be the normal MnasNet Head:
	"""
	# Fully connected & pooling
	architecture = conv_bn(architecture, filters=1152*alpha, kernel_size=1,   strides=1)
	architecture = layers.GlobalAveragePooling2D()(architecture)
	predictions = layers.Dense(n_classes, activation='softmax')(architecture)

	return models.Model(inputs=inputs, outputs=predictions)
	"""
	
	return model
     
# Convolution with batch normalisatio
def conv_bn(l, filters, kernel_size, strides=1, alpha=1, activation=True):
    filters = _make_divisible(filters * alpha)
    l = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                  use_bias=False, kernel_regularizer=regularizers.l2(0.0003))(l)
    l = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(l)  
    if activation:
        l = layers.ReLU(max_value=6)(l)
    return l

# Depth-wise Separable Convolution & batch normalization 
def depth_wise_conv_bn(l, depth_multiplier, kernel_size, strides=1):
    l = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, depth_multiplier=depth_multiplier, padding='same', use_bias=False)(l)
    l = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(l)  
    l = layers.ReLU(max_value=6)(l)
    return l

# Depth-wise Separable Convolution & batch normalization but no skip
def sep_conv_w_bn_noskip(l, filters, kernel_size,  strides=1):
	l = depth_wise_conv_bn(l, depth_multiplier=1, kernel_size=kernel_size, strides=strides)
	l = conv_bn(l, filters=filters, kernel_size=1, strides=1)
	return l

# Inverted bottleneck block & identity skip connection
def mb_conv_idskip(l_input, filters, kernel_size,  strides=1, filters_multiplier=1, alpha=1):

	depthwise_conv_filters = _make_divisible(l_input.shape[3])
	pointwise_conv_filters = _make_divisible(filters * alpha)

	l = conv_bn(l_input, filters=depthwise_conv_filters * filters_multiplier, kernel_size=1, strides=1)
	l = depth_wise_conv_bn(l, depth_multiplier=1, kernel_size=kernel_size, strides=strides)
	l = conv_bn(l, filters=pointwise_conv_filters, kernel_size=1, strides=1, activation=False)

	# Try residual connection if possible
	if strides==1 and l.shape[3] == l_input.shape[3]:
		return  layers.add([l_input, l])
	else: 
		return l

#----------------------------------------------------------------------------------------------------------------

# From the original tf repository (mobilenet). With this, all layers have a channel number that is divisible by 8
def _make_divisible(v, divisor=8, min_value=None):
	if min_value is None:
		min_value = divisor
	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v


if __name__ == "__main__":

	model = mnas_net()
	model.compile(optimizer='adam')
	model.summary()
