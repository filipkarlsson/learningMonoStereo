from tensorflow.python.keras.layers import Input, Dense, Conv2D, concatenate, Lambda, Conv2DTranspose
from tensorflow.python.keras.models import Model
from tensorflow import slice 



#keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

# Load data set

# Design layers


# 256x192x(3*2)
image_pair = Input(shape = (256,192,6))

layer_1a = Conv2D(32,kernel_size=(9,1), strides=(2,1), activation='relu',padding='same',input_shape=(256,192,6))(image_pair)
layer_1b =  Conv2D(32,kernel_size=(1, 9), strides=(1,2), activation='relu', padding='same')(layer_1a)
print(layer_1a)
print(layer_1b)

# 128x96x32

layer_2a = Conv2D(32,kernel_size=(7,1), strides=(2,1), activation='relu', padding='same')(layer_1b)
layer_2b =  Conv2D(32,kernel_size=(1, 7), strides=(1,2), activation='relu', padding='same')(layer_2a)

print(layer_2a)
print(layer_2b)
# 64x48x32


# input from previous estimate only avialbile in rec net
aux_input = Input( shape = (64,48,8))
aux_input_layer_a = Conv2D(32,kernel_size=(3,1), strides=(1,1), activation='relu',padding='same',input_shape=(64,48,8))(aux_input)
aux_input_layer_b = Conv2D(32,kernel_size=(1,3), strides=(1,1), activation='relu',padding='same')(aux_input_layer_a)


print(aux_input_layer_a)
print(aux_input_layer_b)

# Merge
merged_input = concatenate([aux_input_layer_b, layer_2b])

print(merged_input)


layer3a =  Conv2D(64,kernel_size=(3, 1), strides=(1,1), activation='relu', padding='same')(merged_input)
# layer3b should be feeded forward later
layer3b =  Conv2D(64,kernel_size=(1, 3), strides=(1,1), activation='relu', padding='same')(layer3a)

print(layer3b)

layer4a =  Conv2D(128,kernel_size=(5, 1), strides=(2,1), activation='relu', padding='same')(layer3b)
layer4b =  Conv2D(128,kernel_size=(1, 5), strides=(1,2), activation='relu', padding='same')(layer4a)
print(layer4b)

layer5a =  Conv2D(128,kernel_size=(3, 1), strides=(1,1), activation='relu', padding='same')(layer4b)
# Layer5b shall be feeded forward
layer5b =  Conv2D(128,kernel_size=(1, 3), strides=(1,1), activation='relu', padding='same')(layer5a)
print(layer5b)


layer6a =  Conv2D(256,kernel_size=(5, 1), strides=(2,1), activation='relu', padding='same')(layer5b)
layer6b =  Conv2D(256,kernel_size=(1, 5), strides=(1,2), activation='relu', padding='same')(layer6a)
print(layer6b)
# 16x12x256

layer7a =  Conv2D(256,kernel_size=(3, 1), strides=(1,1), activation='relu', padding='same')(layer6b)
# Layer7b shall be feeded forward
layer7b =  Conv2D(256,kernel_size=(1, 3), strides=(1,1), activation='relu', padding='same')(layer7a)
print(layer7b)
# 16x12x256

layer8a =  Conv2D(512,kernel_size=(3, 1), strides=(2,1), activation='relu', padding='same')(layer7b)
layer8b =  Conv2D(512,kernel_size=(1, 3), strides=(1,2), activation='relu', padding='same')(layer8a)
print(layer8b)

layer9a =  Conv2D(512,kernel_size=(3, 1), strides=(1,1), activation='relu', padding='same')(layer8b)
# Layer9b shall be feeded forward
layer9b =  Conv2D(512,kernel_size=(1, 3), strides=(1,1), activation='relu', padding='same')(layer9a)
print(layer9b)
# conv finished, now first prediciton and up conv

# First prediction branch
layerFirstPred_1 =  Conv2D(24,kernel_size=(3, 3), strides=(1,1), activation='relu', padding='same')(layer9b)
print(layerFirstPred_1)

layerFirstPred_2 =  Conv2D(4,kernel_size=(3, 3), strides=(1,1), activation=None, padding='same')(layerFirstPred_1)
print(layerFirstPred_2)

# Split tensor into optical flow (8x6x2) and into a confidense estimate (8x6x2)
first_pred_optical_flow = Lambda(lambda x: slice(x, (0, 0, 0, 0), (-1,-1,-1,2)))(layerFirstPred_2)
first_pred_confidence = Lambda(lambda x: slice(x, (0, 0, 0, 2), (-1,-1,-1,2)))(layerFirstPred_2)


print(first_pred_optical_flow)
print(first_pred_confidence)

merged_output = concatenate([first_pred_optical_flow, first_pred_confidence])

print(merged_output)

upconv1_first_pred = Conv2DTranspose(4,kernel_size = (4,4), strides = (2,2),  padding = 'same')(merged_output)
print(upconv1_first_pred)
# End of prediciton branch
upconv1 = Conv2DTranspose(256,kernel_size = (4,4), strides = (2,2),  padding = 'same', activation = 'relu')(layer9b)

upconv1_merge = concatenate([upconv1_first_pred, upconv1, layer7b])
print(upconv1_merge)

upconv2 = Conv2DTranspose(128,kernel_size = (4,4), strides = (2,2),  padding = 'same', activation = 'relu')(upconv1_merge)
print(upconv2)
upconv2_merge = concatenate([upconv2, layer5b])
print(upconv2_merge)


upconv3 = Conv2DTranspose(64,kernel_size = (4,4), strides = (2,2),  padding = 'same', activation = 'relu')(upconv2_merge)

upconv3_merge = concatenate([upconv3, layer3b])
print(upconv3_merge)


layer10 = Conv2D(24,kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(upconv3_merge)
print(layer10)

layer11 = Conv2D(4,kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(layer10)
print(layer11)

# Output :D

optical_flow_output = Lambda(lambda x: slice(x, (0, 0, 0, 0), (-1,-1,-1,2)))(layer11)
confidence_output = Lambda(lambda x: slice(x, (0, 0, 0, 2), (-1,-1,-1,2)))(layer11)


print(optical_flow_output)
print(confidence_output)