# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Reshape
# from tensorflow.keras.layers import Input
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, BatchNormalization, ReLU, Concatenate, Activation
# from tensorflow.keras.constraints import max_norm
# from tensorflow.keras import backend
# def conv_block(x, filters, kernel_size, strides=2):
#    x = Conv2D(filters=filters,
# 			  kernel_size=kernel_size,
# 			  strides=strides,
# 			  padding='same')(x)
#    x = BatchNormalization()(x)
#    x = ReLU()(x)
#    return x

# def deconv_block(x, filters, kernel_size):
#    x = Conv2DTranspose(filters=filters,
# 					   kernel_size=kernel_size,
# 					   strides=2,
# 					   padding='same')(x)
#    x = BatchNormalization()(x)
#    x = ReLU()(x)
#    return x

# class Autoencoder:
# 	def build(self, height, width, channels):
# 		# Энкодер
# 		# вход encoder
# 		input_img = Input(shape=(height, width, channels))

# 		conv_block1 = conv_block(input_img, 32, 3)
# 		conv_block2 = conv_block(conv_block1, 64, 3)
# 		conv_block3 = conv_block(conv_block2, 128, 3)
# 		conv_block4 = conv_block(conv_block3, 256, 3)
# 		encoded = conv_block(conv_block4, 256, 3, 1)

# 		# Декодер
# 		# вход decoder
# 		deconv_block1 = deconv_block(encoded, 128, 3)
# 		merge1 = Concatenate()([deconv_block1, conv_block3])
# 		deconv_block2 = deconv_block(merge1, 128, 3)
# 		merge2 = Concatenate()([deconv_block2, conv_block2])
# 		deconv_block3 = deconv_block(merge2, 64, 3)
# 		merge3 = Concatenate()([deconv_block3, conv_block1])
# 		deconv_block4 = deconv_block(merge3, 32, 3)
# 		final_deconv = Conv2DTranspose(filters=3, kernel_size=3, padding='same')(deconv_block4)
# 		decoded = Activation('sigmoid', name='decoded')(final_deconv)

# 		autoencoder = Model(input_img, decoded, name="autoencoder")
# 		return autoencoder
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from tensorflow.keras.layers import Reshape

class Autoencoder:
    # Размерность кодированного представления
    encoding_dim = 49 * 3

    def build(self, height, width, channels):
        input_img = Input(shape=(height, width, channels))

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        return autoencoder




