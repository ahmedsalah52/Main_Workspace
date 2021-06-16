import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten , Dense ,Lambda , Conv2D, MaxPooling2D ,Dropout , Cropping2D


import tensorflow as tf
from numba import cuda 
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.models import load_model
import sklearn

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))




import keras
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, files, batch_size=2, dim=(160, 160), n_channels=3,shuffle=False):
        'Initialization'
        self.dim = dim
        self.files = files
        self.batch_size = int(batch_size / 6) # because each line of data generates 6 photos and an angle
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.len = int(np.floor(len(self.files) / self.batch_size)) 


    def __len__(self):
        'Denotes the number of batches per epoch'
        print()
        print ("Number of batches per epoch" , self.len)
        print()
        return self.len
    
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        files_temp = [self.files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(files_temp)

        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, files_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        samples = files_temp
        num_samples = len(samples)
        images = []
        measurements = []
        Correction = 20 
        
        for line in samples:
            source_path = line[0]
            filename = source_path.split('/')[-1]
            current_path = 'data/IMG/' + filename
            image = cv2.imread(current_path)   
            images.append(image)
            measurement = float(line[3])
            measurement *= 100  #############################################################
            measurements.append(measurement)

            image_flipped = np.fliplr(image)
            images.append(image_flipped)    
            measurement_flipped = (-1) * measurement
            measurements.append(measurement_flipped)            

            source_pathL = line[1]
            filenameL = source_pathL.split('/')[-1]
            current_pathL = 'data/IMG/' + filenameL
            imageL = cv2.imread(current_pathL)   
            images.append(imageL)
            measurements.append(measurement + Correction)

            image_flippedL = np.fliplr(imageL)
            images.append(image_flippedL)    
            measurement_flippedL = ((-1) * (measurement + Correction))
            measurements.append(measurement_flipped)



            source_pathR = line[2]
            filenameR = source_pathR.split('/')[-1]
            current_pathR = 'data/IMG/' + filenameR
            imageR = cv2.imread(current_pathR)   
            images.append(imageR)
            measurements.append(measurement - Correction)

            image_flippedR = np.fliplr(imageR)
            images.append(image_flippedR)    
            measurement_flippedR = ((-1) * (measurement - Correction))
            measurements.append(measurement_flipped)


        images = np.array(images)
        measurements = np.array(measurements)
        X, Y = sklearn.utils.shuffle(images, measurements)
        return  X, Y  #keras.utils.to_categorical(y, num_classes=self.n_classes)

   

...



lines = []
with open('data/driving_log.csv') as csvfile ,open('data/driving_log2.csv') as csvfile2:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
      
    reader2 = csv.reader(csvfile2)
    for line in reader2:
        lines.append(line)
        
# compile and train the model using the generator function

samples_num = len(lines)
ratio = 0.8


lines = sklearn.utils.shuffle(lines)

train = lines[0:int(ratio * samples_num)]

valid = lines[int(ratio * samples_num):]

params = {'dim': (160,320),
              'batch_size': 18,
              'n_channels': 3,
              'shuffle': True}

            
train_generator =  DataGenerator(train,**params)

validation_generator = DataGenerator(valid, **params)


model = load_model('model.h5')


#adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss='mse' , optimizer = 'adam')


history_object = model.fit_generator(train_generator, steps_per_epoch= train_generator.len ,
validation_data=validation_generator, validation_steps=validation_generator.len, epochs=2,verbose = 1 , workers=16
                                    ,max_queue_size=10) 



model.save('model_3.h5')

print('Model Saved')


### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


#clear memory
device = cuda.get_current_device()
device.reset()

print('Cuda reset')