import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

train_data_gen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
)
validation_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
        'data/train',
        target_size=(48, 48),
        batch_size=60,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = validation_data_gen.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=15,
        color_mode="grayscale",
        class_mode='categorical')

expression_model = Sequential()

expression_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
expression_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
expression_model.add(MaxPooling2D(pool_size=(2, 2)))

expression_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
expression_model.add(MaxPooling2D(pool_size=(2, 2)))
expression_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
expression_model.add(MaxPooling2D(pool_size=(2, 2)))

expression_model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
expression_model.add(MaxPooling2D(pool_size=(2, 2)))


expression_model.add(Flatten())
expression_model.add(Dense(1024, activation='relu'))
expression_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

expression_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001, decay=1e-6), metrics=['accuracy'])

history = expression_model.fit(train_generator, epochs=30, validation_data=validation_generator)

fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
fig.set_size_inches(12,4)

ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('Training Accuracy vs Validation Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Validation'], loc='upper left')

ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Training Loss vs Validation Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Validation'], loc='upper left')

plt.show()
# expression_model_info = expression_model.fit_generator(
#         train_generator,
#         steps_per_epoch=2100//60,
#         epochs=30,
#         validation_data=validation_generator,
#         validation_steps=630//15)

model_json = expression_model.to_json()
with open("expression_model.json", "w") as json_file:
    json_file.write(model_json)

expression_model.save_weights('expression_model.h5')