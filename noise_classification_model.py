import os
import tensorflow as tf
import matplotlib.pyplot as plt

base_dir = r"C:/Users/Sandeep/Deep_Learning_Project_Work/Noise_Classification_5classes/"
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(300, 300), color_mode='grayscale', batch_size=20, class_mode='categorical')
val_generator = train_datagen.flow_from_directory(val_dir, target_size=(300, 300), color_mode='grayscale', batch_size=20, class_mode='categorical', shuffle=False)

for data_batch, label_batch in train_generator:
    print('data batch shape:',data_batch.shape)
    print('label batch shape:', label_batch.shape)
    break



model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(8, kernel_size=(3,3), activation='relu',input_shape=(300, 300, 1), padding='same'))
model.add(tf.keras.layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))

model.add(tf.keras.layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))

model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))

model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))

model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))


model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.20))

model.add(tf.keras.layers.Dense(2048, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax))

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
checkpoint = tf.keras.callbacks.ModelCheckpoint('model-epoch{epoch:02d}.h5py', monitor='val_acc', verbose=1,
                                                save_best_only=True, save_weights_only=False, mode='auto', period=1)

callbacks_list = [earlystop, checkpoint]
history = model.fit_generator(train_generator, steps_per_epoch=1400, epochs=30, validation_data=val_generator,
                              validation_steps=300, verbose=2, callbacks=callbacks_list)


model.save('final_epoch.h5py')

accuracy = history.history['acc']
loss = history.history['loss']
val_accuracy = history.history['val_acc']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs,val_accuracy, 'r', label='Validation accuracy')
plt.xlabel('Number of Epoch')
plt.ylabel('Accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('model6_accuracy30.png')
plt.gcf().clear()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Number of Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('model6_loss30.png')
