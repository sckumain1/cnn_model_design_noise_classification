import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


base_dir = r"C:/Users/Sandeep/Deep_Learning_Project_Work/Noise_Classification_5classes/"
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')


test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(300, 300), batch_size=20, color_mode='grayscale', class_mode='categorical', shuffle=False)

# Load the model
print(test_generator.class_indices)
model = tf.keras.models.load_model('model-epoch21.h5py')


# ***********Evaluation of test Accuracy ****************

scores = model.evaluate_generator(test_generator, 6000)
print("  Test Accuracy = ", scores[1])
print("  Test Loss =", scores[0])

# Evaluation of predicted result and print confusion matrix and classification report
batch_size = 20
# y_pred = model.predict_generator(test_generator, 1300 // batch_size+1)
y_pred = model.predict_generator(test_generator, test_generator.samples // test_generator.batch_size)
y_pred = np.argmax(y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))

print('Classification Report')
num_classes = 5
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_generator.classes, y_pred, target_names=target_names))


# Print result in csv files
labels = test_generator.class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in y_pred]
filename = test_generator.filenames
results = pd.DataFrame({"Filename": filename, " Predictions ": predictions})
results.to_csv("result.csv", index=False)

