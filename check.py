from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2

# Define the mapping of label indices to characters
labels = [u'\u091E',u'\u091F',u'\u0920',u'\u0921',u'\u0922',u'\u0923',u'\u0924',u'\u0925',u'\u0926',u'\u0927',u'\u0915',u'\u0928',u'\u092A',u'\u092B',u'\u092c',u'\u092d',u'\u092e',u'\u092f',u'\u0930',u'\u0932',u'\u0935',u'\u0916',u'\u0936',u'\u0937',u'\u0938',u'\u0939','ksha','tra','gya',u'\u0917',u'\u0918',u'\u0919',u'\u091a',u'\u091b',u'\u091c',u'\u091d',u'\u0966',u'\u0967',u'\u0968',u'\u0969',u'\u096a',u'\u096b',u'\u096c',u'\u096d',u'\u096e',u'\u096f']

# Define the path to the test image
test_image_path = r"C:\Users\Arshad\OneDrive\Desktop\saniraj\Hindi_Dataset\Hindi_Dataset\Test\character_23_ba\87901.png"

# Load the test image
test_image = cv2.imread(test_image_path)

# Convert the image to float32
test_image = test_image.astype(np.float32) / 255.0

# Convert the image to grayscale
gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# Expand dimensions to match the expected input shape of the model
input_image = np.expand_dims(gray_image, axis=0)
input_image = np.expand_dims(input_image, axis=3)

# Load the trained model
model = load_model(r"C:\Users\Arshad\OneDrive\Desktop\saniraj\HindiModel2.keras")

# Predict the label for the test image
prediction = model.predict(input_image)[0]
predicted_label_index = np.argmax(prediction)
predicted_label = labels[predicted_label_index]

# Calculate the accuracy percentage
accuracy_percentage = prediction[predicted_label_index] * 100

# Print the predicted letter and accuracy percentage
print("Predicted letter:", predicted_label)
print("Accuracy:", accuracy_percentage, "%")
