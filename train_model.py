from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Image settings
img_size = (100, 100)
batch_size = 32

# Load dataset
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train = datagen.flow_from_directory('dataset', target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='training')
val = datagen.flow_from_directory('dataset', target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='validation')

# CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train, validation_data=val, epochs=10)

# Save model
model.save('model/mask_detector.model')
