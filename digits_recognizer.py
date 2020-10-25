import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras import layers
from keras.utils.np_utils import to_categorical


# Raw Data Inputs
test_data, train_data = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Digits/test.csv"), pd.read_csv(
    "/content/drive/My Drive/Colab Notebooks/Digits/train.csv")

# Sort X From Y
y_train, x_train = train_data["label"], train_data.drop(labels=["label"], axis=1)
x_test = test_data.values.astype('float32')

# Reshaping data to fit the model and -> gray
x_train = x_train.values.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Normalization
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One hot Encoding
y_train = to_categorical(y_train)

# Layers
inputs = keras.Input(shape=(28, 28, 1))
x = layers.experimental.preprocessing.RandomZoom(0.1)(inputs)
x = layers.Conv2D(32, (5, 5), activation='relu')(x)
x = layers.BatchNormalization(axis=1)(x)
x = layers.Conv2D(32, (5, 5), activation='relu')(x)
x = layers.BatchNormalization(axis=1)(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)
x = layers.BatchNormalization(axis=1)(x)
x = layers.Dropout(0.2)(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.BatchNormalization(axis=1)(x)
x = layers.Conv2D(64, (3, 3), activation='relu', )(x)
x = layers.BatchNormalization(axis=1)(x)
x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
x = layers.Flatten()(x)
x = layers.BatchNormalization(axis=1)(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.BatchNormalization(axis=1)(x)
output = layers.Dense(10, activation='softmax')(x)

# Models
digits_model = keras.Model(inputs, output)

# Compilation
optimizer = keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9, momentum=0.9, epsilon=1e-07, decay=0.0005)
'''optimizer = keras.optimizers.Adam(lr=0.1, decay=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1 ** (-8))'''
loss = keras.losses.CategoricalCrossentropy()
digits_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Training
history = digits_model.fit(x_train, y_train, batch_size=86, epochs=30, validation_split=0.1 )

# Plotting some graph to have a better view of the model results
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

predictions = digits_model.predict(x_test)

predictions = np.argmax(predictions, axis=1)

predictions = pd.Series(predictions, name="Label")
submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), predictions], axis=1)

submission.to_csv("/content/drive/My Drive/Colab Notebooks/Digits/predictions.csv", index=False)
