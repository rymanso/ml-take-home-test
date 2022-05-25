# TensorFlow and tf.keras
import string
import tensorflow as tf

# Helper libraries
import pandas as pd
import re
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import losses

from sklearn.preprocessing import LabelEncoder


def get_dataset(values, labels, batch_size=32):
    return tf.data.Dataset.from_tensor_slices((values, labels)).batch(32)


# This code is based on: https://www.tensorflow.org/tutorials/keras/text_classification
def main():
    # Get the datasets
    data = pd.read_csv("product_sentiment.csv", skiprows=1).to_numpy()

    # Clean
    data = np.array([row for row in data if type(row[1]) == str and type(row[3]) == str])

    n_train = int(0.8 * len(data))

    X = data[:, 1]
    Y = np.array(LabelEncoder().fit_transform(data[:, 3]))

    raw_train_ds = get_dataset(X[:n_train], Y[:n_train])
    raw_test_ds = get_dataset(X[n_train:], Y[n_train:])

    #  Preprocess the data
    max_features = 10000
    sequence_length = 250
    vectorize_layer = layers.TextVectorization(
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    # Make a text-only dataset (without labels), then call adapt
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    train_ds = raw_train_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Make the model
    embedding_dim = 16
    model = tf.keras.Sequential(
        [
            layers.Embedding(max_features + 1, embedding_dim),
            layers.Dropout(0.2),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.2),
            layers.Dense(1),
        ]
    )

    model.summary()

    # Loss function and optimizer
    model.compile(
        loss=losses.BinaryCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=tf.metrics.BinaryAccuracy(threshold=0.0),
    )

    # Train
    model.fit(train_ds, epochs=10)

    # Test
    loss, accuracy = model.evaluate(test_ds)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    # Prepare it for export
    export_model = tf.keras.Sequential([vectorize_layer, model, layers.Activation("sigmoid")])

    export_model.compile(
        loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=["accuracy"]
    )

    # Test it with `raw_test_ds`, which yields raw strings
    loss, accuracy = export_model.evaluate(raw_test_ds)
    print("export model with raw data accuracy:", accuracy)
    export_model.save("saved_model/my_model")


if __name__ == "__main__":
    main()
