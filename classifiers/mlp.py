import tensorflow as tf
import numpy as np
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from config import BARCLAYS_CSV_21_24, BARCLAYS_CSV_24_25, CATEGORIES
import pandas as pd
from utils.barclays.format_csv import format_memo
from utils.evaluate import accuracy

BERT_DIM = 384
NUM_LABELS = len(CATEGORIES)
labels_to_idx = { k : v for v, k in enumerate(CATEGORIES)}
idx_to_labels = { k : v for k, v in enumerate(CATEGORIES)}
train_df = pd.read_csv(BARCLAYS_CSV_21_24)
test_df = pd.read_csv(BARCLAYS_CSV_24_25)

def create_model(input=BERT_DIM, num_classes=NUM_LABELS):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train(model, raw_labels, data, epochs):
    # Load BERT model
    bert = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert text descriptions into embeddings
    embeddings = bert.encode(data)
    labels = np.array([labels_to_idx[label] for label in raw_labels])

    model.fit(
        embeddings, labels,
        epochs=epochs,
        batch_size=16,
        validation_split=0.2
    )

    return model


def predict(model, data, test_df):
    # Load BERT model
    bert = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert text descriptions into embeddings
    embeddings = bert.encode(data)

    predictions = model.predict(embeddings)
    maxProbs = predictions.max(-1)
    predicted_indices = np.argmax(predictions, axis=-1)
    predicted_categories = [idx_to_labels[idx] for idx in predicted_indices]
    return predicted_categories, maxProbs


def weightAverage(weights_1, weights_2):
    newWeights = []
    for w1, w2 in zip(weights_1, weights_2):
        newWeights.append(np.add(np.multiply(w1, 0.5), np.multiply(w2, 0.5)))
    return newWeights

def inspect_weights(weights):
    print(len(weights))
    for w in weights:
        print(w.shape, type(w))

# half = len(train_df) // 2
# first_half = train_df.iloc[:half, :]
# second_half = train_df.iloc[half:, :].reset_index()
# model1 = train(create_model(), first_half['Category'], first_half['Memo'], 15)
# model2 = train(create_model(), second_half['Category'], second_half['Memo'], 15)

# averaged_model = create_model()
# averaged_model.set_weights(weightAverage(model1.get_weights(), model2.get_weights()))

# inspect_weights(averaged_model.get_weights())

# predictions, probabilities = predict(averaged_model, test_df['Memo'], test_df)
# predicted_df = test_df.assign(Category=predictions)
# print(f"Averaged MLP accuracy: {accuracy(predicted_df, test_df) * 100:.2f}%")

model = train(create_model(), train_df['Category'], train_df['Memo'], 15)
model.save("model_1.h5")
predictions, probabilities = predict(model, test_df['Memo'], test_df)
predicted_df = test_df.assign(Category=predictions)

print(f"MLP accuracy: {accuracy(predicted_df, test_df) * 100:.2f}%")