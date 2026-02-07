# adding libraries
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification


# loading dataset
df = pd.read_csv("fake_or_real_news1E.csv")

# clean and map labels safely
df['label'] = df['label'].astype(str).str.lower()

df['label'] = df['label'].map({
    'fake': 1,
    'real': 0,
    '1': 1,
    '0': 0
})

# remove rows where label is still NaN
df = df.dropna(subset=['label'])

df['label'] = df['label'].astype(int)


# combine title and text
df['combined_text'] = df['title'].astype(str) + " " + df['text'].astype(str)


# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['combined_text'].tolist(),
    df['label'].values,
    test_size=0.2,
    random_state=42
)


# tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(
    X_train,
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="tf"
)

test_encodings = tokenizer(
    X_test,
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="tf"
)


# load BERT model
model = TFDistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)


# compile model
model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)


# handle class imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = {
    0: class_weights[0],
    1: class_weights[1]
}


# train model
print("\nTraining BERT model...\n")

model.fit(
    train_encodings,
    y_train,
    epochs=2,   
    batch_size=8,
    class_weight=class_weight_dict,
    verbose=1
)


# predictions
logits = model.predict(test_encodings).logits
y_pred = np.argmax(logits, axis=1)


# evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\nEvaluation Metrics (BERT):\n")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)
print("\nConfusion Matrix:\n", cm)


# Save the model and tokenizer 
model.save_pretrained("./news_model")
tokenizer.save_pretrained("./news_model")
print("Model saved successfully!")