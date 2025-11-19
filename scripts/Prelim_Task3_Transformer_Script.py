# This script is meant to train and create a transformer model for classification. It uses Keras. 
# The R data file was converted to CSV locally. It was used here and will also be uploaded to the git with this script* (Edit: Nvm, forgot the CSV is too large oops. It's just claims clean but CSV). 
# -Ben 


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

vocab_size = 3000
max_length = 150
embed_dim = 64
num_heads = 2
ff_dim = 128
batch_size = 32
epochs = 30

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, max_length, embed_dim):
        super().__init__()
        self.token_embed = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_embed = layers.Embedding(input_dim=max_length, output_dim=embed_dim)
    
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        return self.token_embed(inputs) + self.pos_embed(positions)


def create_model(num_classes=2):
    inputs = layers.Input(shape=(max_length,))
    x = PositionalEmbedding(vocab_size, max_length, embed_dim)(inputs)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


data_path = r"C:\Users\bentu\Downloads\pstat197_ucsb\module-2-claims-data-table-8\data\claims-clean.csv"
df = pd.read_csv(data_path)

# forcing labels to work lol
texts = df['text_clean'].fillna('').astype(str).tolist()
label_mapping = {label: idx for idx, label in enumerate(df['bclass'].unique())}
labels = df['bclass'].map(label_mapping).values


vectorizer = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=max_length
)
vectorizer.adapt(texts)

X = vectorizer(np.array(texts)).numpy()
y = labels

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = create_model(num_classes=2)
model.summary()


history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    verbose=1
)

# .5 coinflip equiv. 
y_pred = (model.predict(X_val) > 0.5).astype(int).flatten()
accuracy = accuracy_score(y_val, y_pred)


print(classification_report(y_val, y_pred))

print(accuracy)
