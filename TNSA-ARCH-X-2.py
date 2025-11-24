
import tensorflow as tf
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time

# Load the Wikipedia dataset
dataset = load_dataset("wikitext", 'wikitext-2-v1', split="train")
texts = dataset['text']

# Define your own tokenizer class
class XTokenizer:
    def __init__(self):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

    def fit_on_texts(self, texts):
        self.tokenizer.fit_on_texts(texts)

    def texts_to_sequences(self, texts):
        return self.tokenizer.texts_to_sequences(texts)

    def pad_sequences(self, sequences, maxlen=None, padding='pre', truncating='pre', value=0):
        return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding=padding, truncating=truncating, value=value)

    def sequences_to_texts(self, sequences):
        return self.tokenizer.sequences_to_texts(sequences)

# Initialize your custom tokenizer
tokenizer = XTokenizer()
tokenizer.fit_on_texts(texts)

# Calculate sequences after fitting the tokenizer
sequences = tokenizer.texts_to_sequences(texts)

# Print the number of tokens and vocabulary size
print("Number of tokens:", sum(len(seq) for seq in sequences))
print("Vocabulary size:", len(tokenizer.tokenizer.word_index) + 1)

max_length = 128  # Adjust as needed

# Model hyperparameters
vocab_size = len(tokenizer.tokenizer.word_index) + 1
d_model = 768
num_layers = 12
num_heads = 12
dff = 3072
dropout_rate = 0.1
batch_size = 12



# Define the MultiHeadAttention class
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention_logits = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        output = self.dense(output)
        return output, attention_weights

# Define the FeedForward class
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        super(FeedForward, self).__init__()
        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x

# Define the model architecture
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, look_ahead_mask=None):
        attn_output, _ = self.mha(x, x, x, look_ahead_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class ARCH_X(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, max_length, dropout=0.1):
        super(ARCH_X, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(max_length, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)]
        self.dense = tf.keras.layers.Dense(vocab_size)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.tokenizer = tokenizer

    def call(self, x, look_ahead_mask=None, training=False):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, training, look_ahead_mask)
        output = self.dense(x)
        return output

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates


    def generate_text(self, start_string, num_generate=100, temperature=1.0):
        input_eval = [self.tokenizer.tokenizer.word_index.get(s, 0) for s in start_string.split()]
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []

        for _ in range(num_generate):
            predictions = self(input_eval)
            predictions = predictions[:, -1, :]
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
            input_eval = tf.concat([input_eval, tf.expand_dims([predicted_id], 0)], axis=-1)
            text_generated.append(self.tokenizer.tokenizer.index_word.get(predicted_id, '<UNK>'))

        return start_string + ' ' + ' '.join(text_generated)

# Preprocess the text data
padded_sequences = tokenizer.pad_sequences(sequences, maxlen=max_length)


# Initialize the model
model = ARCH_X(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    vocab_size=vocab_size,
    max_length=max_length,
    dropout=dropout_rate
)

# Define loss function and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# Define accuracy metric
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

# Define the training step function
@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_object(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_accuracy.update_state(targets, predictions)
    return loss

# Define the number of epochs
num_epochs = 4

# Training loop with benchmarking
for epoch in range(num_epochs):
    start_time = time.time()
    num_batches = int(np.ceil(len(padded_sequences) / batch_size))
    with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
        for batch in range(num_batches):
            batch_start = batch * batch_size
            batch_end = min(batch_start + batch_size, len(padded_sequences))
            batch_inputs = padded_sequences[batch_start:batch_end]
            batch_targets = np.roll(batch_inputs, shift=-1, axis=1)
            loss = train_step(batch_inputs, batch_targets)
            pbar.set_postfix({"loss": f"{loss.numpy():.4f}", "accuracy": f"{train_accuracy.result().numpy():.4f}"})
            pbar.update(1)
    end_time = time.time()
    print(f"Epoch {epoch + 1}, Loss: {loss.numpy():.4f}, Accuracy: {train_accuracy.result().numpy():.4f}, Time: {end_time - start_time:.2f} seconds")


model.save("arch_x_model")
# Save only the weights of the model
model.save_weights("arch_x_model_weights.h5")


# Load the entire model
loaded_model = tf.keras.models.load_model("arch_x_model", custom_objects={"ARCH_X": ARCH_X, "DecoderLayer": DecoderLayer, "MultiHeadAttention": MultiHeadAttention, "FeedForward": FeedForward})

# Use the loaded model for text generation
start_string = "Nachiketh is good boy and"
generated_text = loaded_model.generate_text(start_string, num_generate=100, temperature=7.0)
print(generated_text)
