import cv2
import csv
import numpy as np
import os
import tensorflow as tf
import load_dataset
import matplotlib.pyplot as plt
import time


tf.enable_eager_execution()

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_UNITS = 256


logdir = "./tensorboard_logs/attention_bi_bi_hidden"
writer = tf.contrib.summary.create_file_writer(logdir)
with writer.as_default():
    tf.contrib.summary.always_record_summaries()
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
checkpoint_directory = './training_checkpoints/attention_bi_bi_hidden'
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
dataset = load_dataset.train_input_fn(BATCH_SIZE)


class Encoder(tf.keras.Model):
    def __init__(self, num_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.num_units = num_units
        self.gru_1_f = tf.keras.layers.GRU(self.num_units, return_sequences=True, recurrent_initializer='glorot_uniform', return_state=True)
        self.gru_1_b = tf.keras.layers.GRU(self.num_units, return_sequences=True, recurrent_initializer='glorot_uniform', return_state=True, go_backwards=True)
        self.gru_2_f = tf.keras.layers.GRU(self.num_units, return_sequences=True, recurrent_initializer='glorot_uniform', return_state=True)
        self.gru_2_b = tf.keras.layers.GRU(self.num_units, return_sequences=True, recurrent_initializer='glorot_uniform', return_state=True, go_backwards=True)

    def call(self, init_hidden, encoder_input):
        encoder_input = tf.keras.layers.Dropout(0.2)(encoder_input)
        predictions_f, _ = self.gru_1_f(encoder_input, initial_state=init_hidden)
        predictions_b, _ = self.gru_1_b(encoder_input, initial_state=init_hidden)
        predictions = tf.concat([predictions_f, predictions_b], -1)
        predictions_f, hidden_f = self.gru_2_f(predictions, initial_state=init_hidden)
        predictions_b, hidden_b = self.gru_2_b(predictions, initial_state=init_hidden)
        predictions = tf.concat([predictions_f, predictions_b], -1)
        hidden = tf.concat([hidden_f, hidden_b], -1)

        return predictions, hidden, [hidden_f, hidden_b]

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.num_units))


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, hidden, encoder_output):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(encoder_output) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class FullyConnectedLayer(tf.keras.Model):
    def __init__(self, num_units):
        super().__init__()
        self.dense = tf.keras.layers.Dense(num_units, activation='tanh')

    def call(self, input):
        x = self.dense(input)
        return x


class Decoder(tf.keras.Model):
    def __init__(self, num_units, batch_size, projection_num_units):
        super().__init__()
        self.batch_size = batch_size
        self.num_units = num_units
        self.projection_num_units = projection_num_units
        self.gru_1_f = tf.keras.layers.GRU(self.num_units, return_sequences=True, recurrent_initializer='glorot_uniform', return_state=True)
        self.gru_1_b = tf.keras.layers.GRU(self.num_units, return_sequences=True, recurrent_initializer='glorot_uniform', return_state=True, go_backwards=True)

        self.gru_2_f = tf.keras.layers.GRU(self.num_units, return_sequences=True, recurrent_initializer='glorot_uniform', return_state=True)
        self.gru_2_b = tf.keras.layers.GRU(self.num_units, return_sequences=True, recurrent_initializer='glorot_uniform', return_state=True, go_backwards=True)

        self.projection = FullyConnectedLayer(self.projection_num_units)
        self.attention = BahdanauAttention(self.num_units)

    def call(self, x, decoder_hidden, encoder_output, last_encoder_hidden):
        context_vector, attention_weights = self.attention(decoder_hidden, encoder_output)

        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        predictions_f, _ = self.gru_1_f(x, initial_state=last_encoder_hidden[0])
        predictions_b, _ = self.gru_1_b(x, initial_state=last_encoder_hidden[1])
        predictions = tf.concat([predictions_f, predictions_b], -1)

        predictions_f, hidden_f = self.gru_2_f(predictions)
        predictions_b, hidden_b = self.gru_2_b(predictions, initial_state=last_encoder_hidden[1])

        predictions = tf.concat([predictions_f, predictions_b], -1)
        hidden = tf.concat([hidden_f, hidden_b], axis=-1)
        projection = self.projection(tf.reshape(predictions, [predictions.shape[0], predictions.shape[2]]))
        return projection, hidden, context_vector


def loss_function(real, pred):
    loss = tf.losses.mean_squared_error(real, pred)
    return loss


encoder = Encoder(NUM_UNITS, BATCH_SIZE)
decoder = Decoder(NUM_UNITS, BATCH_SIZE, 120)


def train_step_network(input, encoder_hidden):

    with tf.GradientTape() as tape:
        encoder_output, concatenated_encoder_hidden, encoder_hidden = encoder(encoder_hidden, input)
        decoder_hidden = concatenated_encoder_hidden

        decoder_input = tf.slice(input, [0, 0, 0], [input.shape[0], input.shape[1] - 1, input.shape[2]])
        zeros = tf.zeros([input.shape[0], 1, input.shape[2]])
        decoder_input = tf.concat([zeros, decoder_input], 1)

        reconstruction = None
        for i in range(0, input.shape[1]):
            input_slice = tf.slice(decoder_input, [0, i, 0], [decoder_input.shape[0], 1, decoder_input.shape[2]])
            projection, decoder_hidden, _ = decoder(input_slice, decoder_hidden, encoder_output, encoder_hidden)
            if reconstruction is None:
                reconstruction = tf.expand_dims(projection, axis=1)
            else:
                reconstruction = tf.concat([reconstruction, tf.expand_dims(projection, axis=1)], axis=1)

        reverse_input = tf.reverse(input, [1])
        loss = loss_function(reverse_input, reconstruction)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss


def train_network(epochs, batch_size, learning_rate, num_units, checkpoint_directory):
    global EPOCHS
    global BATCH_SIZE
    global LEARNING_RATE
    global NUM_UNITS
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    LEARNING_RATE = learning_rate
    NUM_UNITS = num_units

    global_step = tf.train.create_global_step()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder, global_step=global_step)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
    with writer.as_default():
        tf.contrib.summary.initialize()
        begin_time = time.time()
        for epoch in range(1, EPOCHS + 1):
            dataset = load_dataset.train_input_fn(BATCH_SIZE)
            epoch_time = now_time = time.time()
            total_loss = 0
            num_step = 0
            for (batch, (input, filenames, labels)) in enumerate(dataset):
                if input['image'].shape[0] < BATCH_SIZE:
                    break
                encoder_hidden = encoder.initialize_hidden_state()
                input['image'] = (input['image'] / 255 * 2) - 1
                batch_loss = train_step_network(input['image'], encoder_hidden)
                total_loss += batch_loss
                num_step = batch + 1
                global_step.assign_add(1)
                with tf.contrib.summary.record_summaries_every_n_global_steps(1):
                    tf.contrib.summary.scalar('loss_attention', batch_loss)
                now_time = time.time()
                print('Epoch {} Step {} Loss {:.4f} Elapsed time {}'.format(epoch, num_step, batch_loss.numpy(), time.strftime("%H:%M:%S", time.gmtime(now_time - begin_time))))
            print('Epoch {} Loss {:.4f} Duration {}'.format(epoch, total_loss / num_step, time.strftime("%H:%M:%S", time.gmtime(now_time - epoch_time))))
            if epoch % 10 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)


def extract_reconstructed_spectrograms(dataset, spectrograms_directory, checkpoint_directory="./training_checkpoints/attention"):
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
    os.makedirs(os.path.dirname(spectrograms_directory), exist_ok=True)
    for input, filenames, labels in dataset:
        encoder_hidden = encoder.initialize_hidden_state()
        if input['image'].shape[0] < BATCH_SIZE:
            zeros = tf.zeros([BATCH_SIZE - input['image'].shape[0], input['image'].shape[1], input['image'].shape[2]])
            input['image'] = tf.concat([input['image'], zeros], 0)
        input['image'] = (input['image'] / 255 * 2) - 1
        encoder_ouput, decoder_hidden = encoder(encoder_hidden, input)
        decoder_input = tf.slice(input, [0, 0, 0], [input.shape[0], input.shape[1] - 1, input.shape[2]])
        zeros = tf.zeros([input.shape[0], 1, input.shape[2]])
        decoder_input = tf.concat([zeros, decoder_input], 1)
        reconstruction, _ = decoder(decoder_hidden, decoder_input, encoder_ouput)
        reconstruction = tf.reshape(reconstruction, [reconstruction.shape[0], reconstruction.shape[2],
                                                    reconstruction.shape[1]]).numpy()

        for i in range(labels.shape[0]):
            spectrogram = (((reconstruction[i] + 1) / 2) * 255).astype(np.uint8)
            filename = filenames['filename'].numpy()[i].decode("utf-8")
            cv2.imwrite(os.path.join(spectrograms_directory, filename), spectrogram)


def extract_features(dataset, features_path, checkpoint_directory="./training_checkpoints/attention_bi_bi_hidden"):
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    list = []
    for input, filenames, labels in dataset:
        encoder_hidden = encoder.initialize_hidden_state()
        if input['image'].shape[0] < BATCH_SIZE:
            zeros = tf.zeros([BATCH_SIZE - input['image'].shape[0], input['image'].shape[1], input['image'].shape[2]])
            input['image'] = tf.concat([input['image'], zeros], 0)
        input = (input['image'] / 255 * 2) - 1
        _, decoder_hidden, _ = encoder(encoder_hidden, input)
        for i in range(labels.shape[0]):
            filename = filenames['filename'].numpy()[i].decode("utf-8")
            feature = [filename] + decoder_hidden.numpy()[i].tolist()
            list += [feature]
    with open(features_path, 'w', newline='') as features:
        list = sorted(list)
        writer = csv.writer(features, delimiter=',')
        for line in list:
            writer.writerow(line)


def extract_all_features(features_directory, features_name, checkpoint_directory=checkpoint_directory):
    extract_features(load_dataset.train_input_fn(BATCH_SIZE), features_directory + '/' + features_name + ".train.csv", checkpoint_directory=checkpoint_directory)
    extract_features(load_dataset.test_input_fn(BATCH_SIZE), features_directory + '/' + features_name + ".test.csv", checkpoint_directory=checkpoint_directory)
    extract_features(load_dataset.val_input_fn(BATCH_SIZE), features_directory + '/' + features_name + ".devel.csv", checkpoint_directory=checkpoint_directory)


def extract_all_reconstructed_spectrograms(spectrograms_directory, checkpoint_directory=checkpoint_directory):
    extract_reconstructed_spectrograms(load_dataset.train_input_fn(BATCH_SIZE), spectrograms_directory + "/train", checkpoint_directory=checkpoint_directory)
    extract_reconstructed_spectrograms(load_dataset.train_input_fn(BATCH_SIZE), spectrograms_directory + "/test", checkpoint_directory=checkpoint_directory)
    extract_reconstructed_spectrograms(load_dataset.train_input_fn(BATCH_SIZE), spectrograms_directory + "/devel", checkpoint_directory=checkpoint_directory)

if __name__ == "__main__":
    train_network(10, 64, 0.001, 256, './training_checkpoints/attention_bi_bi_hidden')

