import cv2
import csv
import numpy as np
import os
import tensorflow as tf
import load_dataset
import matplotlib.pyplot as plt
import time


tf.enable_eager_execution()

EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_UNITS = 256


logdir = "./tensorboard_logs/attention_bi_uni_fc"
writer = tf.contrib.summary.create_file_writer(logdir)
with writer.as_default():
    tf.contrib.summary.always_record_summaries()
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
checkpoint_directory = './training_checkpoints/attention_bi_uni_fc'
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)


class Encoder(tf.keras.Model):
    def __init__(self, num_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.num_units = num_units
        self.gru_1_f = tf.keras.layers.GRU(self.num_units, return_sequences=True, recurrent_initializer='glorot_uniform', return_state=True, kernel_regularizer=regularizer)
        self.gru_1_b = tf.keras.layers.GRU(self.num_units, return_sequences=True, recurrent_initializer='glorot_uniform', return_state=True, go_backwards=True, kernel_regularizer=regularizer)
        self.state = tf.keras.layers.Dense(num_units, kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizer)

    def call(self, init_hidden, encoder_input):
        encoder_input = tf.keras.layers.Dropout(0.2)(encoder_input)
        predictions_f, hidden_f = self.gru_1_f(encoder_input, initial_state=init_hidden)
        predictions_b, hidden_b = self.gru_1_b(encoder_input, initial_state=init_hidden)
        predictions = tf.concat([predictions_f, predictions_b], -1)
        hidden = tf.concat([hidden_f, hidden_b], -1)
        state = self.state(hidden)
        return predictions, hidden, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.num_units))


class BahdanauAttention(tf.keras.Model):
    def __init__(self, hidden_units):
        super(BahdanauAttention, self).__init__()
        self.W = tf.keras.layers.Dense(hidden_units, kernel_regularizer=regularizer)
        self.U = tf.keras.layers.Dense(hidden_units, kernel_regularizer=regularizer)
        self.V = tf.keras.layers.Dense(1, kernel_regularizer=regularizer)

    def call(self, hidden, encoder_output):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W(encoder_output) + self.U(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, num_units, batch_size, projection_num_units):
        super().__init__()
        self.batch_size = batch_size
        self.num_units = num_units * 2
        self.projection_num_units = projection_num_units

        self.W   = tf.keras.layers.Dense(self.num_units, kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizer)
        self.W_s = tf.keras.layers.Dense(self.num_units, kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizer)
        self.W_z = tf.keras.layers.Dense(self.num_units, kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizer)
        self.W_r = tf.keras.layers.Dense(self.num_units, kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizer)
        self.C   = tf.keras.layers.Dense(self.num_units, kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizer)
        self.C_z = tf.keras.layers.Dense(self.num_units, kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizer)
        self.C_r = tf.keras.layers.Dense(self.num_units, kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizer)
        self.U   = tf.keras.layers.Dense(self.num_units, kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizer)
        self.U_z = tf.keras.layers.Dense(self.num_units, kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizer)
        self.U_r = tf.keras.layers.Dense(self.num_units, kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizer)
        self.s = None
        self.projection = tf.keras.layers.Dense(projection_num_units, activation='tanh')
        self.attention = BahdanauAttention(num_units)

    def call(self, x, decoder_hidden, encoder_output, init_hidden_state):
        if self.s is None:
            self.s = tf.math.tanh(self.W_s(init_hidden_state))
        x = tf.reshape(x, [x.shape[0], x.shape[2]])
        x = tf.keras.layers.Dropout(0.2)(x)
        context_vector, _ = self.attention(decoder_hidden, encoder_output)
        r_i = tf.math.sigmoid(self.W_r(x) + self.U_r(self.s) + self.C_r(context_vector))
        z_i = tf.math.sigmoid(self.W_z(x) + self.U_z(self.s) + self.C_z(context_vector))
        s_updated = tf.math.tanh(self.W(x) + self.U(tf.math.multiply(r_i, self.s)) + self.C(context_vector))
        self.s = tf.math.multiply((1-z_i), self.s) + tf.math.multiply(z_i, s_updated)
        projection = self.projection(self.s)
        return projection, self.s


def loss_function(real, pred):
    loss = tf.losses.mean_squared_error(real, pred)
    return loss


encoder = Encoder(NUM_UNITS, BATCH_SIZE)
decoder = Decoder(NUM_UNITS, BATCH_SIZE, 120)


def train_step_network(input, encoder_hidden):

    with tf.GradientTape() as tape:
        encoder_output, concatenated_encoder_hidden, last_hidden = encoder(encoder_hidden, input)
        decoder_hidden = concatenated_encoder_hidden

        decoder_input = tf.slice(input, [0, 0, 0], [input.shape[0], input.shape[1] - 1, input.shape[2]])
        zeros = tf.zeros([input.shape[0], 1, input.shape[2]])
        decoder_input = tf.concat([zeros, decoder_input], 1)

        reconstruction = None
        for i in range(0, input.shape[1]):
            input_slice = tf.slice(decoder_input, [0, i, 0], [decoder_input.shape[0], 1, decoder_input.shape[2]])
            projection, decoder_hidden = decoder(input_slice, decoder_hidden, encoder_output, last_hidden)
            if reconstruction is None:
                reconstruction = tf.expand_dims(projection, axis=1)
            else:
                reconstruction = tf.concat([reconstruction, tf.expand_dims(projection, axis=1)], axis=1)

        reverse_input = tf.reverse(input, [1])
        loss = loss_function(reverse_input, reconstruction) + tf.losses.get_regularization_loss()

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss


def train_network(epochs, batch_size, learning_rate, num_units, checkpoint_directory):
    global EPOCHS
    global BATCH_SIZE
    global LEARNING_RATE
    global NUM_UNITS
    global optimizer

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


if __name__ == "__main__":
    train_network(50, 64, 0.001, 256, './training_checkpoints/attention_bi_uni_fc')

