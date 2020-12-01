import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.constraints import MaxNorm

from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization, PReLU
from tensorflow.keras.layers import Input, MaxPooling2D, Reshape
from tensorflow.keras.layers import Multiply, Activation


class FullGatedConv2D(Conv2D):

    def __init__(self, filters, **kwargs):
        super(FullGatedConv2D, self).__init__(filters=filters * 2, **kwargs)
        self.nb_filters = filters

    def call(self, inputs):
        output = super(FullGatedConv2D, self).call(inputs)
        linear = Activation("linear")(output[:, :, :, :self.nb_filters])
        sigmoid = Activation("sigmoid")(output[:, :, :, self.nb_filters:])
        return Multiply()([linear, sigmoid])

    def compute_output_shape(self, input_shape):
        output_shape = super(FullGatedConv2D, self).compute_output_shape(input_shape)
        return tuple(output_shape[:3]) + (self.nb_filters,)

    def get_config(self):
        config = super(FullGatedConv2D, self).get_config()
        config['nb_filters'] = self.nb_filters
        del config['filters']
        return config


class myModel:

    def __init__(self,
                 input_size,
                 vocab_size,
                 greedy=False,
                 beam_width=10,
                 top_paths=1):

        self.input_size = input_size
        self.vocab_size = vocab_size

        self.model = None
        self.greedy = greedy
        self.beam_width = beam_width
        self.top_paths = max(1, top_paths)


    def load_checkpoint(self, target):
        if os.path.isfile(target):
            if self.model is None:
                self.compile()
            self.model.load_weights(target)

    def model_01(self, input_size, d_model):
        input_data = Input(name="input", shape=input_size)

        cnn = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer="he_uniform")(input_data)
        cnn = PReLU(shared_axes=[1, 2])(cnn)
        cnn = BatchNormalization(renorm=True)(cnn)
        cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn)

        cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
        cnn = PReLU(shared_axes=[1, 2])(cnn)
        cnn = BatchNormalization(renorm=True)(cnn)
        cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)

        cnn = Conv2D(filters=40, kernel_size=(2, 4), strides=(2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
        cnn = PReLU(shared_axes=[1, 2])(cnn)
        cnn = BatchNormalization(renorm=True)(cnn)
        cnn = FullGatedConv2D(filters=40, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
        cnn = Dropout(rate=0.2)(cnn)

        cnn = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
        cnn = PReLU(shared_axes=[1, 2])(cnn)
        cnn = BatchNormalization(renorm=True)(cnn)
        cnn = FullGatedConv2D(filters=48, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
        cnn = Dropout(rate=0.2)(cnn)

        cnn = Conv2D(filters=56, kernel_size=(2, 4), strides=(2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
        cnn = PReLU(shared_axes=[1, 2])(cnn)
        cnn = BatchNormalization(renorm=True)(cnn)
        cnn = FullGatedConv2D(filters=56, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
        cnn = Dropout(rate=0.2)(cnn)

        cnn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
        cnn = PReLU(shared_axes=[1, 2])(cnn)
        cnn = BatchNormalization(renorm=True)(cnn)

        cnn = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)

        shape = cnn.get_shape()
        bgru = Reshape((shape[1], shape[2] * shape[3]))(cnn)

        bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
        bgru = Dense(units=256)(bgru)

        bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
        output_data = Dense(units=d_model, activation="softmax")(bgru)

        return (input_data, output_data)



    def compile(self, learning_rate=None, initial_step=0):
        inputs, outputs = self.model_01(self.input_size, self.vocab_size + 1)
        if learning_rate is None:
            learning_rate = CustomSchedule(d_model=self.vocab_size + 1, initial_step=initial_step)
            self.learning_schedule = True
        else:
            self.learning_schedule = False
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=optimizer, loss=self.ctc_loss)


    def predict(self,
                x,
                batch_size=None,
                steps=1,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False,
                ctc_decode=True):

        out = self.model.predict(x=x, batch_size=batch_size, verbose=0, steps=steps,
                                 callbacks=callbacks, max_queue_size=max_queue_size,
                                 workers=workers, use_multiprocessing=use_multiprocessing)

        if not ctc_decode:
            return np.log(out.clip(min=1e-8)), []

        steps_done = 0

        batch_size = int(np.ceil(len(out) / steps))
        input_length = len(max(out, key=len))

        predicts, probabilities = [], []

        while steps_done < steps:
            index = steps_done * batch_size
            until = index + batch_size

            x_test = np.asarray(out[index:until])
            x_test_len = np.asarray([input_length for _ in range(len(x_test))])

            decode, log = K.ctc_decode(x_test,
                                       x_test_len,
                                       greedy=self.greedy,
                                       beam_width=self.beam_width,
                                       top_paths=self.top_paths)

            probabilities.extend([np.exp(x) for x in log])
            decode = [[[int(p) for p in x if p != -1] for x in y] for y in decode]
            predicts.extend(np.swapaxes(decode, 0, 1))

            steps_done += 1

        return (predicts, probabilities)


    def ctc_loss(self, y_true, y_pred):
        if len(y_true.shape) > 2:
            y_true = tf.squeeze(y_true)
        input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
        input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)
        label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64")
        loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        loss = tf.reduce_mean(loss)
        return loss


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, initial_step=0, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, dtype="float32")
        self.initial_step = initial_step
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step + self.initial_step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


