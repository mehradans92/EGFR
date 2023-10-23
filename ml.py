from types import SimpleNamespace
import tensorflow as tf
import json


class RNNModel(tf.keras.Model):
    def __init__(self, config, input_shapes, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        """Initializes the RNN model.
        """
        if isinstance(config, dict):
            self.config = SimpleNamespace(**config)
        else:
            self.config = config
        self.fingerprint_shape = None
        self.descriptor_shape = None
        self.encoding_shape = input_shapes[0]
        self.input_shapes = input_shapes
        if len(self.input_shapes) > 1:
            self.fingerprint_shape = input_shapes[1]
        if len(self.input_shapes) > 2:
            self.descriptor_shape = input_shapes[2]
        self.model = self._build_LSTM_model()

    def _build_LSTM_model(self):
        """Builds the RNN model architecture with Embedding, Bidirectional LSTM, Dense and Output layers."""
        tf.keras.backend.clear_session()
        # Inputs
        encoding_input = tf.keras.Input(shape=(self.encoding_shape,))

        inputs = [encoding_input]

        # Encoding Layer
        e = tf.keras.layers.Embedding(
            input_dim=self.config.vocab_size,
            output_dim=self.config.embedding_dim,
            mask_zero=True,
        )(encoding_input)

        # Bidirectional LSTM Layers
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.config.rnn_units, return_sequences=True)
        )(e)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.config.rnn_units, activation=None)
        )(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(self.config.drop_rate)(x)

        if self.fingerprint_shape:
            # Concatenate fingerprint input
            fingerprint_input = tf.keras.Input(shape=(self.fingerprint_shape,))
            x = tf.keras.layers.Concatenate()([x, fingerprint_input])
            inputs.append(fingerprint_input)

        # Dense Layers
        x = tf.keras.layers.Dense(
            self.config.hidden_dim,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.reg_strength),
        )(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(self.config.drop_rate)(x)

        if self.descriptor_shape:
            # Concatenate descriptor_shape input
            descriptor_input = tf.keras.Input(shape=(self.descriptor_shape,))
            x = tf.keras.layers.Concatenate()([x, descriptor_input])
            inputs.append(descriptor_input)

        x = tf.keras.layers.Dense(
            self.config.hidden_dim // 4,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.reg_strength),
        )(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(self.config.drop_rate)(x)

        # Output Layer
        output = tf.keras.layers.Dense(1, activation="sigmoid", bias_initializer=None)(
            x
        )

        return tf.keras.Model(inputs=inputs, outputs=output, name="LSTM")

    def compile(self):
        """Compiles the model with ADAM optimizer, binary cross-entropy loss, and accuracy, AUC and F1-score metrics."""
        opt = tf.optimizers.Adam(self.config.lr)
        self.model.compile(
            opt,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(threshold=0.5),
                tf.keras.metrics.AUC(from_logits=False),
                tf.keras.metrics.F1Score(threshold=0.5),
            ],
        )

    def train(self, X_train_input_list, y_train, validation_data=None, verbose=1):
        """Trains the RNN model with reduced LR on plateau and early stopping based on maximizing validation data AUC.
        If validation data is not provided, the training is done for for the number of epochs defined in config.
        """
        tf.keras.backend.clear_session()
        self.compile()
        callbacks = []

        if validation_data:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc", factor=0.9, patience=8, min_lr=0.00001
            )
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                restore_best_weights=True,
                patience=self.config.early_stopping_patience,
            )
            callbacks.append([reduce_lr, early_stop])

        self.result = self.model.fit(
            X_train_input_list,
            y_train,
            validation_data=validation_data,
            callbacks=callbacks,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=verbose,
        )

        return self.result

    def evaluate(self, x, y, **kwargs):
        """Evaluate the model on the test data."""
        return self.model.evaluate(x, y, **kwargs)

    def summary(self):
        """Prints a summary the architecture and keras model paramteres."""
        self.model.summary()

    def get_config(self):
        return {
            "config": self.config,
            "input_shapes": self.input_shapes,
            "output_bias": self.output_bias,
        }

    def save(self, weights_path, config_path):
        # Saving the weights
        self.model.save_weights(weights_path)

        # Saving the config and other attributes
        attributes = {
            "config": self.config.__dict__,  # assuming self.config is an object with attributes
            "input_shapes": self.input_shapes,
        }
        with open(config_path, "w") as f:
            json.dump(attributes, f, indent=4)

    def load(weights_path, config_path):
        # Load the config
        with open(config_path, "r") as f:
            attributes = json.load(f)

        loaded_model = RNNModel(
            config=attributes["config"], input_shapes=attributes["input_shapes"]
        )

        # Load the weights
        loaded_model.model.load_weights(weights_path)

        return loaded_model
