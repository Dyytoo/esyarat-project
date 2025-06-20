import tensorflow as tf
from tensorflow.keras import layers, Model

class SignLanguageModel:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = self._build_model()

    # def _build_model(self):
    #     """
    #     Build a CNN model for sign language detection
    #     """
    #     # Input layer for hand landmarks (21 points, 3 coordinates each)
    #     inputs = layers.Input(shape=(21, 3))
        
    #     # Reshape for CNN
    #     x = layers.Reshape((21, 3, 1))(inputs)
        
    #     # Convolutional layers
    #     x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    #     x = layers.MaxPooling2D((2, 2))(x)
    #     x = layers.BatchNormalization()(x)
        
    #     x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    #     x = layers.MaxPooling2D((2, 2))(x)
    #     x = layers.BatchNormalization()(x)
        
    #     x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    #     x = layers.MaxPooling2D((2, 2))(x)
    #     x = layers.BatchNormalization()(x)
        
    #     # Flatten and Dense layers
    #     x = layers.Flatten()(x)
    #     x = layers.Dense(256, activation='relu')(x)
    #     x = layers.Dropout(0.5)(x)
    #     x = layers.Dense(128, activation='relu')(x)
    #     x = layers.Dropout(0.3)(x)
        
    #     # Output layer
    #     outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
    #     # Create model
    #     model = Model(inputs=inputs, outputs=outputs)
        
    #     # Compile model
    #     model.compile(
    #         optimizer='adam',
    #         loss='categorical_crossentropy',
    #         metrics=['accuracy']
    #     )
        
    #     return model

    def _build_model(self):
        """
        Build a model for sign language detection with two hands (42, 3)
        """
        inputs = layers.Input(shape=(42, 3))
        x = layers.Flatten()(inputs)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, class_weight=None):
        """
        Train the model
        """
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        return history

    def predict(self, landmarks):
        """
        Make predictions on new data
        """
        return self.model.predict(landmarks)

    def save_model(self, path):
        """
        Save the model
        """
        self.model.save(path)

    def load_model(self, path):
        """
        Load a saved model
        """
        self.model = tf.keras.models.load_model(path) 