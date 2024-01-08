import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import (
    Dense,
    LSTM,
    Input,
    concatenate,
    Dropout,
    BatchNormalization,
)
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import keras.backend as K


# from input_data import train_X1, test_X1, train_X2, test_X2, test_Y, train_Y
# from simple_data_creation import train_x, train_y, test_x, test_y
from data_creation import train_x1, train_x3, train_y, test_x1, test_x3, test_y, NAME


lsmt_model = Sequential(
    [
        Input(shape=train_x1.shape[1:]),
        LSTM(150, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(150, return_sequences=True),
        Dropout(0.1),
        BatchNormalization(),
        LSTM(150),
        Dropout(0.2),
        BatchNormalization(),
    ]
)


# news_model = Sequential(
#     [
#         Input(shape=tar.shape[1:]),
#         LSTM(32, return_sequences=True),
#         Dropout(0.2),
#         BatchNormalization(),
#         LSTM(32),
#         Dropout(0.2),
#         BatchNormalization(),
#     ]
# )


other_features_model = Sequential([Input(shape=train_x3.shape[1:]), Dense(10)])


concatenated_output = concatenate([lsmt_model.output, other_features_model.output])

final_layers = [
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(2, activation="softmax"),
]

output_tensor = concatenated_output
for layer in final_layers:
    output_tensor = layer(output_tensor)


model = Model(
    inputs=[lsmt_model.input, other_features_model.input], outputs=output_tensor
)


print(model.summary())

with open(f"model_structure/{NAME}.json", "w+") as f:
    f.write(model.to_json())


def myprint(s):
    with open(f"model_structure/{NAME}.txt", "a+") as f:
        print(s, file=f)


model.summary(print_fn=myprint)


tensorboard_callback = TensorBoard(
    log_dir=f"logs/{NAME}",
    histogram_freq=1,
)

# Create a ModelCheckpoint callback to save the best weights during training
checkpoint_callback = ModelCheckpoint(
    filepath=f"checkpoints/{NAME}.model.keras",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)

early_stopping = EarlyStopping(
    patience=3, monitor="val_loss", mode="auto", restore_best_weights=True
)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", precision_m, f1_m],
)


# Train the model
model.fit(
    [train_x1, train_x3],
    train_y,
    epochs=40,
    validation_data=([test_x1, test_x3], test_y),
    callbacks=[tensorboard_callback, checkpoint_callback, early_stopping],
    batch_size=64,
)

model.save(f"models/{NAME}.keras")

print(f"Finished training model {NAME}. Model saved.")


# # Predict on new data (adjust 'new_data' accordingly)
# predicted_change_normalized = model.predict(new_data)
# predicted_change = scaler.inverse_transform(predicted_change_normalized)
# print(f"Predicted Change: {predicted_change[0][0]}")
