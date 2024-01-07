import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, concatenate, Dropout, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint


# from input_data import train_X1, test_X1, train_X2, test_X2, test_Y, train_Y
# from simple_data_creation import train_x, train_y, test_x, test_y
from data_creation import train_x1, train_x3, train_y, test_x1, test_x3, test_y, NAME


lsmt_model = Sequential(
    [
        Input(shape=train_x1.shape[1:]),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(128, return_sequences=True),
        Dropout(0.1),
        BatchNormalization(),
        LSTM(128),
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


other_features_model = Sequential([Input(shape=train_x3.shape[1:])])


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


tensorboard_callback = TensorBoard(log_dir=f"logs/{NAME}", histogram_freq=1)

# # Create a ModelCheckpoint callback to save the best weights during training
# checkpoint_callback = ModelCheckpoint(
#     filepath=r"checkpoints",
#     save_weights_only=True,
#     monitor="val_accuracy",
#     mode="max",
#     save_best_only=True,
# )

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train the model
model.fit(
    [train_x1, train_x3],
    train_y,
    epochs=11,
    validation_data=([test_x1, test_x3], test_y),
    callbacks=[tensorboard_callback],
    batch_size=64,
)


# # Predict on new data (adjust 'new_data' accordingly)
# predicted_change_normalized = model.predict(new_data)
# predicted_change = scaler.inverse_transform(predicted_change_normalized)
# print(f"Predicted Change: {predicted_change[0][0]}")
