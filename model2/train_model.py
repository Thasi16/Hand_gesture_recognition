import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from config import NUM_CLASSES, MAX_FRAMES, MODEL_PATH

def main():
    print(" Đang tải dữ liệu từ ổ cứng...")
    X = np.load("X_data.npy")
    y = np.load("y_data.npy")

    y = to_categorical(y, NUM_CLASSES)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Dữ liệu Train:", X_train.shape, y_train.shape)
    print("Dữ liệu Test :", X_test.shape, y_test.shape)

    # Build model
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(MAX_FRAMES, 126)),
        Dropout(0.3),
        LSTM(128),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Huấn luyện
    model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

    model.save(MODEL_PATH)
    print(f" Đã lưu model tại: {MODEL_PATH}")

if __name__ == "__main__":
    main()
