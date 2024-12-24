import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
import numpy as np
import glob
import cv2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from keras.callbacks import History, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import json
import os

# Kiểm tra và thiết lập GPU
print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))

if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)

# Sử dụng History callback để lưu lịch sử của quá trình huấn luyện
history = History()

# Load dữ liệu từ các thư mục train và validation
train_data_folder = 'data/train/'
train_folder_list = ['OK', 'XXX']
validation_data_folder = 'data/validation/'
validation_folder_list = ['OK', 'XXX']

# Tạo hai danh sách riêng biệt cho dữ liệu train và validation
train_data = []
train_label = []
validation_data = []
validation_label = []

# Load dữ liệu từ thư mục train
for folder in train_folder_list:
    for file in glob.glob(train_data_folder + folder + "/*"):
        img = cv2.imread(file)
        if img is not None:
            img = cv2.resize(img, dsize=(200, 200))
            train_data.append(img)
            train_label.append(folder)
        else:
            print("Không thể đọc được hình ảnh:", file)

# Load dữ liệu từ thư mục validation
for folder in validation_folder_list:
    for file in glob.glob(validation_data_folder + folder + "/*"):
        img = cv2.imread(file)
        if img is not None:
            img = cv2.resize(img, dsize=(200, 200))
            validation_data.append(img)
            validation_label.append(folder)
        else:
            print("Không thể đọc được hình ảnh:", file)

# Chuyển nhãn về dạng số
label_encoder = LabelEncoder()
integer_encoded_train = label_encoder.fit_transform(train_label)
integer_encoded_validation = label_encoder.transform(validation_label)

# Chuyển đổi nhãn thành one-hot encoding
onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded_train = integer_encoded_train.reshape(len(integer_encoded_train), 1)
integer_encoded_validation = integer_encoded_validation.reshape(len(integer_encoded_validation), 1)
onehot_encoded_train = onehot_encoder.fit_transform(integer_encoded_train)
onehot_encoded_validation = onehot_encoder.transform(integer_encoded_validation)

# Chuyển sang numpy array
train_label = onehot_encoded_train
validation_label = onehot_encoded_validation
train_data = np.array(train_data)
validation_data = np.array(validation_data)

# Sử dụng StratifiedKFold với k=15 folds trên dữ liệu train
skf = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)

# Danh sách để lưu các độ đo
accuracy_per_fold = []
recall_per_fold = []
precision_per_fold = []
f1_per_fold = []
confusion_matrices = []

# Danh sách để lưu lịch sử huấn luyện của từng fold
histories = []

# Sử dụng VGG16
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

# Đóng băng các layer
for layer in model_vgg16_conv.layers:
    layer.trainable = False

# Tạo model
input = Input(shape=(200, 200, 3), name='image_input')
output_vgg16_conv = model_vgg16_conv(input)
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax', name='predictions')(x)

# Compile
my_model = Model(inputs=input, outputs=predictions)
my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Lặp qua các folds
for fold_idx, (train_index, test_index) in enumerate(skf.split(train_data, train_label.argmax(axis=1))):
    print(f"Training Fold {fold_idx + 1}")
    X_train, X_val = train_data[train_index], train_data[test_index]
    y_train, y_val = train_label[train_index], train_label[test_index]

    # Fit model trên dữ liệu train của fold hiện tại, sử dụng History callback
    hist = my_model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=20, validation_data=(X_val, y_val), verbose=1, callbacks=[history, reduce_lr])

    # Lưu lịch sử huấn luyện của fold này
    histories.append(hist.history.copy())

    # Đánh giá mô hình trên dữ liệu validation và lưu các độ đo của fold này
    y_pred = my_model.predict(X_val)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_val_labels = np.argmax(y_val, axis=1)

    accuracy = accuracy_score(y_val_labels, y_pred_labels)
    recall = recall_score(y_val_labels, y_pred_labels)
    precision = precision_score(y_val_labels, y_pred_labels)
    f1 = f1_score(y_val_labels, y_pred_labels)
    cm = confusion_matrix(y_val_labels, y_pred_labels)

    accuracy_per_fold.append(accuracy)
    recall_per_fold.append(recall)
    precision_per_fold.append(precision)
    f1_per_fold.append(f1)
    confusion_matrices.append(cm)

# Convert numpy arrays to lists before saving
for i in range(len(confusion_matrices)):
    confusion_matrices[i] = confusion_matrices[i].tolist()

# Lưu lịch sử huấn luyện vào file
with open('train_history.json', 'w') as f:
    json.dump(histories, f)

# Lưu các độ đo vào file JSON
metrics = {
    "accuracy_per_fold": accuracy_per_fold,
    "recall_per_fold": recall_per_fold,
    "precision_per_fold": precision_per_fold,
    "f1_per_fold": f1_per_fold,
    "confusion_matrices": confusion_matrices
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

# Lưu mô hình với tên "cool_model"
my_model.save("cool_model")
print("Saved model!")

# In ra thông báo khi hoàn thành
print("Finish model!")
