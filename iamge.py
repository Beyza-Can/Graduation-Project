import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Veri yolları
train_path_str = 'image-Data/train'
val_path_str = 'image-Data/valid'
test_path_str = 'image-Data/test'

input_shape = (224, 224, 3)
num_classes = 4

# Veri jeneratörleri
trainGenerator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    dtype='float32'
)

valGenerator = ImageDataGenerator(rescale=1./255)
testGenerator = ImageDataGenerator(rescale=1./255)

# Veri yükleme
train_data = trainGenerator.flow_from_directory(
    train_path_str,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

val_data = valGenerator.flow_from_directory(
    val_path_str,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

test_data = testGenerator.flow_from_directory(
    test_path_str,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

# Önceden eğitilmiş VGG16 modelini yükleme
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# VGG16 modelinin üzerine sınıflandırma katmanlarını ekleme
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dense(num_classes, activation='softmax')(x)

# Yeni modeli oluşturma
model = Model(inputs=base_model.input, outputs=x)

# VGG16 modelinin üst katmanlarını eğitmeme (transfer learning)
for layer in base_model.layers:
    layer.trainable = False

# Modelin derlenmesi
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model özeti
model.summary()

# Model eğitimi
results = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ModelCheckpoint('model_vgg16.keras', save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='logs_vgg16')
    ]
)

# Model değerlendirme ve görselleştirme
predictions_prob = model.predict(test_data)
predictions = np.argmax(predictions_prob, axis=1)
true_label = test_data.classes

report = classification_report(true_label, predictions)
print(report)

conf_mat = confusion_matrix(true_label, predictions)
sns.heatmap(conf_mat, annot=True, fmt='g', cmap='Blues', xticklabels=test_data.class_indices.keys(), yticklabels=test_data.class_indices.keys())
plt.xlabel('Predictions')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.show()

# Eğitim ve doğrulama kaybı
plt.plot(results.history['loss'], label='Training loss')
plt.plot(results.history['val_loss'], label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Eğitim ve doğrulama doğruluğu
plt.plot(results.history['accuracy'], label='Training accuracy')
plt.plot(results.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
