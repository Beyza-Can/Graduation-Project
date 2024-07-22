from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Veri yükleme için yollar
train_dir = 'image-Data/train'
valid_dir = 'image-Data/valid'
test_dir = 'image-Data/test'

# Modelinizin beklediği hedef boyut
target_size = (299, 299)

# ImageDataGenerator'ı oluşturma ve veri arttırma işlemi
datagen = ImageDataGenerator(rescale=1./255)

# Eğitim verilerini yükleme
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=32,
    class_mode='categorical'
)

# Doğrulama verilerini yükleme
valid_generator = datagen.flow_from_directory(
    valid_dir,
    target_size=target_size,
    batch_size=32,
    class_mode='categorical'
)

# Test verilerini yükleme
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Test verilerini karıştırmadan yükleme
)

# Test verilerini ve etiketlerini alma
X_test, y_true = [], []
for i in range(len(test_generator)):
    x, y = test_generator[i]
    X_test.extend(x)
    y_true.extend(y)

X_test = np.array(X_test)
y_true = np.array(y_true)

# Modelinizi yükleyin (modelinizi eğittiğiniz dosyada model.save() fonksiyonunu kullandıysanız)
model = load_model('final_model_xception_finetuned.keras')

# Model tahminlerini yapma
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_true, axis=1)

# Accuracy hesaplama
accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"Test Accuracy: {accuracy:.4f}")

# Karışıklık matrisi oluşturma
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Etiketler
class_labels = ['adenocarcinomia', 'normal', 'large.cell.carcinomia', 'squamous.cell.carcinomia']

# Karışıklık matrisini görselleştirme
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
