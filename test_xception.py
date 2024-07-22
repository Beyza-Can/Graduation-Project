from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
 

# Veri yolları
train_folder = "image-Data/train"
valid_folder = "image-Data/valid"
test_folder = "image-Data/test"

input_shape = (299, 299, 3)
num_classes = 4

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical'
)
validation_generator = val_datagen.flow_from_directory(
    valid_folder,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_folder,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Modelin temelini oluşturma
base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)

# Bazı katmanları yeniden eğitilebilir hale getirme
for layer in base_model.layers[:-10]:  # Son 10 katmanı yeniden eğitilebilir yapma
    layer.trainable = False

# Yeni üst katmanlar ekleme
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Modeli derleme
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,
    callbacks=[
        ModelCheckpoint('best_model_xception_finetuned.keras', monitor='val_accuracy', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
    ]
)

# Modeli değerlendirme
test_loss, test_accuracy = model.evaluate(test_generator)
print('Test accuracy:', test_accuracy)

# Modeli kaydetme
model.save('final_model_xception_finetuned.keras')
