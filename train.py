import os
import sys
import logging
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import coremltools
import keras_tuner as kt


# Konfigurera loggning
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konstanter
EPOCHS = 5
MAX_TRIALS = 5
PROJECT_ROOT = "C:/Users/Anton/my_ml_projects"
TASK = "image_classification"  # eller "object_detection", ange detta enligt önskemål

# Sökvägar
LOG_DIR = f"{PROJECT_ROOT}/logs/fit/{TASK}"
BEST_MODEL_PATH = f'{PROJECT_ROOT}/models/{TASK}/best_model.h5'
SAVED_MODEL_PATH = f'{PROJECT_ROOT}/models/{TASK}/saved_model.h5'
TFLITE_MODEL_PATH = f'{PROJECT_ROOT}/models/{TASK}/model.tflite'
COREML_MODEL_PATH = f'{PROJECT_ROOT}/models/{TASK}/model.mlmodel'


# Uppgiftskonfigurationer
TASK_CONFIGS = {
    "image_classification": {
        "input_shape": (28, 28, 1),
    },
    "object_detection": {
        "input_shape": (128, 128, 3),
    }
    # Lägg till fler uppgifter ...
}


def load_data() -> tuple:
    """Laddar in och förbehandlar data från angiven katalogstruktur."""
    data_dir = f"{PROJECT_ROOT}/data/{TASK}"
    train_dir = os.path.join(data_dir, 'train')
    validation_dir = os.path.join(data_dir, 'validate')
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1,
                                       height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,
                                       horizontal_flip=True, fill_mode='nearest')
    validation_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=TASK_CONFIGS[TASK]["input_shape"][:2], batch_size=32, class_mode='categorical')
    validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=TASK_CONFIGS[TASK]["input_shape"][:2], batch_size=32, class_mode='categorical')
    return train_generator, validation_generator



def build_model(hp, num_classes: int) -> tf.keras.Model:
    """Bygger en sekventiell modell med variabla hyperparametrar."""
    input_shape = TASK_CONFIGS[TASK]["input_shape"]
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model



def main():
    """Huvudfunktion för att styra tränings- och exportprocessen."""
    logger.info(f"Träning för uppgift: {TASK}")
    logger.info(f"Loggar till: {LOG_DIR}")
    logger.info(f"Bästa modellen sparas till: {BEST_MODEL_PATH}")

    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    train_gen, val_gen = load_data()
    num_classes = len(train_gen.class_indices)

    tuner = kt.RandomSearch(lambda hp: build_model(hp, num_classes), objective='val_accuracy', max_trials=MAX_TRIALS)
    tuner.search(train_gen, epochs=EPOCHS, validation_data=val_gen)

    model = tuner.get_best_models(num_models=1)[0]

    callbacks = [
        TensorBoard(log_dir=LOG_DIR, histogram_freq=1),
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True)
    ]
    
    model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=callbacks)
    model.save(SAVED_MODEL_PATH)
    
    export_model(model)


def export_model(model):
    """Exporterar modellen till TensorFlow Lite och Core ML."""
    # Exportera till TensorFlow Lite (för Android)
    os.makedirs(os.path.dirname(TFLITE_MODEL_PATH), exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    
    # Försök exportera till Core ML (för iOS)
    os.makedirs(os.path.dirname(COREML_MODEL_PATH), exist_ok=True)
    try:
        coreml_model = coremltools.converters.convert(SAVED_MODEL_PATH)
        coreml_model.save(COREML_MODEL_PATH)
    except Exception as e:
        logger.error(f"Fel: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        TASK = sys.argv[1]
    main()
