# Skript för Träning och Exportering av Maskininlärningsmodeller

Detta skript är designat för att träna maskininlärningsmodeller för uppgifter inom bildklassificering eller objektdetektering, samt exportera de tränade modellerna till TensorFlow Lite och Core ML-format för användning på Android- och iOS-enheter.

## Förberedelser

### Systemkrav:
- Python 3.x
- TensorFlow
- Keras
- CoreMLTools
- Keras Tuner

### Installation: 
```bash
pip install tensorflow keras coremltools keras-tuner


## Mappstruktur

Organisera din mappstruktur enligt följande:

my_ml_projects/
    data/
        image_classification/ eller object_detection/
            train/ (träningsbilder)
            validate/ (valideringsbilder)
    logs/
    models



## Steg-för-steg-anvisningar

### 1. Förbered Din Data:
   - Organisera din data i en `train` och `validate` mapp under den respektive uppgiftsmappen (`image_classification` eller `object_detection`).
   - Bilderna ska grupperas i undermappar per klass i både `train` och `validate` mapparna.

### 2. Konfigurera Skriptet:
   - Modifiera `PROJECT_ROOT` variabeln i skriptet för att peka på din `my_ml_projects` mapp.
   - Ställ in `TASK` variabeln till antingen `image_classification` eller `object_detection` beroende på uppgiften du vill utföra.

### 3. Kör Skriptet:
   - Öppna en terminal och navigera till mappen som innehåller skriptet.
   - Kör skriptet med kommandot: 
   ```bash
   python script_name.py



## Modellträning:

   - Skriptet startar automatiskt träningen av modellen med den tillhandahållna datan.
   - Träningsloggar, den bästa modellen och den slutliga modellen sparas i logs, models/{TASK}/best_model.h5 och models/{TASK}/saved_model.h5 mapparna respektive.

## Exportera Modellen:

   - Efter träningen försöker skriptet exportera modellen till TensorFlow Lite och Core ML format.
   - De exporterade modellerna sparas i models/{TASK}/model.tflite och models/{TASK}/model.mlmodel mapparna respektive.

## Övervaka Träningen:

- Du kan övervaka träningsprocessen med TensorBoard. Kör följande kommando för att starta TensorBoard: tensorboard --logdir C:/Users/Anton/my_ml_projects/logs/fit/{TASK}

## Anpassa Träningen:

- Du kan anpassa träningsprocessen genom att modifiera EPOCHS, MAX_TRIALS och andra variabler i skriptet.
- Du kan också anpassa build_model funktionen för att definiera din egen modellarkitektur.


## Felsökning

 - Om du stöter på importfel, se till att alla nödvändiga bibliotek är installerade och att Python-miljön är korrekt konfigurerad.
  - Kontrollera logs mappen för loggfiler som kan ge mer information om eventuella fel som uppstår
