import os
from pathlib import Path

import cv2
import numpy as np


def load_env(env_path: Path) -> None:
    """Carga variables de entorno desde un archivo .env si existe."""
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue

        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip())


BASE_DIR = Path(__file__).resolve().parent
load_env(BASE_DIR / '.env')

# Ruta base donde se encuentran los rostros capturados
dataPath = os.environ.get('DATA_DIR', str(BASE_DIR / 'data'))

# Lista de carpetas (personas) en el directorio de datos
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

# Recopila los rostros y sus etiquetas para entrenamiento
for nameDir in peopleList:
    personPath = os.path.join(dataPath, nameDir)
    print('Leyendo las imágenes')

    for fileName in os.listdir(personPath):
        print('Rostros: ', os.path.join(nameDir, fileName))
        labels.append(label)
        facesData.append(cv2.imread(os.path.join(personPath, fileName), 0))
    label += 1

# Crea y entrena el modelo de reconocimiento facial (EigenFace por defecto)
face_recognizer = cv2.face.EigenFaceRecognizer_create()

# Entrena el modelo
print('Entrenando...')
face_recognizer.train(facesData, np.array(labels))

# Guarda el modelo entrenado
model_path = os.environ.get('EIGENFACE_MODEL_PATH', str(BASE_DIR / 'modeloEigenFace.xml'))
face_recognizer.write(model_path)
print('Modelo almacenado...')