import os
from pathlib import Path

import cv2
import numpy as np


def load_env(env_path: Path) -> None:
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

dataPath = os.environ.get('DATA_DIR', str(BASE_DIR / 'data'))
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = os.path.join(dataPath, nameDir)
    print('Leyendo las imágenes')

    for fileName in os.listdir(personPath):
        print('Rostros: ', os.path.join(nameDir, fileName))
        labels.append(label)
        facesData.append(cv2.imread(os.path.join(personPath, fileName), 0))
    label += 1

# Métodos para entrenar el reconocedor
face_recognizer = cv2.face.EigenFaceRecognizer_create()
# face_recognizer = cv2.face.FisherFaceRecognizer_create()
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenando el reconocedor de rostros
print('Entrenando...')
face_recognizer.train(facesData, np.array(labels))

# Almacenando el modelo obtenido
model_path = os.environ.get('EIGENFACE_MODEL_PATH', str(BASE_DIR / 'modeloEigenFace.xml'))
face_recognizer.write(model_path)
# face_recognizer.write('modeloFisherFace.xml')
# face_recognizer.write('modeloLBPHFace.xml')
print('Modelo almacenado...')