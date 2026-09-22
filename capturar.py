import os
from pathlib import Path

import cv2
import imutils


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

personName = 'Juan'  # Nombre de la persona a capturar
dataPath = os.environ.get('DATA_DIR', str(BASE_DIR / 'data'))
personPath = os.path.join(dataPath, personName)

if not os.path.exists(personPath):
    print('Carpeta creada: ', personPath)
    os.makedirs(personPath, exist_ok=True)

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('Video.mp4')

faceClassif = cv2.CascadeClassifier(
    os.environ.get('FACE_CASCADE_PATH', str(BASE_DIR / 'haarcascade_frontalface_default.xml'))
)
count = 0

while True:
    ret, frame = cap.read()
    if ret is False:
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(personPath, f'rostro_{count}.jpg'), rostro)
        count += 1

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 300:
        break

cap.release()
cv2.destroyAllWindows()