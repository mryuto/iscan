import os
from pathlib import Path

import cv2
import imutils

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

# Nombre de la persona cuyos rostros se capturarán
personName = input('Por favor, ingrese el nombre de la persona: ')

# Ruta base donde se guardarán los rostros capturados
dataPath = os.environ.get('DATA_DIR', str(BASE_DIR / 'data'))
personPath = os.path.join(dataPath, personName)

# Crea la carpeta si no existe
if not os.path.exists(personPath):
    print('Carpeta creada: ', personPath)
    os.makedirs(personPath, exist_ok=True)

# Inicializa la captura de video (cámara web)
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('Video.mp4')  # Para capturar desde un video

# Carga el clasificador de rostros
faceClassif = cv2.CascadeClassifier(
    os.environ.get('FACE_CASCADE_PATH', str(BASE_DIR / 'haarcascade_frontalface_default.xml'))
)

# Contador para nombres de imágenes
count = 0

while True:
    ret, frame = cap.read()
    if ret is False:
        break

    # Redimensiona el frame para mejor procesamiento
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    # Detecta rostros en el frame
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Dibuja un rectángulo alrededor del rostro detectado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Extrae el rostro y lo guarda como imagen
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(personPath, f'rostro_{count}.jpg'), rostro)
        count += 1

    # Muestra el frame en una ventana
    cv2.imshow('frame', frame)

    # Pausa con ESC o cuando se alcance el límite de capturas
    k = cv2.waitKey(1)
    if k == 27 or count >= 300:
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()
