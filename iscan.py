import os
from abc import ABC, abstractmethod
from pathlib import Path
import cv2
from datetime import datetime

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


# Directorio base del script
BASE_DIR = Path(__file__).resolve().parent
load_env(BASE_DIR / '.env')  # Carga variables de entorno (.env)

# Ruta base donde se encuentran los datos de entrenamiento
dataPath = os.environ.get('DATA_DIR', str(BASE_DIR / 'data'))

# Lista de carpetas (personas) en el directorio de datos
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)  # Muestra las imágenes disponibles

# Inicializa el reconocedor de rostros (EigenFace por defecto)
face_recognizer = cv2.face.EigenFaceRecognizer_create()

# Carga el modelo entrenado
model_path = os.environ.get('EIGENFACE_MODEL_PATH', str(BASE_DIR / 'modeloEigenFace.xml'))
face_recognizer.read(model_path)

# Inicializa la captura de video (cámara web)
cap = cv2.VideoCapture(0)

# Carga el clasificador para detectar rostros
faceClassif = cv2.CascadeClassifier(
    os.environ.get('FACE_CASCADE_PATH', str(BASE_DIR / 'haarcascade_frontalface_default.xml'))
)

usuario = None  # Variable para almacenar el nombre del usuario identificado
count = 0  # Contador de rostros detectados
registro = []  # Lista para registrar los usuarios identificados
registro2 = []  # Lista para registrar los usuarios identificados (sin duplicados)
registro3 = []  # Lista para registrar los usuarios identificados (sin duplicados)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Salir si no se puede capturar el frame

    # Convierte a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()  # Copia para procesamiento adicional

    # Detecta rostros en la imagen
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extrae el rostro detectado y lo redimensiona
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

        # Realiza la predicción del rostro
        result = face_recognizer.predict(rostro)

        # Muestra el ID del usuario en la imagen
        cv2.putText(frame, f'{result}', (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
        # print(f'Predicción: {result}')  # Muestra el resultado de la predicción en la consola

        # Verifica si el rostro pertenece a un usuario conocido
        if result[1] < 5700:  # Umbral de confianza para EigenFace
            # Muestra el nombre del usuario
            cv2.putText(frame, f'{imagePaths[result[0]]}', (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Marca el rostro con un rectángulo verde
            print(f'Usuario: {imagePaths[result[0]]}, contador: {count}')  # Muestra el usuario en la consola
            if usuario != imagePaths[result[0]]:
                usuario = imagePaths[result[0]]  # Actualiza el nombre del usuario identificado
                count = 0  # Reinicia el contador si se identifica un nuevo usuario
            else:
                count += 1

        else:
            # Muestra "Desconocido" si no se identifica al usuario
            cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Marca el rostro con un rectángulo rojo
            count = 0  # Reinicia el contador si el rostro es desconocido

    if usuario and count == 50:  # Registra al usuario si ha sido identificado durante 50s iteraciones consecutivas
        # 1 si entra, 0 si sale
        if usuario not in registro2:
            registro2.append(usuario)
            registro.append([usuario, datetime.now(), 1])  # Registra la entrada del usuario
        else:
            registro2.remove(usuario)
            registro.append([usuario, datetime.now(), 0])  # Registra la salida del usuario
        count = 0  # Reinicia el contador después de registrar al usuario

        if usuario not in registro3:
            registro3.append(usuario)


    # Muestra el frame en una ventana
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27:  # Pausa con la tecla ESC
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()

# Crear archivo html con el registro de usuarios identificados
# tailwindcss.com
with open(f'planilla-{datetime.now().strftime("%Y%m%d%H%M%S")}.html', 'w') as f:
    f.write('<!DOCTYPE html>')
    f.write('<html lang="es"><head>')
    f.write('<meta charset="UTF-8">')
    f.write('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
    f.write('<title>Planilla de Registro</title>')
    f.write('<script src="https://cdn.tailwindcss.com"></script>')
    f.write('</head><body class="bg-slate-100 text-slate-800 antialiased">')
    f.write('<main class="mx-auto max-w-5xl px-4 py-10">')
    f.write(f'<div class="mb-8 rounded-2xl bg-slate-100 p-6 shadow-md ring-1 ring-slate-200">')
    f.write(f'<h2 class="text-xl font-semibold text-slate-700"> Fecha: {datetime.now().strftime("%Y-%m-%d")}</h2>')
    f.write('</div>')
    # Planilla de registro de usuarios identificados (Presente y Ausente)
    # Tabla usuario y presente o ausente, con color verde para presente y rojo para ausente
    f.write('<div class="mb-8 rounded-2xl bg-white p-6 shadow-md ring-1 ring-slate-200">')
    f.write('<h1 class="text-3xl font-bold text-slate-900 text-center">Planilla de Registro</h1>')
    f.write('</div>')
    f.write('<div class="overflow-hidden rounded-2xl bg-white shadow-md ring-1 ring-slate-200">')
    f.write('<table class="min-w-full divide-y divide-slate-200 text-left text-sm">')
    f.write('<thead class="bg-slate-50"><tr><th class="px-6 py-3 font-semibold text-slate-700">Usuario</th><th class="px-6 py-3 font-semibold text-slate-700">Estado</th></tr></thead>')
    f.write('<tbody class="divide-y divide-slate-200">')
    for alumno in imagePaths:
        if alumno in registro3:
            f.write(f'<tr class="bg-emerald-50"><td class="px-6 py-4 font-medium text-slate-900">{alumno}</td><td class="px-6 py-4"><span class="inline-flex rounded-full px-2.5 py-1 text-xs font-semibold text-emerald-700 ring-1 ring-inset ring-emerald-200 bg-emerald-100">Presente</span></td></tr>')
        else:
            f.write(f'<tr class="bg-rose-50"><td class="px-6 py-4 font-medium text-slate-900">{alumno}</td><td class="px-6 py-4"><span class="inline-flex rounded-full px-2.5 py-1 text-xs font-semibold text-rose-700 ring-1 ring-inset ring-rose-200 bg-rose-100">Ausente</span></td></tr>')
    f.write('</tbody></table>')
    f.write('</div>')
    # Espacio entre la planilla y el registro de usuarios identificados
    f.write('<div class="my-16"></div>')
    f.write('<div class="mb-8 rounded-2xl bg-white p-6 shadow-md ring-1 ring-slate-200">')
    f.write('<h1 class="text-3xl font-bold text-slate-900 text-center">Registro de Usuarios Identificados</h1>')
    f.write('</div>')
    f.write('<div class="overflow-hidden rounded-2xl bg-white shadow-md ring-1 ring-slate-200">')
    f.write('<table class="min-w-full divide-y divide-slate-200 text-left text-sm">')
    f.write('<thead class="bg-slate-50"><tr><th class="px-6 py-3 font-semibold text-slate-700">Usuario</th><th class="px-6 py-3 font-semibold text-slate-700">Timestamp</th><th class="px-6 py-3 font-semibold text-slate-700">Estado</th></tr></thead>')
    f.write('<tbody class="divide-y divide-slate-200">')
    for user, timestamp, status in registro:
        status_text = 'Entrada' if status == 1 else 'Salida'
        row_class = 'bg-emerald-50' if status == 1 else 'bg-rose-50'
        status_badge_class = 'inline-flex rounded-full px-2.5 py-1 text-xs font-semibold text-emerald-700 ring-1 ring-inset ring-emerald-200 bg-emerald-100' if status == 1 else 'inline-flex rounded-full px-2.5 py-1 text-xs font-semibold text-rose-700 ring-1 ring-inset ring-rose-200 bg-rose-100'
        f.write(f'<tr class="{row_class}"><td class="px-6 py-4 font-medium text-slate-900">{user}</td><td class="px-6 py-4 text-slate-600">{timestamp}</td><td class="px-6 py-4"><span class="{status_badge_class}">{status_text}</span></td></tr>')
    f.write('</tbody></table>')
    f.write('</div>')
    f.write('</main></body></html>')

print('Registro de usuarios identificados:')
for user, timestamp, status in registro:
    status_text = "Entrada" if status == 1 else "Salida"
    print(f'Usuario: {user}, Timestamp: {timestamp}, Estado: {status_text}')