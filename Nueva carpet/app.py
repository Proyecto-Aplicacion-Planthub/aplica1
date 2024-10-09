from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
import mediapipe as mp
import math

app = Flask(__name__)

# Configuración de Mediapipe para la malla facial
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inicializar captura de video
cap = cv2.VideoCapture(0)

# Función para contar parpadeos y capturar imágenes
def generate_frames(username):
    blink_count = 0
    blink_detected = False

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:
            success, frame = cap.read()
            if not success:
                print("No se pudo capturar el frame.")
                break

            # Convertir el frame a RGB para Mediapipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            # Dibujar la malla facial en el frame
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )

                    # Convertir los puntos a una lista de coordenadas
                    landmarks = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in face_landmarks.landmark]

                    # Coordenadas de los párpados para detectar parpadeos
                    eye_right = landmarks[145], landmarks[159]
                    dist_right_eye = math.hypot(eye_right[0][0] - eye_right[1][0], eye_right[0][1] - eye_right[1][1])

                    eye_left = landmarks[374], landmarks[386]
                    dist_left_eye = math.hypot(eye_left[0][0] - eye_left[1][0], eye_left[0][1] - eye_left[1][1])

                    # Detectar parpadeos si la distancia entre los párpados es pequeña
                    if dist_right_eye < 10 and dist_left_eye < 10 and not blink_detected:
                        blink_count += 1
                        blink_detected = True
                    elif dist_right_eye > 10 and dist_left_eye > 10:
                        blink_detected = False

                    # Mostrar el contador de parpadeos en el frame
                    cv2.putText(frame, f"Parpadeos: {blink_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # Capturar la imagen del rostro después de 3 parpadeos
                    if blink_count >= 3:
                        face_locations = [face_landmarks.landmark]
                        cv2.imwrite(f'static/faces/{username}.jpg', frame)
                        blink_count = 0  # Reiniciar el contador

            # Codificar el frame a formato JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("No se pudo codificar el frame.")
                continue

            frame = buffer.tobytes()

            # Enviar el frame al navegador
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

# Ruta para manejar el registro
@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    username = request.form['username']
    password = request.form['password']

    # Guardar los datos en un archivo
    with open(f'users/{username}.txt', 'w') as f:
        f.write(f'Name: {name}\n')
        f.write(f'Username: {username}\n')
        f.write(f'Password: {password}\n')

    # Redirigir a la captura biométrica
    return redirect(url_for('biometric_capture', username=username))

# Ruta para la captura biométrica
@app.route('/biometric_capture/<username>')
def biometric_capture(username):
    return render_template('capture.html', username=username)

# Ruta para servir el video en tiempo real
@app.route('/video_feed/<username>')
def video_feed(username):
    return Response(generate_frames(username), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Crear carpetas si no existen
    if not os.path.exists('users'):
        os.makedirs('users')
    if not os.path.exists('static/faces'):
        os.makedirs('static/faces')

    app.run(debug=True)
