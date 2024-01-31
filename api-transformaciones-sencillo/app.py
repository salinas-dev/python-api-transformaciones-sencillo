from flask import Flask, request, render_template, jsonify
import base64
import io
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/apply_transformations', methods=['POST'])
def apply_transformations():
    # Obtener los valores de transformación desde la solicitud POST
    data = request.get_json()
    image_data = data['imageData']
    rotation = float(data['rotation'])
    scale = float(data['scale'])
    shear_x = float(data['shearX'])
    shear_y = float(data['shearY'])
    translate_x = float(data['translateX'])
    translate_y = float(data['translateY'])

    # Decodificar la imagen base64
    image_data = image_data.split(',')[1]  # Eliminar el encabezado de datos
    image_data = base64.b64decode(image_data)
    image_np = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Aplicar transformaciones a la imagen
    if rotation != 0:
        rows, cols, _ = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
        image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale)

    if shear_x != 0 or shear_y != 0:
        shear_matrix = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
        image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))

    if translate_x != 0 or translate_y != 0:
        translation_matrix = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
        image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

    # Codificar la imagen resultante a base64
    _, buffer = cv2.imencode('.png', image)
    image_data = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'transformedImageData': image_data})

if __name__ == '__main__':
    app.run(debug=True)

