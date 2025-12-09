import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
import pandas as pd
import io

# Configurar TensorFlow para usar menos memoria
tf.config.experimental.set_memory_growth(
    tf.config.experimental.list_physical_devices('GPU')[0], True
) if tf.config.experimental.list_physical_devices('GPU') else None

app = FastAPI(
    title="Food Classifier API",
    description="API para clasificar imágenes de comida usando CNN",
    version="1.0.0"
)

# Variables globales para el intérprete TFLite
interpreter = None
input_details = None
output_details = None

def load_model():
    """Cargar el modelo TFLite"""
    global interpreter, input_details, output_details
    model_path = "1.tflite"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
    
    try:
        # Cargar el intérprete TFLite
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Obtener detalles de entrada y salida
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("✓ Modelo TFLite cargado exitosamente")
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Output shape: {output_details[0]['shape']}")
        return interpreter
    except Exception as e:
        print(f"✗ Error al cargar el modelo: {e}")
        raise e

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesar imagen siguiendo los mismos pasos del notebook:
    1. Redimensionar a 300x300
    2. Convertir a escala de grises (canal 0)
    3. Normalizar [0,1]
    4. Reshape para el modelo
    """
    try:
        # Redimensionar a 300x300
        image = image.resize((192, 192))
        
        # Convertir a RGB si es necesario, luego a array
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convertir a numpy array
        img_array = np.array(image)
        
        # Extraer solo el canal 0 (como en el notebook)
        #if len(img_array.shape) == 3:
        #    img_array = img_array[:, :, 0]
        
        # Reshape para que sea (300, 300, 1)
        img_array = img_array.reshape((192, 192, 3))
        
        # Convertir a float32 y normalizar
        img_array = img_array.astype(np.uint8)
        
        # Agregar dimensión del batch: (1, 300, 300, 1)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar imagen: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Cargar el modelo al iniciar la aplicación"""
    try:
        load_model()
    except Exception as e:
        print(f"Error al inicializar: {e}")
        # La aplicación seguirá funcionando, pero las predicciones fallarán

@app.get("/")
async def root():
    """Endpoint raíz con información básica"""
    return {
        "message": "Horse vs Human Classifier API",
        "status": "active",
        "model_loaded": interpreter is not None,
        "model_type": "TensorFlow Lite",
        "endpoints": {
            "predict": "/predict - POST con imagen",
            "health": "/health - GET para verificar estado"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": interpreter is not None,
        "model_type": "TensorFlow Lite",
        "tensorflow_version": tf.__version__
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predecir clase de comida
    
    Returns:
        - class: 0 a 2022 dependiendo de clase
        - class_name: nombre de la comida
        - confidence: probabilidad de la predicción
    """
    
    # Verificar que el modelo esté cargado
    if interpreter is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    # Verificar que sea una imagen
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    
    try:
        # Leer la imagen
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocesar la imagen
        processed_image = preprocess_image(image)
        
        # Hacer predicción con TFLite
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        # Extraer resultados
        probabilities = prediction[0].tolist()
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))/2024
        
        # Mapear clase a nombre
        label_map = pd.read_csv('aiy_food_V1_labelmap.csv')
        class_name = label_map.iloc[predicted_class]['name']
        #class_name = class_names[predicted_class]
        
        return {
            "class": predicted_class,
            "class_name": class_name,
            "confidence": confidence,
            "image_info": {
                "filename": file.filename,
                "content_type": file.content_type,
                "processed_shape": processed_image.shape
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
