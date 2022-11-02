from tflite_runtime.interpreter import Interpreter 
from PIL import Image
import numpy as np
import time

def load_labels(path):
# Чтение меток из файла
  with open(path, 'r') as f:
    return [line.strip() for i, line in enumerate(f.readlines())]

def set_input_tensor(interpreter, image):
# Подготовка изображения
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
# Классифицирование изображений 
  set_input_tensor(interpreter, image)

  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))
# Обработка результатов
  scale, zero_point = output_details['quantization']
  output = scale * (output - zero_point)

  ordered = np.argpartition(-output, 1)
  return [(i, output[i]) for i in ordered[:top_k]][0]

model_path = "mobilenet_v1_1.0_224_quant.tflite"
label_path = "labels_mobilenet_quant_v1_224.txt"

interpreter = Interpreter(model_path)
print("Модель загружена")

interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']
print("Формат изображения (", width, ",", height, ")")

# Загрузка изображения
image = Image.open("koala.jpg").convert('RGB').resize((width, height))

# Классификация
label_id, prob = classify_image(interpreter, image)

# Загрузка меток
labels = load_labels(label_path)

# Обработка результатов
classification_label = labels[label_id]
print("На картинке ", classification_label, ", с точностью ", np.round(prob*100, 2), "%.")
