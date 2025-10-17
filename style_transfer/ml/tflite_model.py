import numpy as np
import tensorflow as tf
from PIL import Image

class TFLiteStyleTransfer:
    def __init__(self, model_path):
        # Load the TFLite model and allocate tensors
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess(self, image: Image.Image, target_size=(256, 256)):
        """Convert PIL image to normalized float32 numpy array"""
        image = image.resize(target_size)
        img_array = np.array(image).astype(np.float32)[np.newaxis, ...] / 255.0
        return img_array

    def stylize(self, content_img: Image.Image, style_img: Image.Image):
        """Apply style transfer"""
        content = self.preprocess(content_img)
        style = self.preprocess(style_img)

        # Set inputs
        self.interpreter.set_tensor(self.input_details[0]['index'], content)
        self.interpreter.set_tensor(self.input_details[1]['index'], style)

        # Run inference
        self.interpreter.invoke()

        # Get output tensor
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        output = np.clip(output, 0, 1)

        # Convert to PIL image
        output_image = Image.fromarray((output * 255).astype(np.uint8))
        return output_image
