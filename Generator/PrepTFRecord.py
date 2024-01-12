import os
import tensorflow as tf

class DataGenerator:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.classes = os.listdir(input_folder)
        self.class_to_label = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def get_labels(self, file_path):
        # Extract class name from the file path and convert to label
        parts = tf.strings.split(file_path, os.path.sep)
        class_name = parts[-2]
        label = self.class_to_label[class_name.numpy()]
        return label

    def load_image(self, file_path):
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        return img

    def wrap_func(self, file_path):
        label = tf.py_function(self.get_labels, [file_path], [tf.int64])
        img = self.load_image(file_path)
        return img, label[0]

    def genDS(self):
        # List all image files
        image_paths = []
        for class_name in self.classes:
            class_path = os.path.join(self.input_folder, class_name)
            for img_name in os.listdir(class_path):
                image_paths.append(os.path.join(class_path, img_name))

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(lambda x: self.wrap_func(x))
        return dataset
