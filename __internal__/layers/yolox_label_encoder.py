import tensorflow as tf
from tensorflow.keras import layers

from keras_cv import bounding_box


class YoloXLabelEncoder(layers.Layer):
    """Transforms the raw labels into targets for training.
    Args:
        bounding_box_format:  The format of bounding boxes of input dataset. Refer
            [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
            for more details on supported bounding box formats.
    """

    def __init__(self, bounding_box_format, **kwargs):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format

    def call(self, images, target_boxes):
        """Creates box and classification targets for a batch"""
        if isinstance(images, tf.RaggedTensor):
            raise ValueError(
                "`YoloXLabelEncoder`'s `call()` method does not "
                "support RaggedTensor inputs for the `images` argument.  Received "
                f"`type(images)={type(images)}`."
            )

        if isinstance(target_boxes, tf.RaggedTensor):
            target_boxes = target_boxes.to_tensor(default_value=-1)

        return target_boxes