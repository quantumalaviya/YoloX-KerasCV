import tensorflow as tf
from keras_cv.models.__internal__.darknet_utils import (
    CrossStagePartial,
    DarknetConvBlock,
    DarknetConvBlockDepthwise,
)
from tensorflow import keras


class YoloXHead(keras.layers.Layer):
    def __init__(
        self,
        classes,
        bias_initializer,
        width_multiplier=1.0,
        in_channels=[256, 512, 1024],
        activation="silu",
        use_depthwise=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.stems = []

        self.classification_convs = []
        self.regression_convs = []

        self.classification_preds = []
        self.regression_preds = []
        self.objectness_preds = []

        ConvBlock = DarknetConvBlockDepthwise if use_depthwise else DarknetConvBlock

        for i in range(len(in_channels)):
            self.stems.append(
                DarknetConvBlock(
                    filters=int(256 * width_multiplier),
                    kernel_size=1,
                    strides=1,
                    activation=activation,
                )
            )

            self.classification_convs.append(
                keras.Sequential(
                    [
                        ConvBlock(
                            filters=int(256 * width_multiplier),
                            kernel_size=3,
                            strides=1,
                            activation=activation,
                        ),
                        ConvBlock(
                            filters=int(256 * width_multiplier),
                            kernel_size=3,
                            strides=1,
                            activation=activation,
                        ),
                    ]
                )
            )

            self.regression_convs.append(
                keras.Sequential(
                    [
                        ConvBlock(
                            filters=int(256 * width_multiplier),
                            kernel_size=3,
                            strides=1,
                            activation=activation,
                        ),
                        ConvBlock(
                            filters=int(256 * width_multiplier),
                            kernel_size=3,
                            strides=1,
                            activation=activation,
                        ),
                    ]
                )
            )

            self.classification_preds.append(
                keras.layers.Conv2D(
                    filters=classes,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    bias_initializer=bias_initializer,
                )
            )
            self.regression_preds.append(
                keras.layers.Conv2D(
                    filters=4,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    bias_initializer=bias_initializer,
                )
            )
            self.objectness_preds.append(
                keras.layers.Conv2D(
                    filters=1,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                )
            )

    def call(self, inputs, training=False):
        outputs = []

        for i, p_i in enumerate(inputs):
            stem = self.stems[i](p_i)

            classes = self.classification_convs[i](stem)
            classes = self.classification_preds[i](classes)

            boxes = self.regression_convs[i](stem)
            boxes = self.regression_preds[i](boxes)

            objectness = self.objectness_preds[i](stem)

            output = tf.keras.layers.Concatenate(axis=-1)([boxes, objectness, classes])
            outputs.append(output)

        return outputs
