import tensorflow as tf
from keras_cv.models.__internal__.darknet_utils import (
    CrossStagePartial,
    DarknetConvBlock,
    DarknetConvBlockDepthwise,
)
from tensorflow import keras


class YoloXPAFPN(keras.layers.Layer):
    def __init__(
        self,
        depth_multiplier=1.0,
        width_multiplier=1.0,
        in_channels=[256, 512, 1024],
        use_depthwise=False,
        activation="silu",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels

        ConvBlock = DarknetConvBlockDepthwise if use_depthwise else DarknetConvBlock

        self.lateral_conv0 = DarknetConvBlock(
            filters=int(in_channels[1] * width_multiplier),
            kernel_size=1,
            strides=1,
            activation=activation,
        )
        self.C3_p4 = CrossStagePartial(
            filters=int(in_channels[1] * width_multiplier),
            num_bottlenecks=round(3 * depth_multiplier),
            residual=False,
            use_depthwise=use_depthwise,
            activation=activation,
        )

        self.reduce_conv1 = DarknetConvBlock(
            filters=int(in_channels[0] * width_multiplier),
            kernel_size=1,
            strides=1,
            activation=activation,
        )
        self.C3_p3 = CrossStagePartial(
            filters=int(in_channels[0] * width_multiplier),
            num_bottlenecks=round(3 * depth_multiplier),
            residual=False,
            use_depthwise=use_depthwise,
            activation=activation,
        )

        self.bu_conv2 = ConvBlock(
            filters=int(in_channels[0] * width_multiplier),
            kernel_size=3,
            strides=2,
            activation=activation,
        )
        self.C3_n3 = CrossStagePartial(
            filters=int(in_channels[1] * width_multiplier),
            num_bottlenecks=round(3 * depth_multiplier),
            residual=False,
            use_depthwise=use_depthwise,
            activation=activation,
        )

        self.bu_conv1 = ConvBlock(
            filters=int(in_channels[1] * width_multiplier),
            kernel_size=3,
            strides=2,
            activation=activation,
        )
        self.C3_n4 = CrossStagePartial(
            filters=int(in_channels[2] * width_multiplier),
            num_bottlenecks=round(3 * depth_multiplier),
            residual=False,
            use_depthwise=use_depthwise,
            activation=activation,
        )

        self.concat = keras.layers.Concatenate(axis=-1)
        self.upsample_2x = keras.layers.UpSampling2D(2)

    def call(self, inputs, training=False):
        c3_output, c4_output, c5_output = inputs

        fpn_out0 = self.lateral_conv0(c5_output)
        f_out0 = self.upsample_2x(fpn_out0)
        f_out0 = self.concat([f_out0, c4_output])
        f_out0 = self.C3_p4(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = self.upsample_2x(fpn_out1)
        f_out1 = self.concat([f_out1, c3_output])
        pan_out2 = self.C3_p3(f_out1)

        p_out1 = self.bu_conv2(pan_out2)
        p_out1 = self.concat([p_out1, fpn_out1])
        pan_out1 = self.C3_n3(p_out1)

        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = self.concat([p_out0, fpn_out0])
        pan_out0 = self.C3_n4(p_out0)

        return pan_out2, pan_out1, pan_out0
