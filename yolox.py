import keras_cv
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from keras_cv import bounding_box
from keras_cv.models.object_detection.object_detection_base_model import (
    ObjectDetectionBaseModel,
)
from tensorflow import keras

from __internal__.layers.yolox_decoder import DecodePredictions
from __internal__.layers.yolox_head import YoloXHead
from __internal__.layers.yolox_label_encoder import YoloXLabelEncoder
from __internal__.layers.yolox_pafpn import YoloXPAFPN

DEPTH_MULTIPLIERS = {
    "tiny": 0.33,
    "s": 0.33,
    "m": 0.67,
    "l": 1.00,
    "x": 1.33,
}
WIDTH_MULTIPLIERS = {
    "tiny": 0.375,
    "s": 0.50,
    "m": 0.75,
    "l": 1.00,
    "x": 1.25,
}


class YoloX(ObjectDetectionBaseModel):
    def __init__(
        self,
        classes,
        bounding_box_format,
        phi,
        backbone,
        include_rescaling=None,
        backbone_weights=None,
        label_encoder=None,
        prediction_decoder=None,
        feature_pyramid=None,
        evaluate_train_time_metrics=False,
        name="YoloX",
        **kwargs,
    ):
        label_encoder = label_encoder or YoloXLabelEncoder(
            bounding_box_format=bounding_box_format
        )

        super().__init__(
            bounding_box_format=bounding_box_format,
            label_encoder=label_encoder,
            name=name,
            **kwargs,
        )
        self.evaluate_train_time_metrics = evaluate_train_time_metrics

        self.depth_multiplier = DEPTH_MULTIPLIERS[phi]
        self.width_multiplier = WIDTH_MULTIPLIERS[phi]

        self.bounding_box_format = bounding_box_format
        self.classes = classes
        self.backbone = _parse_backbone(
            backbone,
            include_rescaling,
            backbone_weights,
            self.depth_multiplier,
            self.width_multiplier,
        )

        self.prediction_decoder = prediction_decoder or DecodePredictions(
            bounding_box_format=bounding_box_format,
            classes=classes,
        )

        self.feature_pyramid = feature_pyramid or YoloXPAFPN()
        bias_initializer = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.yolox_head = YoloXHead(classes, bias_initializer)

        self._metrics_bounding_box_format = None
        self.loss_metric = tf.keras.metrics.Mean(name="loss")

        self.classification_loss_metric = tf.keras.metrics.Mean(
            name="classification_loss"
        )
        self.objectness_loss_metric = tf.keras.metrics.Mean(name="objectness_loss")
        self.box_loss_metric = tf.keras.metrics.Mean(name="box_loss")
        self.regularization_loss_metric = tf.keras.metrics.Mean(
            name="regularization_loss"
        )

    def decode_predictions(self, x, predictions):
        # no-op if default decoder is used.
        pred_for_inference = self.prediction_decoder(x, predictions)
        return bounding_box.convert_format(
            pred_for_inference,
            source=self.prediction_decoder.bounding_box_format,
            target=self.bounding_box_format,
            images=x,
        )

    def compile(
        self,
        box_loss=None,
        objectness_loss=None,
        classification_loss=None,
        loss=None,
        metrics=None,
        **kwargs,
    ):
        super().compile(metrics=metrics, **kwargs)
        if loss is not None:
            raise ValueError(
                "`YoloX` does not accept a `loss` to `compile()`. "
                "Instead, please pass `box_loss`, `objectness_loss` and `classification_loss`. "
                "`loss` will be ignored during training."
            )
        self.box_loss = box_loss
        self.objectness_loss = objectness_loss
        self.classification_loss = classification_loss
        metrics = metrics or []

        if hasattr(classification_loss, "from_logits"):
            if not classification_loss.from_logits:
                raise ValueError(
                    "YoloX.compile() expects `from_logits` to be True for "
                    "`classification_loss`. Got "
                    "`classification_loss.from_logits="
                    f"{classification_loss.from_logits}`"
                )
        if hasattr(objectness_loss, "from_logits"):
            if not objectness_loss.from_logits:
                raise ValueError(
                    "YoloX.compile() expects `from_logits` to be True for "
                    "`objectness_loss`. Got "
                    "`objectness_loss.from_logits="
                    f"{objectness_loss.from_logits}`"
                )
        if hasattr(box_loss, "bounding_box_format"):
            if box_loss.bounding_box_format != self.bounding_box_format:
                raise ValueError(
                    "Wrong `bounding_box_format` passed to `box_loss` in "
                    "`YoloX.compile()`. "
                    f"Got `box_loss.bounding_box_format={box_loss.bounding_box_format}`, "
                    f"want `box_loss.bounding_box_format={self.bounding_box_format}`"
                )

        if len(metrics) != 0:
            self._metrics_bounding_box_format = metrics[0].bounding_box_format
        else:
            self._metrics_bounding_box_format = self.bounding_box_format

        any_wrong_format = any(
            [
                m.bounding_box_format != self._metrics_bounding_box_format
                for m in metrics
            ]
        )
        if metrics and any_wrong_format:
            raise ValueError(
                "All metrics passed to YoloX.compile() must have "
                "the same `bounding_box_format` attribute.  For example, if one metric "
                "uses 'xyxy', all other metrics must use 'xyxy'.  Received "
                f"metrics={metrics}."
            )

    @property
    def metrics(self):
        return super().metrics + self.train_metrics

    @property
    def train_metrics(self):
        return [
            self.loss_metric,
            self.classification_loss_metric,
            self.objectness_loss_metric,
            self.box_loss_metric,
        ]

    def call(self, x, training=False):
        backbone_outputs = self.backbone(x, training=training)
        features = self.feature_pyramid(backbone_outputs)

        return self.yolox_head(features)

    def _update_metrics(self, y_true, y_pred):
        y_true = bounding_box.convert_format(
            y_true,
            source=self.bounding_box_format,
            target=self._metrics_bounding_box_format,
        )
        y_pred = bounding_box.convert_format(
            y_pred,
            source=self.bounding_box_format,
            target=self._metrics_bounding_box_format,
        )
        self.compiled_metrics.update_state(y_true, y_pred)

    def compute_losses(self, y_true, y_pred, input_shape):
        pass

    def _backward(self, y_true, y_pred, input_shape):
        loss = self.compute_losses(y_true, y_pred, input_shape=input_shape)

        self.loss_metric.update_state(loss)
        return loss

    def train_step(self, data):
        x, y = data
        y_for_metrics, y_training_target = y

        y_training_target = bounding_box.convert_format(
            y_training_target,
            source=self.bounding_box_format,
            target="xywh",
            images=x,
        )

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self._backward(y_training_target, y_pred, input_shape=x.shape[1:3])

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Early exit for no train time metrics
        if not self.evaluate_train_time_metrics:
            # To minimize GPU transfers, we update metrics AFTER we take grads and apply
            # them.
            return {m.name: m.result() for m in self.train_metrics}

        predictions = self.decode_predictions(x, y_pred)
        predictions = predictions.to_tensor(default_value=-1)
        self._update_metrics(y_for_metrics, predictions)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_for_metrics, y_training_target = y
        y_pred = self(x, training=False)
        _ = self._backward(y_training_target, y_pred)

        predictions = self.decode_predictions(x, y_pred)
        self._update_metrics(y_for_metrics, predictions)
        return {m.name: m.result() for m in self.metrics}

    def predict(self, x, **kwargs):
        predictions = super().predict(x, **kwargs)
        predictions = self.decode_predictions(x, predictions)
        return predictions


def _parse_backbone(
    backbone,
    include_rescaling,
    backbone_weights,
    depth_multiplier=1.0,
    width_multiplier=1.0,
    use_depthwise=False,
):
    if isinstance(backbone, str) and include_rescaling is None:
        raise ValueError(
            "When using a preconfigured backbone, please do provide a "
            "`include_rescaling` parameter.  `include_rescaling` is passed to the "
            "Keras application constructor for the provided backbone.  When "
            "`include_rescaling=True`, image inputs are passed through a "
            "`layers.Rescaling(1/255.0)` layer. When `include_rescaling=False`, no "
            "downscaling is performed. "
            f"Received backbone={backbone}, include_rescaling={include_rescaling}."
        )

    if isinstance(backbone, str):
        if backbone == "cspdarknet":
            return _cspdarknet_backbone(
                include_rescaling,
                backbone_weights,
                depth_multiplier,
                width_multiplier,
                use_depthwise,
            )
        else:
            raise ValueError(
                "backbone expected to be one of ['cspdarknet', keras.Model]. "
                f"Received backbone={backbone}."
            )
    if include_rescaling or backbone_weights:
        raise ValueError(
            "When a custom backbone is used, include_rescaling and "
            f"backbone_weights are not supported.  Received backbone={backbone}, "
            f"include_rescaling={include_rescaling}, and "
            f"backbone_weights={backbone_weights}."
        )
    if not isinstance(backbone, keras.Model):
        raise ValueError(
            "Custom backbones should be subclasses of a keras.Model. "
            f"Received backbone={backbone}."
        )
    return backbone


def _cspdarknet_backbone(
    include_rescaling,
    backbone_weights,
    depth_multiplier,
    width_multiplier,
    use_depthwise,
):
    inputs = keras.layers.Input(shape=(None, None, 3))
    x = inputs

    backbone = keras_cv.models.CSPDarkNet(
        include_rescaling=include_rescaling,
        include_top=False,
        weights=backbone_weights,
        depth_multiplier=depth_multiplier,
        width_multiplier=width_multiplier,
        use_depthwise=use_depthwise,
        input_tensor=x,
    )

    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["dark3_csp", "dark4_csp", "dark5_csp"]
    ]
    return keras.Model(inputs=inputs, outputs=[c3_output, c4_output, c5_output])
