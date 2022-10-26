import keras_cv
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from keras_cv import bounding_box
from keras_cv.models.object_detection.object_detection_base_model import \
    ObjectDetectionBaseModel
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

        self.feature_pyramid = feature_pyramid or YoloXPAFPN(
            depth_multiplier=self.depth_multiplier,
            width_multiplier=self.width_multiplier
        )
        bias_initializer = None
        self.yolox_head = YoloXHead(classes, bias_initializer, width_multiplier=self.width_multiplier)

        self._metrics_bounding_box_format = None
        self.loss_metric = tf.keras.metrics.Mean(name="loss")

        self.classification_loss_metric = tf.keras.metrics.Mean(
            name="classification_loss"
        )
        self.objectness_loss_metric = tf.keras.metrics.Mean(name="objectness_loss")
        self.box_loss_metric = tf.keras.metrics.Mean(name="box_loss")

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
        '''
        if hasattr(box_loss, "bounding_box_format"):
            if box_loss.bounding_box_format != self.bounding_box_format:
                raise ValueError(
                    "Wrong `bounding_box_format` passed to `box_loss` in "
                    "`YoloX.compile()`. "
                    f"Got `box_loss.bounding_box_format={box_loss.bounding_box_format}`, "
                    f"want `box_loss.bounding_box_format={self.bounding_box_format}`"
                )
        '''
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

    def get_assignments(
        self,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        obj_preds_per_image,
        cls_preds_per_image,
        x_shifts,
        y_shifts,
        expanded_strides,
        num_classes,
        num_gt,
        total_num_anchors,
    ):
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            x_shifts,
            y_shifts,
            expanded_strides,
            num_gt,
            total_num_anchors,
        )
        bboxes_preds_per_image = tf.boolean_mask(
            bboxes_preds_per_image, fg_mask, axis=0
        )
        obj_preds_ = tf.boolean_mask(obj_preds_per_image, fg_mask, axis=0)
        cls_preds_ = tf.boolean_mask(cls_preds_per_image, fg_mask, axis=0)
        num_in_boxes_anchor = tf.shape(bboxes_preds_per_image)[0]
        
        tf.debugging.assert_equal(tf.shape(obj_preds_), [tf.math.count_nonzero(fg_mask, dtype=tf.int32), 1])
        tf.debugging.assert_equal(tf.shape(bboxes_preds_per_image), [tf.math.count_nonzero(fg_mask, dtype=tf.int32), 4])
        tf.debugging.assert_equal(tf.shape(cls_preds_), [tf.math.count_nonzero(fg_mask, dtype=tf.int32), self.classes])

        pair_wise_ious = bounding_box.compute_iou(
            gt_bboxes_per_image, bboxes_preds_per_image, "center_xywh"
        )
        pair_wise_ious_loss = -tf.math.log(pair_wise_ious + 1e-8)
        gt_cls_per_image = tf.tile(
            tf.expand_dims(tf.one_hot(tf.cast(gt_classes, tf.int32), num_classes), 1),
            (1, num_in_boxes_anchor, 1),
        )
        cls_preds_ = K.sigmoid(
            tf.tile(tf.expand_dims(cls_preds_, 0), (num_gt, 1, 1))
        ) * K.sigmoid(tf.tile(tf.expand_dims(obj_preds_, 0), (num_gt, 1, 1)))

        pair_wise_cls_loss = tf.reduce_sum(
            K.binary_crossentropy(gt_cls_per_image, tf.sqrt(cls_preds_)), -1
        )

        tf.debugging.assert_equal(tf.shape(pair_wise_ious), [num_gt, tf.math.count_nonzero(fg_mask, dtype=tf.int32)])
        tf.debugging.assert_equal(tf.shape(cls_preds_), [num_gt, tf.math.count_nonzero(fg_mask, dtype=tf.int32), self.classes])
        tf.debugging.assert_equal(tf.shape(gt_cls_per_image), [num_gt, tf.math.count_nonzero(fg_mask, dtype=tf.int32), self.classes])
        tf.debugging.assert_equal(tf.shape(pair_wise_cls_loss), [num_gt, tf.math.count_nonzero(fg_mask, dtype=tf.int32)])

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0
            * tf.cast((~is_in_boxes_and_center), K.dtype(bboxes_preds_per_image))
        )

        (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, fg_mask, gt_classes, num_gt)
        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        x_shifts,
        y_shifts,
        expanded_strides,
        num_gt,
        total_num_anchors,
        center_radius=2.5,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = tf.tile(
            tf.expand_dims(((x_shifts[0] + 0.5) * expanded_strides_per_image), 0),
            [num_gt, 1],
        )
        y_centers_per_image = tf.tile(
            tf.expand_dims(((y_shifts[0] + 0.5) * expanded_strides_per_image), 0),
            [num_gt, 1],
        )
        
        tf.debugging.assert_equal(tf.shape(expanded_strides_per_image), [5376])
        tf.debugging.assert_equal(tf.shape(x_centers_per_image), [num_gt, 5376])
        tf.debugging.assert_equal(tf.shape(y_centers_per_image), [num_gt, 5376])

        gt_bboxes_per_image_l = tf.tile(
            tf.expand_dims(
                (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]), 1
            ),
            [1, total_num_anchors],
        )
        gt_bboxes_per_image_r = tf.tile(
            tf.expand_dims(
                (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]), 1
            ),
            [1, total_num_anchors],
        )
        gt_bboxes_per_image_t = tf.tile(
            tf.expand_dims(
                (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]), 1
            ),
            [1, total_num_anchors],
        )
        gt_bboxes_per_image_b = tf.tile(
            tf.expand_dims(
                (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]), 1
            ),
            [1, total_num_anchors],
        )

        tf.debugging.assert_equal(tf.shape(gt_bboxes_per_image_l), [num_gt, 5376])
        tf.debugging.assert_equal(tf.shape(gt_bboxes_per_image_r), [num_gt, 5376])
        tf.debugging.assert_equal(tf.shape(gt_bboxes_per_image_t), [num_gt, 5376])
        tf.debugging.assert_equal(tf.shape(gt_bboxes_per_image_b), [num_gt, 5376])

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = tf.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = tf.reduce_min(bbox_deltas, axis=-1) > 0.0
        is_in_boxes_all = (
            tf.reduce_sum(tf.cast(is_in_boxes, K.dtype(gt_bboxes_per_image)), axis=0)
            > 0.0
        )

        tf.debugging.assert_equal(tf.shape(bbox_deltas), [num_gt, 5376, 4])
        tf.debugging.assert_equal(tf.shape(is_in_boxes), [num_gt, 5376])
        tf.debugging.assert_equal(tf.shape(is_in_boxes_all), [5376])

        gt_bboxes_per_image_l = tf.tile(
            tf.expand_dims(gt_bboxes_per_image[:, 0], 1), [1, total_num_anchors]
        ) - center_radius * tf.expand_dims(expanded_strides_per_image, 0)
        gt_bboxes_per_image_r = tf.tile(
            tf.expand_dims(gt_bboxes_per_image[:, 0], 1), [1, total_num_anchors]
        ) + center_radius * tf.expand_dims(expanded_strides_per_image, 0)
        gt_bboxes_per_image_t = tf.tile(
            tf.expand_dims(gt_bboxes_per_image[:, 1], 1), [1, total_num_anchors]
        ) - center_radius * tf.expand_dims(expanded_strides_per_image, 0)
        gt_bboxes_per_image_b = tf.tile(
            tf.expand_dims(gt_bboxes_per_image[:, 1], 1), [1, total_num_anchors]
        ) + center_radius * tf.expand_dims(expanded_strides_per_image, 0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = tf.stack([c_l, c_t, c_r, c_b], 2)

        is_in_centers = tf.reduce_min(center_deltas, axis=-1) > 0.0
        is_in_centers_all = (
            tf.reduce_sum(tf.cast(is_in_centers, K.dtype(gt_bboxes_per_image)), axis=0)
            > 0.0
        )
        tf.debugging.assert_equal(tf.shape(center_deltas), [num_gt, 5376, 4])
        tf.debugging.assert_equal(tf.shape(is_in_centers), [num_gt, 5376])
        tf.debugging.assert_equal(tf.shape(is_in_centers_all), [5376])

        fg_mask = tf.cast(is_in_boxes_all | is_in_centers_all, tf.bool)

        is_in_boxes_and_center = tf.boolean_mask(
            is_in_boxes, fg_mask, axis=1
        ) & tf.boolean_mask(is_in_centers, fg_mask, axis=1)
        
        tf.debugging.assert_equal(tf.shape(fg_mask), [5376])
        tf.debugging.assert_equal(tf.shape(is_in_boxes_and_center), [num_gt, tf.math.count_nonzero(fg_mask, dtype=tf.int32)])

        return fg_mask, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, fg_mask, gt_classes, num_gt):
        matching_matrix = tf.zeros_like(cost)

        tf.debugging.assert_equal(tf.shape(matching_matrix), [num_gt, tf.math.count_nonzero(fg_mask, dtype=tf.int32)])
        tf.debugging.assert_equal(tf.shape(cost), [num_gt, tf.math.count_nonzero(fg_mask, dtype=tf.int32)])

        n_candidate_k = tf.minimum(10, tf.shape(pair_wise_ious)[1])
        topk_ious, _ = tf.nn.top_k(pair_wise_ious, n_candidate_k)
        dynamic_ks = tf.maximum(tf.reduce_sum(topk_ious, 1), 1)

        tf.debugging.assert_equal(tf.shape(topk_ious), [num_gt, n_candidate_k])
        tf.debugging.assert_equal(tf.shape(dynamic_ks), [num_gt])

        def loop_body_1(b, matching_matrix):
            _, pos_idx = tf.nn.top_k(-cost[b], k=tf.cast(dynamic_ks[b], tf.int32))
            matching_matrix = tf.concat(
                [
                    matching_matrix[:b],
                    tf.expand_dims(
                        tf.reduce_max(tf.one_hot(pos_idx, tf.shape(cost)[1]), 0), 0
                    ),
                    matching_matrix[b + 1 :],
                ],
                axis=0,
            )
            return b + 1, matching_matrix

        _, matching_matrix = tf.while_loop(
            lambda b, *args: b < tf.cast(num_gt, tf.int32),
            loop_body_1,
            [0, matching_matrix],
        )

        anchor_matching_gt = tf.reduce_sum(matching_matrix, 0)
        
        tf.debugging.assert_equal(tf.shape(anchor_matching_gt), [tf.math.count_nonzero(fg_mask, dtype=tf.int32)])
    
        biger_one_indice = tf.reshape(tf.where(anchor_matching_gt > 1), [-1])

        def loop_body_2(b, matching_matrix):
            indice_anchor = tf.cast(biger_one_indice[b], tf.int32)
            indice_gt = tf.math.argmin(cost[:, indice_anchor])
            matching_matrix = tf.concat(
                [
                    matching_matrix[:, :indice_anchor],
                    tf.expand_dims(tf.one_hot(indice_gt, tf.cast(num_gt, tf.int32)), 1),
                    matching_matrix[:, indice_anchor + 1 :],
                ],
                axis=-1,
            )
            return b + 1, matching_matrix

        _, matching_matrix = tf.while_loop(
            lambda b, *args: b < tf.cast(tf.shape(biger_one_indice)[0], tf.int32),
            loop_body_2,
            [0, matching_matrix],
        )

        fg_mask_inboxes = tf.reduce_sum(matching_matrix, 0) > 0.0
        num_fg = tf.reduce_sum(tf.cast(fg_mask_inboxes, K.dtype(cost)))
        
        tf.debugging.assert_equal(tf.shape(fg_mask_inboxes), [tf.math.count_nonzero(fg_mask, dtype=tf.int32)])
    
        fg_mask_indices = tf.reshape(tf.where(fg_mask), [-1])
        fg_mask_inboxes_indices = tf.reshape(tf.where(fg_mask_inboxes), [-1, 1])
        fg_mask_select_indices = tf.gather_nd(fg_mask_indices, fg_mask_inboxes_indices)
        fg_mask = tf.cast(
            tf.reduce_max(tf.one_hot(fg_mask_select_indices, tf.shape(fg_mask)[0]), 0),
            K.dtype(fg_mask),
        )

        matched_gt_inds = tf.math.argmax(
            tf.boolean_mask(matching_matrix, fg_mask_inboxes, axis=1), 0
        )
        gt_matched_classes = tf.gather_nd(
            gt_classes, tf.reshape(matched_gt_inds, [-1, 1])
        )

        pred_ious_this_matching = tf.boolean_mask(
            tf.reduce_sum(matching_matrix * pair_wise_ious, 0), fg_mask_inboxes
        )
        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def compute_losses(self, y_true, y_pred, input_shape=(640, 640)):
        num_levels = len(y_pred)

        if y_true.shape[-1] != 5:
            raise ValueError(
                "y_true should have shape (None, None, 5).  Got "
                f"y_true.shape={tuple(y_true.shape)}"
            )

        for i in range(num_levels):
            if y_pred[i].shape[-1] != self.classes + 5:
                raise ValueError(
                    "y_pred should be a list with tensors of shape (None, None, None, classes + 5). "
                    f"Got y_pred[{i}].shape={tuple(y_pred[i].shape)}.  Does your model's `classes` "
                    "parameter match your losses `classes` parameter?"
                )

        x_shifts = []
        y_shifts = []
        expanded_strides = []
        outputs = []

        for i in range(num_levels):
            output = y_pred[i]

            grid_shape = tf.shape(output)[1:3]
            stride = input_shape[0] / tf.cast(grid_shape[0], tf.float32)

            grid_x, grid_y = tf.meshgrid(
                tf.range(grid_shape[1]), tf.range(grid_shape[0])
            )
            grid = tf.cast(
                tf.reshape(tf.stack((grid_x, grid_y), 2), (1, -1, 2)), tf.float32
            )

            output = tf.reshape(
                output, [tf.shape(y_pred[i])[0], grid_shape[0] * grid_shape[1], -1]
            )
            output_xy = (output[..., :2] + grid) * stride
            output_wh = tf.exp(output[..., 2:4]) * stride
            output = tf.concat([output_xy, output_wh, output[..., 4:]], -1)

            x_shifts.append(grid[..., 0])
            y_shifts.append(grid[..., 1])
            expanded_strides.append(tf.ones_like(grid[..., 0]) * stride)
            outputs.append(output)
        
        x_shifts = tf.concat(x_shifts, 1)
        y_shifts = tf.concat(y_shifts, 1)
        expanded_strides = tf.concat(expanded_strides, 1)
        outputs = tf.concat(outputs, 1)

        tf.debugging.assert_equal(tf.shape(x_shifts), [1, 5376])
        tf.debugging.assert_equal(tf.shape(y_shifts), [1, 5376])
        tf.debugging.assert_equal(tf.shape(expanded_strides), [1, 5376])
        tf.debugging.assert_equal(tf.shape(outputs), [tf.shape(y_true)[0], 5376, 25])

        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4:5]
        cls_preds = outputs[:, :, 5:]

        nlabel = tf.reduce_sum(
            tf.cast(tf.reduce_sum(y_true, -1) > 0, K.dtype(outputs)), -1
        )
        total_num_anchors = tf.shape(outputs)[1]

        num_fg = 0.0
        loss_obj = 0.0
        loss_cls = 0.0
        loss_iou = 0.0

        def loop_body(b, num_fg, loss_iou, loss_obj, loss_cls):
            num_gt = tf.cast(nlabel[b], tf.int32)
            
            gt_bboxes_per_image = y_true[b][:num_gt, :4]
            gt_classes = y_true[b][:num_gt, 4]

            bboxes_preds_per_image = bbox_preds[b]
            obj_preds_per_image = obj_preds[b]
            cls_preds_per_image = cls_preds[b]
            
            tf.debugging.assert_equal(tf.shape(gt_bboxes_per_image), [num_gt, 4])
            tf.debugging.assert_equal(tf.shape(gt_classes), [num_gt])
            tf.debugging.assert_equal(tf.shape(bboxes_preds_per_image), [5376, 4])
            tf.debugging.assert_equal(tf.shape(obj_preds_per_image), [5376, 1])
            tf.debugging.assert_equal(tf.shape(cls_preds_per_image), [5376, 20])

            gt_bboxes_per_image = tf.ensure_shape(gt_bboxes_per_image, [None, 4])
            bboxes_preds_per_image = tf.ensure_shape(bboxes_preds_per_image, [None, 4])
            obj_preds_per_image = tf.ensure_shape(obj_preds_per_image, [None, 1])
            cls_preds_per_image = tf.ensure_shape(cls_preds_per_image, [None, self.classes])

            def f1():
                num_fg_img = tf.cast(tf.constant(0), K.dtype(outputs))
                cls_target = tf.cast(tf.zeros((0, self.classes)), K.dtype(outputs))
                reg_target = tf.cast(tf.zeros((0, 4)), K.dtype(outputs))
                obj_target = tf.cast(tf.zeros((total_num_anchors, 1)), K.dtype(outputs))
                fg_mask = tf.cast(tf.zeros(total_num_anchors), tf.bool)
                return num_fg_img, cls_target, reg_target, obj_target, fg_mask

            def f2():
                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg_img,
                ) = self.get_assignments(
                    gt_bboxes_per_image,
                    gt_classes,
                    bboxes_preds_per_image,
                    obj_preds_per_image,
                    cls_preds_per_image,
                    x_shifts,
                    y_shifts,
                    expanded_strides,
                    self.classes,
                    num_gt,
                    total_num_anchors,
                )
                reg_target = tf.cast(
                    tf.gather_nd(
                        gt_bboxes_per_image, tf.reshape(matched_gt_inds, [-1, 1])
                    ),
                    K.dtype(outputs),
                )
                cls_target = tf.cast(
                    tf.one_hot(tf.cast(gt_matched_classes, tf.int32), self.classes)
                    * tf.expand_dims(pred_ious_this_matching, -1),
                    K.dtype(outputs),
                )
                obj_target = tf.cast(tf.expand_dims(fg_mask, -1), K.dtype(outputs))
                return num_fg_img, cls_target, reg_target, obj_target, fg_mask

            num_fg_img, cls_target, reg_target, obj_target, fg_mask = tf.cond(
                tf.equal(num_gt, 0), f1, f2
            )
            num_fg += num_fg_img
            loss_iou += tf.math.reduce_sum(
                self.box_loss(
                    reg_target, tf.boolean_mask(bboxes_preds_per_image, fg_mask)
                )
            loss_obj += tf.math.reduce_sum(
                self.objectness_loss(obj_target, obj_preds_per_image)
            )
            loss_cls += tf.math.reduce_sum(
                self.classification_loss(
                    cls_target,
                    tf.boolean_mask(cls_preds_per_image, fg_mask)
                )
            )
            return b + 1, num_fg, loss_iou, loss_obj, loss_cls

        _, num_fg, loss_iou, loss_obj, loss_cls = tf.while_loop(
            lambda b, *args: b < tf.cast(tf.shape(outputs)[0], tf.int32),
            loop_body,
            [0, num_fg, loss_iou, loss_obj, loss_cls],
        )

        self.classification_loss_metric.update_state(loss_cls/num_fg)
        self.box_loss_metric.update_state(loss_iou/num_fg)
        self.objectness_loss_metric.update_state(loss_obj/num_fg)

        num_fg = tf.cast(tf.maximum(num_fg, 1), K.dtype(outputs))
        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls

        return loss / num_fg

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
            target="center_xywh",
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
        _ = self._backward(y_training_target, y_pred, input_shape=x.shape[1:3])

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
