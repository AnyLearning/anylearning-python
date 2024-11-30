"""NanoDet object detection model implementation using ONNXRuntime.

This module provides a class for running NanoDet object detection models in ONNX format.
It handles model loading, preprocessing, inference and visualization of results.
"""

import cv2
import numpy as np
import argparse
import onnxruntime as ort
import math


class NanoDet:
    """NanoDet object detector class.

    This class implements the NanoDet object detection model with ONNX runtime.
    It handles model initialization, image preprocessing, inference and post-processing.

    Attributes:
        classes (list): List of class names
        num_classes (int): Number of object classes
        prob_threshold (float): Confidence threshold for detections
        iou_threshold (float): IoU threshold for NMS
        mean (ndarray): Mean values for image normalization
        std (ndarray): Standard deviation values for image normalization
        net (onnxruntime.InferenceSession): ONNX Runtime inference session
        input_shape (tuple): Model input shape (height, width)
        reg_max (int): Maximum regression value
        project (ndarray): Project array for regression
        strides (tuple): Feature map strides
        mlvl_anchors (list): Multi-level anchor boxes
        keep_ratio (bool): Whether to keep aspect ratio in preprocessing
    """

    def __init__(
        self, model_pb_path, label_path, prob_threshold=0.4, iou_threshold=0.3
    ):
        """Initialize NanoDet detector.

        Args:
            model_pb_path (str): Path to ONNX model file
            label_path (str): Path to class labels file
            prob_threshold (float): Confidence threshold for filtering detections
            iou_threshold (float): IoU threshold for NMS
        """
        self.classes = list(
            map(lambda x: x.strip(), open(label_path, "r").readlines())
        )
        self.num_classes = len(self.classes)
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold

        # Normalization parameters
        self.mean = np.array(
            [103.53, 116.28, 123.675], dtype=np.float32
        ).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(
            1, 1, 3
        )

        # Initialize ONNX Runtime session
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(model_pb_path, so)
        self.input_shape = (
            self.net.get_inputs()[0].shape[2],
            self.net.get_inputs()[0].shape[3],
        )

        # Model parameters
        self.reg_max = (
            int((self.net.get_outputs()[0].shape[-1] - self.num_classes) / 4)
            - 1
        )
        self.project = np.arange(self.reg_max + 1)
        self.strides = (8, 16, 32, 64)
        self.mlvl_anchors = []

        # Generate anchors
        for i in range(len(self.strides)):
            anchors = self._make_grid(
                (
                    math.ceil(self.input_shape[0] / self.strides[i]),
                    math.ceil(self.input_shape[1] / self.strides[i]),
                ),
                self.strides[i],
            )
            self.mlvl_anchors.append(anchors)
        self.keep_ratio = False

    def _make_grid(self, featmap_size, stride):
        """Generate grid of anchor points.

        Args:
            featmap_size (tuple): Feature map size (h, w)
            stride (int): Stride of feature map

        Returns:
            ndarray: Grid of anchor points
        """
        feat_h, feat_w = featmap_size
        shift_x = np.arange(0, feat_w) * stride
        shift_y = np.arange(0, feat_h) * stride
        xv, yv = np.meshgrid(shift_x, shift_y)
        xv = xv.flatten()
        yv = yv.flatten()
        return np.stack((xv, yv), axis=-1)

    def softmax(self, x, axis=1):
        """Apply softmax function.

        Args:
            x (ndarray): Input array
            axis (int): Axis to apply softmax

        Returns:
            ndarray: Softmax output
        """
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s

    def _normalize(self, img):
        """Normalize image.

        Args:
            img (ndarray): Input image

        Returns:
            ndarray: Normalized image
        """
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        return img

    def resize_image(self, srcimg, keep_ratio=True):
        """Resize image to model input size.

        Args:
            srcimg (ndarray): Source image
            keep_ratio (bool): Whether to keep aspect ratio

        Returns:
            tuple: (resized image, new height, new width, top padding, left padding)
        """
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_shape[0], int(
                    self.input_shape[1] / hw_scale
                )
                img = cv2.resize(
                    srcimg, (neww, newh), interpolation=cv2.INTER_AREA
                )
                left = int((self.input_shape[1] - neww) * 0.5)
                img = cv2.copyMakeBorder(
                    img,
                    0,
                    0,
                    left,
                    self.input_shape[1] - neww - left,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
            else:
                newh, neww = (
                    int(self.input_shape[0] * hw_scale),
                    self.input_shape[1],
                )
                img = cv2.resize(
                    srcimg, (neww, newh), interpolation=cv2.INTER_AREA
                )
                top = int((self.input_shape[0] - newh) * 0.5)
                img = cv2.copyMakeBorder(
                    img,
                    top,
                    self.input_shape[0] - newh - top,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
        else:
            img = cv2.resize(
                srcimg, self.input_shape, interpolation=cv2.INTER_AREA
            )
        return img, newh, neww, top, left

    def post_process(self, preds, scale_factor=1, rescale=False):
        """Post-process model predictions.

        Args:
            preds (ndarray): Raw model predictions
            scale_factor (float): Scale factor for bbox rescaling
            rescale (bool): Whether to rescale bboxes

        Returns:
            tuple: (bboxes, confidence scores, class IDs)
        """
        mlvl_bboxes = []
        mlvl_scores = []
        ind = 0
        for stride, anchors in zip(self.strides, self.mlvl_anchors):
            cls_score, bbox_pred = (
                preds[ind : (ind + anchors.shape[0]), : self.num_classes],
                preds[ind : (ind + anchors.shape[0]), self.num_classes :],
            )
            ind += anchors.shape[0]
            bbox_pred = self.softmax(
                bbox_pred.reshape(-1, self.reg_max + 1), axis=1
            )
            bbox_pred = np.dot(bbox_pred, self.project).reshape(-1, 4)
            bbox_pred *= stride

            nms_pre = 1000
            if nms_pre > 0 and cls_score.shape[0] > nms_pre:
                max_scores = cls_score.max(axis=1)
                topk_inds = max_scores.argsort()[::-1][0:nms_pre]
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                cls_score = cls_score[topk_inds, :]

            bboxes = self.distance2bbox(
                anchors, bbox_pred, max_shape=self.input_shape
            )
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(cls_score)

        mlvl_bboxes = np.concatenate(mlvl_bboxes, axis=0)
        if rescale:
            mlvl_bboxes /= scale_factor
        mlvl_scores = np.concatenate(mlvl_scores, axis=0)

        bboxes_wh = mlvl_bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes_wh[:, 2:4] - bboxes_wh[:, 0:2]  # xywh
        classIds = np.argmax(mlvl_scores, axis=1)
        confidences = np.max(mlvl_scores, axis=1)  # max_class_confidence

        indices = cv2.dnn.NMSBoxes(
            bboxes_wh.tolist(),
            confidences.tolist(),
            self.prob_threshold,
            self.iou_threshold,
        ).flatten()
        if len(indices) > 0:
            mlvl_bboxes = mlvl_bboxes[indices]
            confidences = confidences[indices]
            classIds = classIds[indices]
            return mlvl_bboxes, confidences, classIds
        else:
            print("Nothing detected")
            return np.array([]), np.array([]), np.array([])

    def distance2bbox(self, points, distance, max_shape=None):
        """Convert distance predictions to bounding boxes.

        Args:
            points (ndarray): Anchor points
            distance (ndarray): Distance predictions
            max_shape (tuple): Maximum shape for clipping

        Returns:
            ndarray: Predicted bounding boxes
        """
        # Handle empty arrays
        if points.size == 0 or distance.size == 0:
            return np.zeros((0, 4))
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def detect(self, srcimg):
        """Run detection on an image.

        Args:
            srcimg (ndarray): Source image

        Returns:
            ndarray: Image with detection visualizations
        """
        img, newh, neww, top, left = self.resize_image(
            srcimg, keep_ratio=self.keep_ratio
        )
        img = self._normalize(img)
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[
            0
        ].squeeze(axis=0)
        det_bboxes, det_conf, det_classid = self.post_process(outs)

        ratioh, ratiow = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        for i in range(det_bboxes.shape[0]):
            xmin, ymin, xmax, ymax = (
                max(int((det_bboxes[i, 0] - left) * ratiow), 0),
                max(int((det_bboxes[i, 1] - top) * ratioh), 0),
                min(int((det_bboxes[i, 2] - left) * ratiow), srcimg.shape[1]),
                min(int((det_bboxes[i, 3] - top) * ratioh), srcimg.shape[0]),
            )
            cv2.rectangle(
                srcimg, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=1
            )
            print(
                self.classes[det_classid[i]]
                + ": "
                + str(round(det_conf[i], 3))
            )
            cv2.putText(
                srcimg,
                self.classes[det_classid[i]]
                + ": "
                + str(round(det_conf[i], 3)),
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                thickness=1,
            )
        return srcimg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imgpath",
        type=str,
        default="helmet_jacket_10256.jpg",
        help="image path"
    )
    parser.add_argument(
        "--modelpath",
        type=str,
        default="hetmet_jacket_detection.onnx",
        help="onnx filepath"
    )
    parser.add_argument(
        "--classfile",
        type=str,
        default="class.names",
        help="classname filepath"
    )
    parser.add_argument(
        "--conf-threshold",
        default=0.4,
        type=float,
        help="class confidence"
    )
    parser.add_argument(
        "--nms-threshold",
        default=0.6,
        type=float,
        help="nms iou thresh"
    )
    args = parser.parse_args()

    source_image = cv2.imread(args.imgpath)
    detector = NanoDet(
        args.modelpath,
        args.classfile,
        prob_threshold=args.conf_threshold,
        iou_threshold=args.nms_threshold,
    )
    output_image = detector.detect(source_image)

    window_name = "Deep learning object detection in ONNXRuntime"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
