"""Official STU point metrics; project command-line evaluation lives in src.evaluate."""

import numpy as np
from sklearn.metrics import auc, average_precision_score, roc_curve


class PointOODMetricsCalculator:
    min_eval_distance = 2.5
    max_eval_distance = 50
    min_num_points_to_eval = 5

    def __init__(
        self,
    ):
        self.all_scores = []
        self.all_labels = []

    def update(self, points, scores, target):
        """Update the stored scores and labels with new data.

        Args:
            points (np.ndarray): Point cloud coordinates
            scores (np.ndarray): Anomaly scores (higher means more anomalous).
            target (np.ndarray): Ground truth labels
        """
        distances = np.linalg.norm(points, axis=1)
        # Process labels and apply distance mask
        inlier_labels = np.where(target != 0, 0, -1)
        processed_labels = np.where(target == 2, 1, inlier_labels)
        processed_labels = np.where(
            (distances > self.max_eval_distance) | (distances < self.min_eval_distance),
            -1,
            processed_labels,
        )
        ignore_mask = processed_labels != -1
        labels = processed_labels[ignore_mask]

        # Only evaluate if sufficient anomaly points
        if np.sum(labels) < self.min_num_points_to_eval:
            return
        if len(scores) != len(target):
            raise ValueError("Prediction and label count mismatch")

        prediction = scores[ignore_mask]
        self.all_scores.append(prediction)
        self.all_labels.append(labels)

    def compute_metrics(self):
        """Compute OOD detection metrics on accumulated data.

        Returns:
            dict: Metrics including AP, FPR95, AUROC, and optimal threshold.
        """
        if not self.all_scores:
            return {}

        targets = np.concatenate(self.all_labels, axis=0)
        predictions = np.concatenate(self.all_scores, axis=0)

        AP = average_precision_score(y_true=targets, y_score=predictions)
        roc_auc, fpr, threshold = self._calculate_auroc(predictions, targets)

        return {
            "AP": AP * 100,
            "FPR95": fpr * 100,
            "AUROC": roc_auc * 100,
            "threshold": threshold,
        }

    @staticmethod
    def _calculate_auroc(predictions, targets):
        fpr, tpr, thresholds = roc_curve(y_true=targets, y_score=predictions)
        roc_auc = auc(fpr, tpr)
        fpr_best = 0
        optimal_threshold = 0

        for tpr_val, fpr_val, thr in zip(tpr, fpr, thresholds):
            if tpr_val > 0.95:
                fpr_best = fpr_val
                optimal_threshold = thr
                break

        return roc_auc, fpr_best, optimal_threshold
