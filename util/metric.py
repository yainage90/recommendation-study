from .models import Metrics
import numpy as np

class MetricCalculator:
    def metrics(
        self,
        true_items: list[float],
        pred_items: list[float],
        true_user2items: dict[int, list[int]],
        pred_user2items: dict[int, list[int]],
        k: int,
    ) -> Metrics:
        rmse = self.rmse(true_items, pred_items)
        precision_at_k = self.mean_precision_at_k(true_user2items, pred_user2items, k=k)
        recall_at_k = self.mean_recall_at_k(true_user2items, pred_user2items, k=k)

        return Metrics(
            rmse=rmse,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k
        )
    
    def rmse(
        self,
        true_items: list[float],
        pred_items: list[float],
    ) -> float:
        return np.sqrt(np.mean([(true_items[i] - pred_items[i]) ** 2 for i in range(len(pred_items))]))

    def mean_precision_at_k(
        self,
        true_user2items: list[int],
        pred_user2items: list[int],
        k: int,
    ):
        result = 0
        for user_id in true_user2items:
            result += self._precision_at_k(
                true_user2items[user_id],
                pred_user2items[user_id],
                k
            )
        result /= len(pred_user2items.keys())

        return result

    def _precision_at_k(
        self,
        true_items: list[int],
        pred_items: list[int],
        k: int,
    ):
        return len(set(true_items).intersection(pred_items[:k])) / k

    def mean_recall_at_k(
        self,
        true_user2items: dict[int, list[int]],
        pred_user2items: dict[int, list[int]],
        k: int,
    ):
        result = 0
        for user_id in true_user2items:
            result += self._recall_at_k(
                true_user2items[user_id],
                pred_user2items[user_id],
                k
            )
        result /= len(pred_user2items.keys())

        return result
            

    def _recall_at_k(
        self,
        true_items: list[int],
        pred_items: list[int],
        k: int,
    ):
        if not true_items:
            return 0.0
        
        return len(set(true_items).intersection(pred_items[:k])) / len(true_items)
