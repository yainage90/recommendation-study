from abc import ABC, abstractmethod
from util.models import Dataset, RecommendResult
from util.data import DataLoader
from util.metric import MetricCalculator


class BaseRecommender(ABC):

    @abstractmethod
    def recommend(self, dataset: Dataset, k: int, **kwargs) -> RecommendResult:
        raise NotImplementedError()

    def run_sample(self):
        dataset = DataLoader().load()
        recommendations = self.recommend(dataset, k=10)

        metrics = MetricCalculator().metrics(
            true_items=dataset.test.rating.to_list(),
            pred_items=recommendations.rating.to_list(),
            true_user2items=dataset.test_user2items,
            pred_user2items=recommendations.user2items,
            k=10
        )
        
        print(metrics)
