from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame
    test_user2items: dict[int, list[int]]
    item_content: pd.DataFrame


@dataclass(frozen=True)
class Metrics:
    rmse: float
    precision_at_k: float
    recall_at_k: float

    def __repr__(self) -> str:
        return f'rmse: {self.rmse:.3f}, precision@K: {self.precision_at_k:.3f}, recall@K: {self.recall_at_k:.3f}'


@dataclass(frozen=True)
class RecommendResult:
    rating: pd.DataFrame
    user2items: dict[int, list[int]]
