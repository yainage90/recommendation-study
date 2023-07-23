import os
import os
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import shutil
import pandas as pd
from .models import Dataset


class DataLoader:
    
    def __init__(
        self,
        data_dir: str = f'{os.environ.get("HOME")}/workspace/recommendation-study/datasets/movie_lens',
        num_users: int = 1000,
        num_test_items: int = 5,
    ):
        self.data_dir = data_dir
        self.num_users = num_users
        self.num_test_items = num_test_items

    def load(self):
        self._download()

        df, movie_df = self._load()
        train_df, test_df = self._split_data(df)

        test_user2items = test_df[test_df['rating'] >= 4].groupby('user_id').agg({'movie_id': list})['movie_id'].to_dict()

        return Dataset(train_df, test_df, test_user2items, movie_df)

    def _load(self):
        m_cols = ['movie_id', 'title', 'genre']
        movie_df = pd.read_csv(f'{self.data_dir}/movies.dat', names=m_cols, sep='::', encoding='latin-1', engine='python')
        movie_df['genre'] = movie_df['genre'].apply(lambda x: x.split('|'))

        t_cols = ['user_id', 'movie_id', 'tag', 'timestamp']
        tag_df = pd.read_csv(f'{self.data_dir}/tags.dat', names=t_cols, sep='::', engine='python')
        tag_df['tag'] = tag_df['tag'].apply(lambda x: str(x).lower())

        movie_tags = tag_df.groupby('movie_id').agg({'tag': list})
        movie_df = movie_df.merge(movie_tags, on='movie_id', how='left')

        r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
        rating_df = pd.read_csv(f'{self.data_dir}/ratings.dat', names=r_cols, sep='::', engine='python')

        valid_user_ids = sorted(rating_df.user_id.unique())[:self.num_users]
        rating_df = rating_df[rating_df.user_id.isin(valid_user_ids)]
        df = rating_df.merge(movie_df, on='movie_id')

        return df, movie_df

    def _split_data(self, df: pd.DataFrame):
        df['rating_order'] = df.groupby('user_id')['timestamp'].rank(ascending=False, method='first')

        train_df = df[df['rating_order'] > self.num_test_items]
        test_df = df[df['rating_order'] <= self.num_test_items]

        train_df = train_df.drop(columns='rating_order')
        test_df = test_df.drop(columns='rating_order')

        return train_df, test_df


    def _download(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"{self.data_dir} 디렉터리 생성됨.")

            http_response = urlopen('https://files.grouplens.org/datasets/movielens/ml-10m.zip')

            with ZipFile(BytesIO(http_response.read())) as zip:
                for filename in zip.namelist():
                    if filename.endswith('.dat'):
                        save_name = filename.split('/')[-1]
                        source = zip.open(filename)
                        with open(os.path.join(self.data_dir, save_name), "wb") as target:
                            shutil.copyfileobj(source, target)
    