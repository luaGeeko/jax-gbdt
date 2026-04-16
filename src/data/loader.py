import pandas as pd
from typing import Optional
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

class CaliforniaHousingLoader:
    def __init__(self, test_size=0.2, random_state=42):
        self.data = fetch_california_housing()
        # create a DataFrame for easy handling
        self.df = pd.DataFrame(self.data.data, columns=self.data.feature_names)
        self.df['target'] = self.data.target
        self.test_size = test_size
        self.random_state = random_state

        # create data set splits
        self.get_splits()

    def get_splits(self):
        """Returns train/test splits."""
        X = self.df.drop(columns=['target'])
        y = self.df['target']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def get_test_samples(self, n: int = 1):
        # lets switch on stochastic testing
        if n > self.X_test.shape[0]:
            print(f"[DATALOADER] number of samples asked by user is more then test size. Stochastic testing is enabled!")
            samples = self.X_test.sample(n, replace=True)
        else:
            samples = self.X_test.sample(n) 
        return samples

    def get_random_samples(self, n=1):
        """Fetch N random samples for inference testing."""
        sample = self.df.sample(n)
        X_sample = sample.drop(columns=['target'])
        return X_sample

    @property
    def feature_names(self):
        return self.data.feature_names