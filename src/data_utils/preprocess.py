import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

COLS_DICT = {
    'carat': 'carat',
    'cut': 'cut',
    'color': 'color',
    'clarity': 'clarity',
    'depth': 'depth',
    'table': 'table',
    'price': 'price',
    'x': 'x',
    'y': 'y',
    'z': 'z'
    }
PCA_FEATURE_NAME = 'dimension_pca'


class DataProcessor:
    def __init__(self,
                 reduce_dimensions: bool,
                 headers_dict: dict = COLS_DICT,
                 ) -> None:
        if not all(col_name in headers_dict.keys()
                   for col_name in ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z']):
            raise KeyError(('Columns name dictionary must have all the required columns declared. '
                            'Required columns names to be declared are: \'carat\', \'cut\', \'color\', \'clarity\', '
                            '\'depth\', \'table\', \'price\', \'x\', \'y\', \'z\''))
        self.headers_dict = headers_dict

        # Define custom order for cut categories
        self.cut_custom_order = {'Ideal': 1, 'Premium': 2, 'Very Good': 3, 'Good': 4, 'Fair': 5}
        # Define custom order for color categories
        self.color_custom_order = {'D': 1, 'E': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7}
        # Define custom order for clarity categories
        self.clarity_custom_order = {'IF': 1, 'VVS1': 2, 'VVS2': 3, 'VS1': 4, 'VS2': 5, 'SI1': 6, 'SI2': 7, 'I1': 8}
        
        self.perform_pca = reduce_dimensions
        # Attribute to store the pca compressor and load it afterward
        self.pca = None
        # Dictionary to store calculated mean and standard deviation for inference
        self.mean_std_dict = None

        if self.perform_pca:
            self.feature_cols = [self.headers_dict['depth'],
                                 self.headers_dict['table'],
                                 self.headers_dict['cut'],
                                 self.headers_dict['color'],
                                 self.headers_dict['clarity'],
                                 PCA_FEATURE_NAME,
                                 ]
        else:
            self.feature_cols = [self.headers_dict['depth'],
                                 self.headers_dict['table'],
                                 self.headers_dict['cut'],
                                 self.headers_dict['color'],
                                 self.headers_dict['clarity'],
                                 self.headers_dict['x'],
                                 self.headers_dict['y'],
                                 self.headers_dict['z'],
                                 self.headers_dict['carat'],
                                 ]
        self.target_col = [self.headers_dict['price']]

    @staticmethod
    def load_data(filepath: str,
                  sep: str = ',',
                  header: int = 0,
                  ) -> pd.DataFrame:
        return pd.read_csv(filepath,
                           sep=sep,
                           header=header,
                           )

    def process_data(self,
                     df: pd.DataFrame,
                     ) -> pd.DataFrame:
        df = self.clean_data(df=df)
        df = self.convert_categorical(df=df)
        if self.perform_pca:
            df = self.reduce_dimensions(df=df)
        df = self.normalize_values(df=df)

        return df

    def clean_data(self,
                   df: pd.DataFrame,
                   ) -> pd.DataFrame:
        # Delete negative prices
        df = df[df[self.headers_dict['price']] >= 0]

        # Delete negative dimensions
        df = df[(df[self.headers_dict['x']] > 0) &
                (df[self.headers_dict['y']] > 0) &
                (df[self.headers_dict['z']] > 0)]

        # Delete negative carat values
        df = df[df[self.headers_dict['carat']] > 0]

        # Delete wrong cut values
        df = df[df[self.headers_dict['cut']].isin(['Ideal', 'Premium', 'Very Good', 'Good', 'Fair'])]

        # Delete wrong color values
        df = df[df[self.headers_dict['color']].isin(['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                                                     'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                                                     'V', 'W', 'X', 'Y', 'Z'])]
        
        # Delete wrong clarity values
        df = df[df[self.headers_dict['clarity']].isin(['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'])]

        # Delete wrong table and depth values
        df = df[(df[self.headers_dict['table']] > 0) & (df[self.headers_dict['table']] < 100)]
        df = df[(df[self.headers_dict['depth']] > 0) & (df[self.headers_dict['depth']] < 100)]

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        return df
    
    def convert_categorical(self,
                            df: pd.DataFrame,
                            ) -> pd.DataFrame:
        # Convert categorical variables into numerical with the order predefined in the constructor method
        df[self.headers_dict['cut']] = df[self.headers_dict['cut']].map(self.cut_custom_order)
        df[self.headers_dict['color']] = df[self.headers_dict['color']].map(self.color_custom_order)
        df[self.headers_dict['clarity']] = df[self.headers_dict['clarity']].map(self.clarity_custom_order)

        return df

    def reduce_dimensions(self,
                          df: pd.DataFrame,
                          pca: PCA = None,
                          ) -> pd.DataFrame:
        # Apply cube root transformation to the variable
        df['cbrt_carat'] = np.cbrt(df[[self.headers_dict['carat']]])
        
        selected_data = df[['cbrt_carat', 
                            self.headers_dict['x'], 
                            self.headers_dict['y'],
                            self.headers_dict['z']]]

        if not pca:
            # Initialize PCA with desired number of components
            self.pca = PCA(n_components=1)
            # Fit PCA to the selected features
            self.pca.fit(selected_data)

        # Transform the selected features to their principal components
        selected_data_pca = self.pca.transform(selected_data)

        df[PCA_FEATURE_NAME] = selected_data_pca

        return df

    def normalize_values(self,
                         df: pd.DataFrame,
                         ) -> pd.DataFrame:
        if not self.mean_std_dict:
            self.mean_std_dict = {}
            # Save normalization values for inference
            for col in df.columns:
                if col in self.feature_cols:
                    self.mean_std_dict[col] = {'mean': df[col].mean(), 'std': df[col].std()}

            # If the dictionary with mean and std dev values has not been initialized, compute the new values
            # Normalize values
            df[self.feature_cols] = (df[self.feature_cols] - df[self.feature_cols].mean()) / df[
                self.feature_cols].std()
        else:
            # If the dictionary with mean and std dev values has been loaded from file
            # Apply normalization using loaded mean and standard deviation
            for col in df.columns:
                if col in self.feature_cols:
                    mean = self.mean_std_dict[col]['mean']
                    std = self.mean_std_dict[col]['std']
                    df[col] = (df[col] - mean) / std

        return df
