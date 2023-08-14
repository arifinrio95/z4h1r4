import pandas as pd
import numpy as np
import json

class DataAnalytics():

    def __init__(self, df, style):
        self.df = df

        # get dataframe info
        schema_dict = self.df.dtypes.apply(lambda x: x.name).to_dict()
        self.schema_str = json.dumps(schema_dict)
        self.r, self.c = self.df.shape

    def basic(self):
        ## descriptive statistics
        self.desc_num = self.df.describe(exclude=['o'])
        self.desc_obj = self.df.describe(include=['o'])

        ## missing values
        missing_data = self.df.isnull().sum()

        # Create a new DataFrame with required information
        missing_info = pd.DataFrame({
            'Column names': missing_data.index,
            'Percentage of missing values': (missing_data / self.r) * 100,
            'Total missing values': missing_data
        })

        # Filter out columns with no missing values
        self.missing_df = missing_info[missing_info['Total missing values'] > 0]
