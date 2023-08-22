import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

class AutoVizManual:
    def __init__(self, dataframe):
        self.df = dataframe
        self.numeric_cols = self.df.select_dtypes(['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns
        self.categorical_cols = self.df.select_dtypes(exclude=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns

    def histogram_density(self, column):
        fig, ax = plt.subplots()
        sns.histplot(self.df[column], kde=True)
        st.pyplot(fig)

    def scatter_regplot(self, col1, col2):
        fig, ax = plt.subplots()
        sns.regplot(x=col1, y=col2, data=self.df)
        st.pyplot(fig)

    def bar_chart(self, col, selected_numeric_col, selected_categorical_hue):
        fig, ax = plt.subplots()
        sns.barplot(x=col, y=selected_numeric_col, hue=selected_categorical_hue, data=self.df)
        st.pyplot(fig)

    def heatmap_correlation(self):
        corr = self.df[self.numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
        st.pyplot(fig)

    def chi_square(self):
        results = []
        for col1 in self.categorical_cols:
            for col2 in self.categorical_cols:
                if col1 != col2:
                    contingency = pd.crosstab(self.df[col1], self.df[col2])
                    chi2, p, _, _ = chi2_contingency(contingency)
                    results.append((col1, col2, chi2, p))
        st.write(pd.DataFrame(results, columns=["Column 1", "Column 2", "Chi2 Value", "P Value"]))

    def box_plot(self, selected_column, selected_category):
        fig, ax = plt.subplots()
        sns.boxplot(x=selected_category, y=selected_column, data=self.df)
        st.pyplot(fig)

    def pairplot(self, selected_columns):
        fig = sns.pairplot(self.df[selected_columns])
        st.pyplot(fig)

    def pie_chart(self, selected_category):
        fig, ax = plt.subplots()
        self.df[selected_category].value_counts().plot.pie(autopct='%1.1f%%')
        st.pyplot(fig)
