import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DataAnalytics():
    def __init__(self, df):
        self.df = df
        self.missing_df = None
        self.desc_num = self.df.describe(exclude=['object'])
        self.desc_obj = self.df.describe(include=['object'])

        # get dataframe info
        schema_dict = self.df.dtypes.apply(lambda x: x.name).to_dict()
        self.schema_str = json.dumps(schema_dict)
        self.r, self.c = self.df.shape
        self.categorical_columns = self.df.select_dtypes(
            include=['object']).columns.tolist()
        self.numeric_columns = self.df.select_dtypes(
            include=['number']).columns.tolist()
        self.aggregation_methods = {
            'sum': np.sum,
            'mean': np.mean,
            'count': 'count',
            'max': np.max,
            'min': np.min,
        }

    def info(self):
        st.write(self.schema_str)
        st.write(f"Number of Rows : {self.r}, Number of Columns : {self.c}")

    def basic(self):
        ## descriptive statistics
        st.subheader("Data Information")
        st.write("Descriptive Statistics")
        st.dataframe(self.desc_num)
        st.dataframe(self.desc_obj)

        ## missing values
        missing_data = self.df.isnull().sum()

        # Create a new DataFrame with required information
        missing_info = pd.DataFrame({
            'Column names':
            missing_data.index,
            'Percentage of missing values': (missing_data / self.r) * 100,
            'Total missing values':
            missing_data
        })

        # Filter out columns with no missing values
        st.write("Missing Values")
        self.missing_df = missing_info.loc[
            missing_info['Total missing values'] > 0]
        st.dataframe(self.missing_df)

    def barplot(self):
        st.subheader("Bar Plot")
        left_column, right_column = st.columns(2)
        chart_type = left_column.selectbox(
            'Select Chart Type:',
            ['Simple', 'Grouped', 'Stacked', '100% Stacked'])
        column = right_column.selectbox(
            'Select a Categorical Column for Grouping Bar Plot:',
            self.categorical_columns)
        top_n = left_column.selectbox('Select Top N Categories:',
                                      ['5', '10', '20'])
        y_column = None
        aggregation_method = None

        if chart_type != 'Simple':
            y_column = left_column.selectbox('Select a Numeric Column:',
                                             self.numeric_columns,
                                             index=0)
            aggregation_method = right_column.selectbox(
                'Select Aggregation Method:',
                ['sum', 'mean', 'count', 'max', 'min'])

        aggregation_func = self.aggregation_methods[
            aggregation_method] if aggregation_method else None

        orientation = left_column.selectbox('Select Orientation:',
                                            ['Horizontal', 'Vertical'])

        # Opsi warna yang mudah dimengerti
        color_options = [
            'Color-Blind Safe', 'Earth Tone', 'Biscuits & Chocolate'
        ]

        color_option = right_column.selectbox('Select Bar Color:',
                                              color_options)

        color_blind_safe_palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
            "#e377c2", "#7f7f7f"
        ]
        earth_tone_palette = [
            "#8c5942", "#5c7c5d", "#a18f77", "#594c38", "#9f715b", "#7e827a",
            "#736553", "#4c5f6b"
        ]
        biscuits_chocolate_palette = [
            "#a67a5b", "#dcb697", "#89573e", "#f0d9b5", "#4b3423", "#f5e1c7",
            "#725440", "#c19a6b"
        ]

        color_mapping = {
            'Color-Blind Safe': color_blind_safe_palette,
            'Earth Tone': earth_tone_palette,
            'Biscuits & Chocolate': biscuits_chocolate_palette,
        }

        color_pal = color_mapping[color_option]

        sort_option = right_column.selectbox('Sort By:',
                                            ['None', 'Value', 'Category'])
        order = None
        if sort_option == 'Value' and y_column:
            order = self.df.groupby(column).agg({
                y_column: aggregation_method
            }).sort_values(by=y_column, ascending=False).index
        elif sort_option == 'Category':
            order = sorted(self.df[column].unique())

        if not y_column and chart_type != 'Simple':
            st.warning(
                'Please select a Numerical Column for chart types other than Single.'
            )
            return

        # Get the top 10 categories
        top_categories = self.df[column].value_counts().nlargest(
            int(top_n)).index

        # Filter the data for the top 10 categories
        top_categories_data = self.df.loc[self.df[column].isin(
            top_categories)]

        # Create a countplot using the biscuits and chocolate theme palette
        sns.set_palette(color_pal)

        if chart_type == 'Simple':
            # Handle Single chart type
            if orientation == 'Vertical':
                ax = sns.countplot(x=column,
                                   data=top_categories_data,
                                   palette=color_pal)
            elif orientation == 'Horizontal':
                ax = sns.countplot(y=column,
                                   data=top_categories_data,
                                   palette=color_pal)
        else:
            if aggregation_method == 'count':
                data_to_plot = self.df.groupby(column).size().reset_index(
                    name=y_column)
            else:
                data_to_plot = self.df.groupby(column)[y_column].agg(
                    aggregation_func).reset_index()
            y_value = y_column

            if chart_type == 'Grouped':
                if orientation == 'Vertical':
                    ax = sns.barplot(x=column,
                                     y=y_value,
                                     data=data_to_plot,
                                     order=order,
                                     palette=color_pal)
                elif orientation == 'Horizontal':
                    ax = sns.barplot(y=column,
                                     x=y_value,
                                     data=data_to_plot,
                                     order=order,
                                     palette=color_pal)
            elif chart_type == 'Stacked':
                data_to_plot.plot(kind='bar',
                                  x=column,
                                  y=y_value,
                                  stacked=True,
                                  palette=color_pal)
            elif chart_type == '100% Stacked':
                df_stacked = data_to_plot.groupby(column).apply(
                    lambda x: 100 * x / x.sum()).reset_index()
                df_stacked.plot(kind='bar',
                                x=column,
                                y=y_value,
                                stacked=True,
                                palette=color_pal)

        # Add value labels
        if chart_type == 'Grouped':
            for p in ax.patches:
                if orientation == 'Vertical':
                    ax.annotate(
                        f'{p.get_height():.2f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center',
                        va='baseline')
                elif orientation == 'Horizontal':
                    ax.annotate(
                        f'{p.get_width():.2f}',
                        (p.get_width(), p.get_y() + p.get_height() / 2.),
                        ha='left',
                        va='center')

        title = f'{chart_type} Bar Plot of {column}'
        if chart_type != 'Simple':
            title += f' using {aggregation_method} of {y_column}'
        if orientation == 'Horizontal':
            title += ' (Horizontal Orientation)'
        else:
            title += ' (Vertical Orientation)'

        plt.title(title, fontsize=16, fontweight="bold")
        plt.xlabel('Value' if y_column else 'Count', fontsize=12)
        plt.ylabel(column, fontsize=12)
        sns.despine(left=True, bottom=True)
        st.pyplot(plt)
