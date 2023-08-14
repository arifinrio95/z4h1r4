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
        try:
            self.desc_num = self.df.describe(exclude=['object'])
        except Exception as e:
            self.desc_num = None
        try:
            self.desc_obj = self.df.describe(include=['object'])
        except Exception as e:
            self.desc_obj = None

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
        
        if self.desc_num:
            st.dataframe(self.desc_num)
        if self.desc_obj:
            st.dataframe(self.desc_obj)
            
        ## missing values
        missing_data = self.df.isnull().sum()

        # Create a new DataFrame with required information
        missing_info = pd.DataFrame({
            'Column names':
            missing_data.index,
            'Percentage of missing values': ((missing_data / self.r) * 100).apply(lambda x: f'{x:.2f} %'),
            'Total missing values':
            missing_data
        })

        # Filter out columns with no missing values
        st.write("Missing Values")
        self.missing_df = missing_info.loc[
            missing_info['Total missing values'] > 0]
        st.dataframe(self.missing_df)
        
    
    
    def barplot(self):
        def format_value(value):
            if value_format == 'K':
                value = value / 1_000
                suffix = 'K'
            elif value_format == 'Mn':
                value = value / 1_000_000
                suffix = 'Mn'
            elif value_format == 'Bn':
                value = value / 1_000_000_000
                suffix = 'Bn'
            else:
                suffix = ''
            
            decimal_format = f'{{:.{decimal_places}f}}'
            return decimal_format.format(value) + suffix
            
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
            'Ocean Breeze', 'Sunset Serenity', 'Enchanted Forest', 'Fruit Sorbet', 'Cosmic Nebula', 'Biscuits & Chocolate', 'Color-Blind Safe'
        ]

        color_option = right_column.selectbox('Select Bar Color:',
                                              color_options)

        color_blind_safe_palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
            "#e377c2", "#7f7f7f"
        ]
        
        ocean_breeze_palette = ['#89c4f4', '#4d8fd5', '#204a87', '#118a8b', '#00b8a9', '#70ad47', '#dbb994', '#bdd9d5']
        sunset_serenity_palette = ['#ff9a8b', '#f4729a', '#b785c4', '#68a9cf', '#8cc890', '#fbd45b', '#7d7d7d', '#e8e1d1']
        enchanted_forest_palette = ['#356f3c', '#678a4c', '#147048', '#70451b', '#8e3c36', '#b3573d', '#8f9394', '#f2ead5']
        fruit_sorbet_palette = ['#ff5b77', '#ff9347', '#ffcc29', '#8dc63f', '#5975c9', '#835f9b', '#fff0d7', '#d8d8d8']
        cosmic_nebula_palette = ['#190b59', '#4d0579', '#c40473', '#f72c25', '#ff8800', '#ffd100', '#f7dcdc', '#000000']

        
        biscuits_chocolate_palette = [
            "#a67a5b", "#dcb697", "#89573e", "#f0d9b5", "#4b3423", "#f5e1c7",
            "#725440", "#c19a6b"
        ]

        color_mapping = {
            'Ocean Breeze': ocean_breeze_palette,
            'Sunset Serenity': sunset_serenity_palette,
            'Enchanted Forest': enchanted_forest_palette,
            'Fruit Sorbet': fruit_sorbet_palette,
            'Cosmic Nebula':cosmic_nebula_palette,
            'Biscuits & Chocolate': biscuits_chocolate_palette,
            'Color-Blind Safe': color_blind_safe_palette,
        }

        color_pal = color_mapping[color_option]
        # Get the top 10 categories
        top_categories = self.df[column].value_counts().nlargest(
            int(top_n)).index

        # Filter the data for the top 10 categories
        top_categories_data = self.df.loc[self.df[column].isin(
            top_categories)]
        
        sort_option = right_column.selectbox('Sort By:',
                                            ['Value', 'Category'])
        value_format = left_column.selectbox(
            'Select Value Format:',
            ['Normal', 'K', 'Mn', 'Bn'])
        decimal_places = right_column.selectbox(
            'Select Number of Decimal Places:',
            ['0', '1', '2', '3', '4'])
        
        if sort_option == 'Value':
            if chart_type!='Simple' and y_column:
                order = top_categories_data.groupby(column).agg({
                    y_column: aggregation_method
                }).sort_values(by=y_column, ascending=False).index
            else:
                order = top_categories
                
        elif sort_option == 'Category':
            order = sorted(top_categories_data[column].unique())

        if not y_column and chart_type != 'Simple':
            st.warning(
                'Please select a Numerical Column for chart types other than Single.'
            )
            return

        # Create a countplot using the custom theme palette
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
                data_to_plot = top_categories_data.groupby(column).size().reset_index(
                    name=y_column)
            else:
                data_to_plot = top_categories_data.groupby(column)[y_column].agg(
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
        x_label_size = plt.xticks()[1][0].get_size()
        y_label_size = plt.yticks()[1][0].get_size()
        if chart_type == 'Grouped':
            for p in ax.patches:
                value = p.get_height() if orientation == 'Vertical' else p.get_width()
                formatted_value = format_value(value)
                if orientation == 'Vertical':
                    ax.annotate(
                        formatted_value,
                        (p.get_x() + p.get_width() / 2., value),
                        ha='center',
                        va='baseline',
                        fontsize=x_label_size)
                elif orientation == 'Horizontal':
                    ax.annotate(
                        formatted_value,
                        (value, p.get_y() + p.get_height() / 2.),
                        ha='left',
                        va='center',
                        fontsize=y_label_size)

        title = f'{chart_type} Bar Plot of {column}'
        if chart_type != 'Simple':
            title += f' using {aggregation_method} of {y_column}'
        if orientation == 'Horizontal':
            title += ' (Horizontal Orientation)'
        else:
            title += ' (Vertical Orientation)'

        if orientation == 'Vertical':
            # Rotate x-axis labels for better readability (optional)
            plt.xticks(range(len(order)), order, rotation=90, ha="right")
        # else:
        #     plt.xticks(range(len(order)), order)
            
        plt.title(title, fontsize=16, fontweight="bold")
        plt.xlabel('Value' if y_column else 'Count', fontsize=12)
        plt.ylabel(column, fontsize=12)
        sns.despine(left=True, bottom=True)
        st.pyplot(plt)
