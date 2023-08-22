import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


class DataViz():
    def __init__(self, dataloader):
        self.df = dataloader
        # self.df = pd.read_csv('train.csv')
        self.num_df = self.df.select_dtypes(include='number')
        self.cat_df = self.df.select_dtypes(exclude='number')

    def format_value(self, valu):
        '''
        This Code is used for formating Number Value
        '''
        if valu >= 1e9:
            return f"{valu/1e9:.1f} Bn"
        elif valu >= 1e6:
            return f"{valu/1e6:.1f} Mn"
        elif valu >= 1e3:
            return f"{valu/1e3:.1f} K"
        else:
            return f"{valu:.1f}"

    def visualization(self):
        st.title("Auto Data Viz & Insight by Ulik Data")

        tab1, tab2, tab3, tab4 = st.tabs([
            'Simple Business Viz', 'Simple Statistical Viz', 'Advanced Plot',
            'Auto Insight!'
        ])

        with tab1:
            if len(self.cat_df.columns) < 1:
                st.write("Your Data do not have any Categorical Values")
            else:
                ## BAR PLOT
                c1 = st.container()
                desc_col, img_col = c1.columns(2)

                desc_col.subheader("Bar Plot")

                # Widget selection for user input
                VAR = desc_col.selectbox('Bar Plot variable:',
                                         self.cat_df.columns.tolist())

                dc1, dc2 = desc_col.columns(2)

                selected_aggregator = dc1.selectbox(
                    "Select Aggregator:",
                    ['None'] + self.num_df.columns.tolist())
                aggregation_options = [
                    'sum', 'mean', 'median', 'No Aggregation'
                ]
                selected_aggregation = dc2.selectbox(
                    "Select Aggregation Function:", aggregation_options
                    if selected_aggregator != 'None' else ['No Aggregation'])
                selected_total_category = dc1.selectbox(
                    "Select # of Top Categories:", ['5', '10', '20'])
                sorting_options = [
                    'Ascending', 'Descending', 'None (Alphabetical Order)'
                ]
                selected_sorting = dc2.selectbox(
                    "Select Bar Plot Sorting Order:", sorting_options)

                # Check if the number of unique categories is greater than N
                if self.df[VAR].nunique() > int(selected_total_category):
                    # Get the top N categories
                    top_categories = self.df[VAR].value_counts().nlargest(
                        int(selected_total_category)).index.tolist()
                    # Create a new categorical variable that groups top N categories and labels the rest as 'Others'
                    self.df[f'{VAR}NEW'] = self.df[VAR].apply(
                        lambda x: x if x in top_categories else 'Others')
                else:
                    self.df[f'{VAR}NEW'] = self.df[VAR]

                # Apply selected aggregation function or counting
                if selected_aggregation == 'sum':
                    aggregated_data = self.df.groupby(
                        f'{VAR}NEW')[selected_aggregator].sum()
                    LABEL_PLOT = f'Sum of {selected_aggregator} by {VAR}'
                elif selected_aggregation == 'mean':
                    aggregated_data = self.df.groupby(
                        f'{VAR}NEW')[selected_aggregator].mean()
                    LABEL_PLOT = f'Mean of {selected_aggregator} by {VAR}'
                elif selected_aggregation == 'median':
                    aggregated_data = self.df.groupby(
                        f'{VAR}NEW')[selected_aggregator].median()
                    LABEL_PLOT = f'Median of {selected_aggregator} by {VAR}'
                else:
                    aggregated_data = self.df[f'{VAR}NEW'].value_counts()
                    LABEL_PLOT = f'Count of {VAR}'

                # Sort the data based on the selected sorting order
                if selected_sorting == 'Ascending':
                    aggregated_data = aggregated_data.sort_values(
                        ascending=True)
                elif selected_sorting == 'Descending':
                    aggregated_data = aggregated_data.sort_values(
                        ascending=False)
                else:
                    aggregated_data = aggregated_data.sort_index()

                # Insight 1: Identify the category with the highest value
                max_category = aggregated_data.idxmax()
                max_value = aggregated_data.max()

                # Create an aggregated bar plot using Plotly Express
                fig = px.bar(x=aggregated_data.index.astype(str),
                             y=aggregated_data,
                             labels={
                                 'y': LABEL_PLOT.replace(f' by {VAR}', ''),
                                 'x': VAR
                             },
                             title=LABEL_PLOT)

                with desc_col:
                    desc_col.write('##')
                    desc_col.markdown("#### :blue[Insight!]")
                    desc_col.write(
                        f"The category with the highest value is '{max_category}' with a value of {max_value:.2f}."
                    )

                with img_col:
                    # Create a plot using Plotly
                    img_col.plotly_chart(fig)

                st.write('---')

                ## PIE PLOT
                c2 = st.container()
                desc_col, img_col = c2.columns(2)

                desc_col.subheader("Pie Chart")
                # Widget selection for user input
                VAR2 = desc_col.selectbox('Pie Chart variable:',
                                          self.cat_df.columns.tolist())
                selected_style = desc_col.selectbox("Select Pie Chart Style:",
                                                    ['Pie', 'Doughnut'])

                dc1, dc2 = desc_col.columns(2)
                selected_aggregator = dc1.selectbox(
                    "Select Aggregator for Pie Chart:",
                    ['None'] + self.num_df.columns.tolist())
                aggregation_options = ['Sum', 'No Aggregation']
                selected_aggregation = dc2.selectbox(
                    "Select Aggregation Function  for Pie Chart:",
                    aggregation_options
                    if selected_aggregator != 'None' else ['No Aggregation'])

                # selected_total_category = dc1.selectbox(
                #     "Select # of Top Categories for Pie Chart:", ['5', '10', '20'])

                # Check if the number of unique categories is greater than 3
                if self.df[VAR2].nunique() > 3:
                    # Get the top N categories
                    top_categories = self.df[VAR2].value_counts().nlargest(
                        3).index.tolist()
                    # Create a new categorical variable that groups top N categories and labels the rest as 'Others'
                    self.df[f'{VAR2}NEW'] = self.df[VAR2].apply(
                        lambda x: x if x in top_categories else 'Others')
                else:
                    self.df[f'{VAR2}NEW'] = self.df[VAR2]

                # Apply selected aggregation function or counting
                if selected_aggregation == 'Sum':
                    aggregated_data = self.df.groupby(
                        f'{VAR2}NEW')[selected_aggregator].sum()
                    LABEL_PLOT = f'Sum of {selected_aggregator} by {VAR2}'
                else:
                    aggregated_data = self.df[f'{VAR2}NEW'].value_counts()
                    LABEL_PLOT = f'Percent Count of {VAR2}'

                # Calculate the total aggregated value
                total_aggregated_value = aggregated_data.sum()

                # Create an aggregated bar plot using Plotly Express
                if selected_style == 'Pie':
                    fig = px.pie(values=aggregated_data,
                                 names=aggregated_data.index.astype(str),
                                 title=LABEL_PLOT)
                else:
                    fig = px.pie(values=aggregated_data,
                                 names=aggregated_data.index.astype(str),
                                 hole=0.45,
                                 title=LABEL_PLOT)
                    # Add annotation for the total value in the center of the donut chart
                    fig.add_annotation(
                        text=
                        f"Total: {self.format_value(total_aggregated_value)}",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=20))

                with desc_col:
                    desc_col.write('##')
                    desc_col.markdown("#### :blue[Insight!]")
                    # Insight 1: Aggregated value of each category
                    desc_col.write("Aggregated Value of Each Category:")
                    for category, value in aggregated_data.items():
                        percentage = (value / total_aggregated_value) * 100
                        desc_col.write(
                            f"- Category '{category}': {value:.2f} ({percentage:.2f}%)"
                        )

                with img_col:
                    # Create a plot using Plotly
                    img_col.plotly_chart(fig)

                st.write('---')

                ## LINE CHART
                c3 = st.container()
                desc_col, img_col = c3.columns(2)

                desc_col.subheader("Line Chart")
                # Widget selection for user input
                VAR3 = desc_col.selectbox(
                    'Line Chart Horizontal Variable (Date / Category):',
                    self.cat_df.columns.tolist())
                VAR4 = desc_col.selectbox(
                    'Line Chart Vertical Variable (Value):',
                    self.num_df.columns.tolist())
                VAR5 = desc_col.selectbox(
                    'Line Chart Categorical Variable (Hue):', ['None'] +
                    [x for x in self.cat_df.columns.tolist() if x != VAR3])

                dc1, dc2 = desc_col.columns(2)
                # Streamlit checkbox for smooth line option
                smooth_line = dc1.checkbox("Smooth Line")

                # Streamlit checkbox for wide format data
                wide_format = dc2.checkbox("Wide Format Data")

                if VAR5 != 'None':
                    data = self.df[[VAR3, VAR4, VAR5]].copy()
                    # Check if the number of unique categories is greater than 3
                    if data[VAR5].nunique() > 3:
                        # Get the top N categories
                        top_categories = data[VAR5].value_counts().nlargest(
                            3).index.tolist()
                        # Create a new categorical variable that groups top N categories and labels the rest as 'Others'
                        data[VAR5] = data[VAR5].apply(
                            lambda x: x if x in top_categories else 'Others')
                else:
                    data = self.df[[VAR3, VAR4]].copy()

                df_to_plot = data.sort_values(
                    by=VAR4,
                    ascending=True).reset_index().drop(columns=['index'])

                if wide_format:
                    # Reshape the data for Plotly Express
                    df_to_plot = df_to_plot.stack().reset_index()

                try:
                    # Calculate insights
                    max_value = df_to_plot[VAR4].max()
                    min_value = df_to_plot[VAR4].min()
                    max_date = df_to_plot.loc[df_to_plot[VAR4] ==
                                              max_value][VAR3].values[0]
                    min_date = df_to_plot.loc[df_to_plot[VAR4] ==
                                              min_value][VAR3].values[0]

                    # Create a line chart to visualize monthly transaction trends
                    if VAR5 != 'None':
                        fig = px.line(
                            df_to_plot,
                            x=VAR3,
                            y=VAR4,
                            color=VAR5,
                            title=f"{VAR4} Trends by {VAR3}",
                            line_shape='spline' if smooth_line else 'linear')
                    else:
                        fig = px.line(
                            df_to_plot,
                            x=VAR3,
                            y=VAR4,
                            title=f"{VAR4} Trends by {VAR3}",
                            line_shape='spline' if smooth_line else 'linear')

                    with desc_col:
                        desc_col.write('##')
                        desc_col.markdown("#### :blue[Insight!]")
                        # Display insights
                        desc_col.write(
                            f"Maximum Value: {max_value} on {max_date}")
                        desc_col.write(
                            f"Minimum Value: {min_value} on {min_date}")

                        # Calculate trend direction
                        last_value = df_to_plot.iloc[-1][VAR4]
                        first_value = df_to_plot.iloc[0][VAR4]
                        TREND = "Stagnant"
                        if last_value > first_value:
                            TREND = "Positive"
                        elif last_value < first_value:
                            TREND = "Negative"

                        # Display trend direction insight
                        desc_col.write(
                            f"Trend Direction: {TREND.capitalize()} trend")

                    with img_col:
                        # Create a plot using Plotly
                        img_col.plotly_chart(fig)

                except:
                    with desc_col:
                        desc_col.write('##')
                        desc_col.markdown(
                            "Seems your data is not in a correct format, please uncheck Wide Data Format or crosscheck your data format"
                        )

                st.write('---')

                ## PIVOT TABLE
                c4 = st.container()
                desc_col, img_col = c4.columns(2)

                desc_col.subheader("Pivot Table")

                # Streamlit selectbox for rows, columns, and values
                row_options = [None] + self.cat_df.columns.tolist()
                col_options = [None] + self.cat_df.columns.tolist()
                value_options = [None] + self.num_df.columns.tolist()

                selected_rows = desc_col.selectbox("Select Rows:", row_options)
                selected_columns = desc_col.selectbox("Select Columns:",
                                                      col_options)
                if selected_rows is not None and selected_columns is not None:
                    selected_values = desc_col.selectbox(
                        "Select Values:", value_options)
                else:
                    selected_values = desc_col.selectbox(
                        "Select Values:", [None])
                if selected_values is None:
                    selected_value_adjustment = desc_col.selectbox(
                        "Select Value Adjustment:", [
                            None, 'percentage by rows',
                            'percentage by columns', 'percentage by all'
                        ])
                else:
                    selected_value_adjustment = desc_col.selectbox(
                        "Select Value Adjustment:", [
                            'sum', 'average', 'percentage by rows',
                            'percentage by columns', 'percentage by all'
                        ])

                # Create a custom pivot table using pandas
                if selected_rows is None and selected_columns is None:
                    pivot_table = pd.DataFrame({'Count': [self.df.shape[0]]})
                    pivot_table['Count'] = pivot_table['Count'].map(
                        self.format_value)
                elif selected_rows is not None and selected_columns is None:
                    pivot_table = self.df[selected_rows].value_counts(
                    ).reset_index()
                    pivot_table.columns = [selected_rows, 'Count']
                    pivot_table['Count'] = pivot_table['Count'].map(
                        self.format_value)
                elif selected_rows is None and selected_columns is not None:
                    pivot_table = self.df[selected_columns].value_counts(
                    ).reset_index()
                    pivot_table.columns = [selected_columns, 'Count']
                    pivot_table['Count'] = pivot_table['Count'].map(
                        self.format_value)
                elif selected_rows is not None and selected_columns is not None and selected_values is None:
                    if selected_value_adjustment == 'percentage by columns':
                        pivot_table = pd.crosstab(
                            index=self.df[selected_rows],
                            columns=self.df[selected_columns])
                        # Calculate percentages by columns
                        pivot_table = pivot_table.div(pivot_table.sum(axis=0),
                                                      axis=1) * 100
                        # Calculate row and column totals
                        row_totals = pivot_table.sum(axis=1)
                        column_totals = pivot_table.sum(axis=0)

                        # Create a DataFrame for the column totals
                        column_totals_df = pd.DataFrame(
                            [column_totals],
                            columns=pivot_table.columns,
                            index=['All'])

                        # Create a DataFrame for the row totals
                        row_totals_df = pd.DataFrame(row_totals,
                                                     columns=['All'])

                        # Concatenate the crosstab table, row totals, and column totals
                        pivot_table = pd.concat(
                            [pivot_table, column_totals_df])
                        pivot_table = pd.concat([pivot_table, row_totals_df],
                                                axis=1)
                        pivot_table = pivot_table.applymap(
                            lambda x: f"{x:.2f}%" if not np.isnan(x) else "-")
                    elif selected_value_adjustment == 'percentage by rows':
                        pivot_table = pd.crosstab(
                            index=self.df[selected_rows],
                            columns=self.df[selected_columns])
                        # Calculate percentages by rows
                        pivot_table = pivot_table.div(pivot_table.sum(axis=1),
                                                      axis=0) * 100
                        # Calculate row and column totals
                        row_totals = pivot_table.sum(axis=1)
                        column_totals = pivot_table.sum(axis=0)

                        # Create a DataFrame for the column totals
                        column_totals_df = pd.DataFrame(
                            [column_totals],
                            columns=pivot_table.columns,
                            index=['All'])

                        # Create a DataFrame for the row totals
                        row_totals_df = pd.DataFrame(row_totals,
                                                     columns=['All'])

                        # Concatenate the crosstab table, row totals, and column totals
                        pivot_table = pd.concat(
                            [pivot_table, column_totals_df])
                        pivot_table = pd.concat([pivot_table, row_totals_df],
                                                axis=1)
                        pivot_table = pivot_table.applymap(
                            lambda x: f"{x:.2f}%" if not np.isnan(x) else "-")
                    elif selected_value_adjustment == 'percentage by all':
                        pivot_table = pd.crosstab(
                            index=self.df[selected_rows],
                            columns=self.df[selected_columns])

                        all_values = pivot_table.values.sum()

                        # Calculate row and column totals
                        row_totals = pivot_table.sum(axis=1)
                        column_totals = pivot_table.sum(axis=0)

                        # Create a DataFrame for the column totals
                        column_totals_df = pd.DataFrame(
                            [column_totals],
                            columns=pivot_table.columns,
                            index=['All'])

                        # Create a DataFrame for the row totals
                        row_totals_df = pd.DataFrame(row_totals,
                                                     columns=['All'])

                        # Concatenate the crosstab table, row totals, and column totals
                        crosstab_with_totals = pd.concat(
                            [pivot_table, column_totals_df])
                        crosstab_with_totals = pd.concat(
                            [crosstab_with_totals, row_totals_df], axis=1)
                        crosstab_with_totals = crosstab_with_totals / all_values * 100

                        # Format the percentages
                        pivot_table = crosstab_with_totals.applymap(
                            lambda x: f"{x:.2f}%" if not np.isnan(x) else "-")
                    else:
                        pivot_table = pd.crosstab(
                            index=self.df[selected_rows],
                            columns=self.df[selected_columns],
                            margins=True)
                else:
                    if selected_value_adjustment == 'sum':
                        pivot_table = pd.pivot_table(self.df,
                                                     index=selected_rows,
                                                     columns=selected_columns,
                                                     values=selected_values,
                                                     aggfunc='sum',
                                                     fill_value=0,
                                                     margins=True)
                        pivot_table = pivot_table.applymap(self.format_value)
                    elif selected_value_adjustment == 'average':
                        pivot_table = pd.pivot_table(self.df,
                                                     index=selected_rows,
                                                     columns=selected_columns,
                                                     values=selected_values,
                                                     aggfunc='mean',
                                                     fill_value=0,
                                                     margins=True)
                        pivot_table = pivot_table.applymap(self.format_value)
                    elif selected_value_adjustment == 'percentage by rows':
                        pivot_table = pd.pivot_table(self.df,
                                                     index=selected_rows,
                                                     columns=selected_columns,
                                                     values=selected_values,
                                                     aggfunc='sum',
                                                     fill_value=0,
                                                     margins=True)
                        pivot_table = pivot_table.div(pivot_table.iloc[:, -1],
                                                      axis=0) * 100
                        pivot_table = pivot_table.applymap(
                            lambda x: f"{x:.2f}%" if not np.isnan(x) else "-")
                    elif selected_value_adjustment == 'percentage by columns':
                        pivot_table = pd.pivot_table(self.df,
                                                     index=selected_rows,
                                                     columns=selected_columns,
                                                     values=selected_values,
                                                     aggfunc='sum',
                                                     fill_value=0,
                                                     margins=True)
                        pivot_table = pivot_table.div(pivot_table.iloc[-1, :],
                                                      axis=1) * 100
                        pivot_table = pivot_table.applymap(
                            lambda x: f"{x:.2f}%" if not np.isnan(x) else "-")
                    else:
                        pivot_table = pd.pivot_table(self.df,
                                                     index=selected_rows,
                                                     columns=selected_columns,
                                                     values=selected_values,
                                                     aggfunc='sum',
                                                     fill_value=0,
                                                     margins=True)
                        pivot_table = pivot_table / pivot_table.iloc[
                            -1, :-1].sum() * 100
                        pivot_table = pivot_table.applymap(
                            lambda x: f"{x:.2f}%" if not np.isnan(x) else "-")

                # Display the pivot table using Streamlit
                img_col.dataframe(pivot_table, use_container_width=True)

        with tab2:
            if len(self.num_df.columns) < 1:
                st.write("Your Data do not have any Numerical Values")
            else:
                ## HISTOGRAM / DENSITY PLOT
                c1 = st.container()
                desc_col, img_col = c1.columns(2)

                desc_col.subheader("Histogram/Density Plot")

                # Widget selection for user input
                VAR = desc_col.selectbox('Histogram/density variable:',
                                         self.num_df.columns.tolist())
                BIN = desc_col.selectbox('Number of Bins:', ['10', '25', '50'])
                TYPE = desc_col.selectbox('Type:', ['Histogram', 'Density'])

                NORM = None if TYPE == 'Histogram' else 'probability density'
                DENSITY = False if TYPE == 'Histogram' else True

                # Create Plot using user input
                fig = px.histogram(self.num_df[VAR],
                                   nbins=int(BIN),
                                   histnorm=NORM)
                f = fig.full_figure_for_development(warn=False)

                xbins = f.data[0].xbins
                plotbins = list(
                    np.arange(start=xbins['start'],
                              stop=xbins['end'] + xbins['size'],
                              step=xbins['size']))
                data_for_hist = [val for val in f.data[0].x if val is not None]
                counts, bins = np.histogram(data_for_hist,
                                            bins=plotbins,
                                            density=DENSITY)
                IDX = np.argmax(counts)
                s, e = bins[IDX], bins[IDX + 1]
                COUNTS = max(counts)
                with desc_col:
                    desc_col.write('##')
                    desc_col.markdown("#### :blue[Insight!]")
                    desc_col.markdown(
                        f"**Highest Bins**\t: {s:.2f} to {e-0.1:.2f}")
                    desc_col.markdown(
                        f"**Highest Values**\t\t: {counts[IDX]:.2f}")

                with img_col:
                    # Create a plot using Plotly
                    img_col.plotly_chart(fig)

                st.write('---')

                ## BOX PLOT
                c2 = st.container()
                desc_col, img_col = c2.columns(2)

                desc_col.subheader("Box Plot")

                # Widget selection for user input
                VAR2 = desc_col.selectbox('Box Plot Variable:',
                                          self.num_df.columns)

                # Create Plot using user input
                fig = px.box(self.num_df[VAR2])

                # Calculate outliers using the Interquartile Range (IQR) method
                q1 = np.percentile(self.num_df[VAR2].dropna(), 25)
                q3 = np.percentile(self.num_df[VAR2].dropna(), 75)
                iqr = q3 - q1

                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                stats_table = self.num_df[VAR2].describe()
                stats_table['Lower Bound'] = lower_bound
                stats_table['Upper Bound'] = upper_bound

                selected_indexes = [
                    'Lower Bound', '25%', '50%', '75%', 'Upper Bound'
                ]
                selected_desc_stats = stats_table.loc[selected_indexes]

                upper_outliers = [
                    val for val in self.num_df[VAR2] if val > upper_bound
                ]
                lower_outliers = [
                    val for val in self.num_df[VAR2] if val < lower_bound
                ]

                with desc_col:
                    desc_col.write('##')
                    desc_col.markdown("#### :blue[Insight!]")
                    table, outlier_desc = st.columns([1, 2])
                    table.dataframe(selected_desc_stats)
                    if len(upper_outliers) > 1:
                        outlier_desc.markdown(
                            f"**Variables {VAR2} have {len(upper_outliers)} upper outliers ranged from:**\n {min(upper_outliers)} to {max(upper_outliers)}"
                        )
                    elif len(upper_outliers) == 1:
                        outlier_desc.markdown(
                            f"**Variables {VAR2} have {len(upper_outliers)} upper outlier:**\n {min(upper_outliers)}"
                        )
                    if len(lower_outliers) > 1:
                        outlier_desc.markdown(
                            f"**Variables {VAR2} have {len(lower_outliers)} upper outliers ranged from:**\n {min(lower_outliers)} to {max(lower_outliers)}"
                        )
                    elif len(lower_outliers) == 1:
                        outlier_desc.markdown(
                            f"**Variables {VAR2} have {len(lower_outliers)} upper outlier:**\n {min(lower_outliers)}"
                        )
                    if len(upper_outliers) == 0 and len(lower_outliers) == 0:
                        outlier_desc.markdown("No outliers found.")

                with img_col:
                    # Create a plot using Plotly
                    img_col.plotly_chart(fig)

                st.write('---')

                ## Percentile PLOT
                c3 = st.container()
                desc_col, img_col = c3.columns(2)
                # Create a Streamlit app
                desc_col.subheader("Percentile Plot")

                # Widget selection for user input
                VAR3 = desc_col.selectbox('Percentile Variable:',
                                          self.num_df.columns)

                # Select a percentile for analysis
                selected_percentile = desc_col.slider("Select Percentile:", 1,
                                                      99, 80)
                percentile_value = np.percentile(self.num_df[VAR3].dropna(),
                                                 selected_percentile)
                percentiles = np.arange(1, 101)
                percentile_values = np.percentile(self.num_df[VAR3].dropna(),
                                                  percentiles)

                # Create a line plot using Plotly
                fig = px.line(x=percentiles,
                              y=percentile_values,
                              labels={
                                  'x': 'Percentile',
                                  'y': 'Value'
                              })

                # Add horizontal and vertical cross indicators with semi-transparent color
                fig.add_shape(type="line",
                              x0=selected_percentile,
                              x1=selected_percentile,
                              y0=0,
                              y1=max(self.num_df[VAR3].dropna()),
                              line=dict(color="rgba(255, 0, 0, 0.4)",
                                        width=2,
                                        dash='dash'))

                fig.add_shape(type="line",
                              x0=1,
                              x1=100,
                              y0=percentile_value,
                              y1=percentile_value,
                              line=dict(color="rgba(255, 0, 0, 0.4)",
                                        width=2,
                                        dash='dash'))

                with desc_col:
                    desc_col.write('##')
                    desc_col.markdown("#### :blue[Insight!]")
                    desc_col.markdown(
                        f"{selected_percentile}th Percentile Value: {percentile_value:.2f}, this mean that {selected_percentile}% of data in {VAR3} are more than {percentile_value:.2f}"
                    )

                with img_col:
                    # Display the line plot using Streamlit
                    img_col.plotly_chart(fig)

                st.write('---')

            if len(self.num_df.columns) < 2:
                st.write("Your Data only have Single Numerical Values")
            else:
                ## Scatter PLOT
                c1 = st.container()
                desc_col, img_col = c1.columns(2)
                # Create a Streamlit app
                desc_col.subheader("Scatter Plot")

                # Widget selection for user input
                VARX = desc_col.selectbox('Scatter Variable X:',
                                          self.num_df.columns)
                VARY = desc_col.selectbox(
                    'Scatter Variable Y:',
                    [x for x in self.num_df.columns if x != VARX])

                # Calculate correlation coefficient
                x_data = self.num_df[VARX].dropna()
                y_data = self.num_df[VARY].dropna()
                min_length = min(len(x_data), len(y_data))
                x_data = x_data[:min_length]
                y_data = y_data[:min_length]
                correlation = np.corrcoef(x_data, y_data)[0, 1]

                # Create a scatter plot using Plotly Express
                fig = px.scatter(x=self.num_df[VARX],
                                 y=self.num_df[VARY],
                                 labels={
                                     'x': VARX,
                                     'y': VARY
                                 })

                # Annotate the scatter plot with correlation value
                fig.update_layout(annotations=[
                    dict(x=0.5,
                         y=1.15,
                         showarrow=False,
                         text=f'Correlation: {correlation:.2f}',
                         xref="paper",
                         yref="paper")
                ])

                with desc_col:
                    # Determine the strength and direction of the correlation
                    if correlation > 0.7:
                        RELATION = 'Strong Positive'
                    elif correlation > 0.5 and correlation <= 0.7:
                        RELATION = 'Relatively Positive'
                    elif correlation < -0.5 and correlation >= -0.7:
                        RELATION = 'Relatively Negative'
                    elif correlation < -0.7:
                        RELATION = 'Strong Negative'
                    else:
                        RELATION = 'Weak'

                    desc_col.write('##')
                    desc_col.markdown("#### :blue[Insight!]")
                    desc_col.markdown(
                        f"Linear Correlation between {VARX} and {VARY} is {RELATION}"
                    )

                with img_col:
                    # Display the line plot using Streamlit
                    img_col.plotly_chart(fig)

                st.write('---')

                ## Heatmap Correlation PLOT
                c2 = st.container()
                desc_col, img_col = c2.columns(2)
                # Create a Streamlit app
                desc_col.subheader("Correlation Plot")

                # Selectbox for correlation method
                selected_corr_method = desc_col.selectbox(
                    "Select Correlation Method:",
                    ['pearson', 'kendall', 'spearman'])

                # Selectbox to choose target variable
                selected_target = desc_col.selectbox(
                    "Select Target Variable for Correlation Insight:",
                    self.num_df.columns)
                selected_colorscale = desc_col.selectbox(
                    "Select Colorscale:",
                    ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis'])

                # Calculate the correlation matrix
                corr_matrix = self.num_df.corr(method=selected_corr_method)

                # Create a heatmap using Plotly
                fig = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                                x=corr_matrix.columns,
                                                y=corr_matrix.index,
                                                colorscale=selected_colorscale,
                                                text=corr_matrix.round(2)))

                # Customize the heatmap layout
                fig.update_layout(title="Correlation Heatmap",
                                  xaxis_title="Features",
                                  yaxis_title="Features",
                                  width=600,
                                  height=600)
                with desc_col:
                    # Write insights about strong correlations with the selected target variable
                    desc_col.write('##')
                    desc_col.markdown("#### :blue[Insight!]")
                    NO_INSIGHT = True
                    THRESHOLD = 0.7
                    for variable in self.num_df.columns:
                        if variable != selected_target and abs(
                                corr_matrix.loc[selected_target,
                                                variable]) > THRESHOLD:
                            NO_INSIGHT = False
                            desc_col.write(
                                f"Strong correlation between {selected_target} and {variable}: {corr_matrix.loc[selected_target, variable]:.2f}"
                            )
                    if NO_INSIGHT:
                        desc_col.write(
                            f"No variables have strong correlation with {selected_target}"
                        )
                with img_col:
                    # Display the heatmap using Streamlit
                    img_col.plotly_chart(fig)

        with tab3:
            st.write("Under Construct !")

        with tab4:
            st.write("Under Construct !")
