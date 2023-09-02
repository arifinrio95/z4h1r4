import plotly.graph_objects as go
import pandas as pd


def generate_insight_text(insight,
                          var_name,
                          yname,
                          total_categories,
                          aggregation,
                          top_n=3,
                          bottom_n=3):
    text = ""

    # Top N categories
    if f'Top {top_n}' in insight:
        text += f"1. The top {top_n} {var_name} categories by {yname} are:\n"
        if aggregation in ['count', 'sum']:
            for category, value, proportion in insight[f'Top {top_n}']:
                text += f"  - {category} with {aggregation} value = {value} ({proportion}).\n"
        else:
            for category, value, _ in insight[f'Top {top_n}']:
                text += f"  - {category} with {aggregation} value = {value}.\n"

    # Bottom N categories
    bottom_key = f'Bottom {bottom_n}' if bottom_n > 0 else 'No Bottom'
    if f'Bottom {bottom_n}' in insight:
        text += f"\n2. The bottom {bottom_n} from {var_name} categories by {yname} are:\n"
        if aggregation in ['count', 'sum']:
            for category, value, proportion in insight[bottom_key]:
                text += f"  - {category} with {aggregation} value = {value} ({proportion}).\n"
        else:
            for category, value, _ in insight[bottom_key]:
                text += f"  - {category} with {aggregation} value = {value}.\n"
    elif 'No Bottom' in insight:
        text += f"\n2. {insight[bottom_key]}"

    # Missing Data
    if 'Missing Data' in insight:
        if insight['Missing Data'] != "No Missing Data":
            missing_data, missing_proportion = insight['Missing Data']
            text += f"\n\n3. There are {missing_data} missing data points, which make up {missing_proportion:.2f}% of the data."
        else:
            text += "\n\n3. There is no missing data."

    if 'No Bottom' not in insight:
        text += f"\n\n4. Variable {var_name} have {total_categories} unique categories."

    return text


def BarChart(df, col1, pallete, aggregation, col2=None, columns_chart=False):
    '''
    Create barchart figure
    '''
    orientation = 'h' if columns_chart else 'v'
    insight = {}

    if aggregation == 'count':
        # Calculate the frequency of each category
        freq_df = df[col1].value_counts().reset_index()
        yname = 'Frequency'

    else:
        # Calculate the frequency of each category
        freq_df = df.groupby(col1).agg({col2: aggregation}).reset_index()
        yname = f'{aggregation} of {col2}'

    freq_df.columns = [col1, yname]
    total_categories = len(freq_df)

    # Group categories beyond top 20 into 'Others'
    if len(freq_df) > 20:
        top_20 = freq_df.nlargest(20, yname)
        others = freq_df.iloc[20:].sum(numeric_only=True).values[0]
        # Create a DataFrame for the 'Others' category
        others_df = pd.DataFrame([{col1: 'Others', yname: others}])

        # Concatenate top_20 with others_df
        freq_df = pd.concat([top_20, others_df], ignore_index=True)

    # Calculate insights
    total = freq_df[yname].sum()
    # freq_df[yname] = pd.to_numeric(freq_df[yname])
    top_3 = freq_df.nlargest(3, yname)
    bottom_3 = freq_df.nsmallest(3, yname)
    bottom_3 = bottom_3.loc[~(
        bottom_3[col1].isin(top_3[col1].unique().tolist()))]
    missing_data = freq_df[freq_df[col1] == 'Missing Data'][yname].sum()
    insight = {
        f'Top {len(top_3)}': [(row[col1], row[yname],
                               str(round(row[yname] / total * 100, 2)) + " %")
                              for _, row in top_3.iterrows()],
        f'Bottom {len(bottom_3)}' if len(bottom_3) > 0 else 'No Bottom':
        [(row[col1], row[yname],
          str(round(row[yname] / total * 100, 2)) + " %")
         for _, row in bottom_3.iterrows()] if len(bottom_3) != 0 else
        f"Category only have {len(top_3)} unique values",
        'Missing Data':
        (missing_data,
         missing_data / total * 100) if missing_data > 0 else "No Missing Data"
    }

    text = generate_insight_text(insight,
                                 col1,
                                 yname,
                                 total_categories,
                                 aggregation,
                                 top_n=len(top_3),
                                 bottom_n=len(bottom_3))

    x_data = freq_df[col1] if orientation == 'v' else freq_df[yname]
    y_data = freq_df[yname] if orientation == 'v' else freq_df[col1]

    fig = go.Figure(data=[
        go.Bar(x=x_data,
               y=y_data,
               orientation=orientation,
               text=freq_df[yname],
               textposition='outside',
               marker=dict(color=pallete))
    ])

    # Customize hover text
    hovertemplate = f"{col1}=%{{x}}<br>{yname}=%{{y}}"
    fig.update_traces(hovertemplate=hovertemplate, name='')

    # Calculate y-axis tick values
    max_frequency = int(round(freq_df[yname].max() * 1.2, 0))
    tick_step = max_frequency // 6 if max_frequency >= 6 else 1
    tick_vals = [i for i in range(0, max_frequency + 1, tick_step)]

    if orientation == 'v':
        fig.update_layout(
            yaxis_title=f"<b>{yname}</b>",
            yaxis=dict(showgrid=True,
                       tickmode='array',
                       tickvals=tick_vals,
                       range=[0, max_frequency + 1],
                       title_font=dict(size=16, family="Arial, sans-serif")),
            xaxis_title=f"<b>{col1}</b>",
            xaxis=dict(title_font=dict(size=16, family="Arial, sans-serif")))
    else:
        fig.update_layout(
            xaxis_title=f"<b>{yname}</b>",
            xaxis=dict(showgrid=True,
                       tickmode='array',
                       tickvals=tick_vals,
                       range=[0, max_frequency + 1],
                       title_font=dict(size=16, family="Arial, sans-serif")),
            yaxis_title=f"<b>{col1}</b>",
            yaxis=dict(title_font=dict(size=16, family="Arial, sans-serif")))

    return fig, text
