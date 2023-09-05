import numpy as np
from scipy.stats import boxcox


def generate_insight_stats(df, var_name):
    """
    Generate statistical insights for a given variable in a DataFrame.
    
    Parameters:
    - df: DataFrame containing the data
    - var_name: Name of the variable to analyze
    
    Returns:
    - text: A string containing the statistical insights
    """
    data = df[var_name]
    insights = []

    # 1. Distribution Shape
    skewness = np.skew(data)
    if skewness > 1:
        insights.append("1. The data is highly positively skewed.")
    elif 0.5 <= skewness <= 1:
        insights.append("1. The data is moderately positively skewed.")
    elif -0.5 <= skewness <= 0.5:
        insights.append("1. The data is approximately symmetric.")
    elif -1 <= skewness <= -0.5:
        insights.append("1. The data is moderately negatively skewed.")
    else:
        insights.append("1. The data is highly negatively skewed.")

    # 2. Central Tendency
    mean = np.mean(data)
    median = np.median(data)
    insights.append(f"2. The mean is {mean} and the median is {median}.")

    # 3. Spread or Variability
    std_dev = np.std(data)
    insights.append(
        f"3. The standard deviation is {std_dev}, indicating the spread of the data."
    )

    # 4. Outliers
    kurt = np.kurtosis(data)
    if kurt > 0:
        insights.append(
            f"4. The data has heavy tails with kurtosis {kurt}, indicating the presence of outliers."
        )
    else:
        insights.append(
            f"4. The data has light tails wutg kurtosis {kurt}, indicating the absence of outliers."
        )

    # 5. Box-Cox Transformation
    data_pos = data + np.abs(np.min(
        data))  # Adding minimum data to make all data semidefinite positive

    # Perform Box-Cox transformation
    _, lambda_value = boxcox(data_pos)

    insights.append(
        f"Optimal lambda value: {lambda_value:.2f} in Box-Cox transformation to transform the data similar to a normal distribution."
    )

    text = "\n".join(insights)

    return text


def generate_insight_buz(df, aggregation, total_categories):
    """
    Generate insights from a DataFrame with two columns.
    
    Parameters:
    - df (DataFrame): The input DataFrame.
    - aggregation (str): The type of aggregation used ('count', 'sum', etc.).
    - total_categories (int): The total number of unique categories in the DataFrame.
    
    Returns:
    - text (str): A string containing the generated insights.
    """

    col1, yname = df.columns.tolist()
    total = df[yname].sum()

    # Helper function to generate insight text for top or bottom categories
    def generate_category_text(categories, label):
        text = f"{label} {len(categories)} {col1} categories by {yname} are:\n"
        for category, value, proportion in categories:
            text += f"  - {category} with {aggregation} value = {value} ({proportion}).\n"
        return text

    # Get Top 3 and Bottom 3 categories
    top_3 = df.nlargest(3, yname)
    bottom_3 = df.nsmallest(3, yname).loc[~df[col1].isin(top_3[col1])]

    # Calculate proportions
    top_3['Proportion'] = (top_3[yname] / total *
                           100).round(2).astype(str) + " %"
    bottom_3['Proportion'] = (bottom_3[yname] / total *
                              100).round(2).astype(str) + " %"

    # Missing Data
    missing_data = df[df[col1] == 'Missing Data'][yname].sum()
    missing_proportion = (missing_data / total *
                          100) if missing_data > 0 else 0

    # Generate insights text
    text = ""
    if not top_3.empty:
        text += generate_category_text(top_3.itertuples(index=False), "Top")
    if not bottom_3.empty:
        text += f"\n{generate_category_text(bottom_3.itertuples(index=False), 'Bottom')}"
    else:
        text += f"\n2. Category only have {len(top_3)} unique values."

    if missing_data > 0:
        text += f"\n\n3. There are {missing_data} missing data points, which make up {missing_proportion:.2f}% of the data."
    else:
        text += "\n\n3. There is no missing data."

    text += f"\n\n4. Variable {col1} has {total_categories} unique categories."

    return text