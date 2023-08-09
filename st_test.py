import openai
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import pandas as pd
from io import StringIO
import os
import time
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, classification_report, roc_curve, auc, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
from wordcloud import WordCloud
import base64
import pygwalker as pyg

hide_menu = """
<style>
#stHeader {
    visibility:hidden;
}
</style>
"""

# Fungsi untuk mengisi missing values berdasarkan pilihan pengguna
def fill_missing_values(df, column, method):
    if df[column].dtype == 'float64' or df[column].dtype == 'int64': # Jika numeric
        if method == '0':
            df[column].fillna(0, inplace=True)
        elif method == 'Average':
            df[column].fillna(df[column].mean(), inplace=True)
    else: # Jika kategorikal
        if method == 'Modus':
            df[column].fillna(df[column].mode().iloc[0], inplace=True)
        elif method == 'Unknown':
            df[column].fillna('Unknown', inplace=True)


def detect_delimiter(file):
    file.seek(0)  # Reset file position to the beginning
    file_content = file.read(1024).decode(errors='ignore')  # Convert bytes to string, ignoring errors
    delimiter = ","
    try:
        dialect = csv.Sniffer().sniff(file_content)
        delimiter = dialect.delimiter
    except csv.Error:
        pass  # keep the delimiter as ","
    file.seek(0)  # Reset file position to the beginning after reading
    return delimiter

# def load_csv_auto_delimiter(file):
#     delimiter = detect_delimiter(file)
#     file.seek(0)  # Reset file position to the beginning
#     df = pd.read_csv(file, delimiter=delimiter)
#     return df

def load_file_auto_delimiter(file):
    # Get file extension
    file_extension = file.name.split('.')[-1]  # Assuming the file has an extension
    if file_extension == 'csv':
        delimiter = detect_delimiter(file)
        file.seek(0)  # Reset file position to the beginning
        df = pd.read_csv(file, delimiter=delimiter)
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(file)
    else:
        raise ValueError(f'Unsupported file type: {file_extension}')
    return df

def request_prompt(input_pengguna, schema_str, rows_str, error_message=None, previous_script=None, retry_count=0):
    # versi 2 prompt
    # messages = [
    #     {"role": "system", "content": "I only response with python syntax streamlit version, no other text explanation."},
    #     {"role": "user", "content": f"""I have a dataframe name df with the following column schema: {schema_str}, and 2 sample rows: {rows_str}. 
    #     1. Respon pertanyaan atau pernyataan ini: {input_pengguna}. 
    #     2. My dataframe already load previously, named df, use it, do not reload the dataframe.
    #     3. Only respond with python scripts in streamlit version without any text. 
    #     4. Start your response with “import”.
    #     5. Show all your response to streamlit apps.
    #     6. Use Try and Except.
    #     7. Pay attention to the column type before creating the script."""}
    # ]

    # versi 1 prompt
    messages = [
        {"role": "system", "content": "I only response with python syntax streamlit version, no other text explanation."},
        {"role": "user", "content": f"""I have a dataframe name df with the following column schema: {schema_str}, and 2 sample rows: {rows_str}. 
        1. {input_pengguna}. 
        2. My dataframe already load previously, named df, use it, do not reload the dataframe.
        3. Respond with scripts without any text. 
        4. Only code in a single cell. 
        5. Don’t start your response with “Sure, here are”. 
        6. Start your response with “import” inside the python block. 
        7. Give and show with streamlit the title for every steps.
        8. Print with st.write the explanation for every syntax.
        9. Don’t give me any explanation about the script. Response only with python block.
        10. Do not reload the dataframe.
        11. Use Try and Except for each syntax.
        12. Print and show the detail step you did.
        13. Dont forget to show the steps with st.write."""}
    ]
    # Give and show with streamlit the title for every steps. Give an explanation for every syntax. 
    
    if error_message and previous_script:
        messages.append({"role": "user", "content": f"Solve this error: {error_message} in previous Script : {previous_script} to "})

    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo-16k",
        model="gpt-3.5-turbo",
        # model="gpt-4",
        messages=messages,
        max_tokens=3000,
        temperature=0
    )
    script = response.choices[0].message['content']

    return script

def request_story_prompt(dict_stats):
    messages = [
        {"role": "system", "content": "Aku akan membuat laporan story telling menarik yang siap disajikan."},
        {"role": "user", "content": f"""Buatkan laporan berbentuk story telling yang menarik dari data yang deskripsinya saya berikan dalam dictionary berikut:  {dict_stats}"""}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        # model="gpt-3.5-turbo",
        # model="gpt-4",
        messages=messages,
        max_tokens=10000,
        temperature=0.9
    )
    script = response.choices[0].message['content']

    return script

# Function to display descriptive statistics
def show_descriptive_statistics(df):
    st.write('Descriptive Statistics')
    st.write(df.describe())

# Function to display a histogram
def show_histogram(df):
    # Selecting the numeric column
    column = st.selectbox('Select a Numeric Column for Histogram:', df.select_dtypes(include=['number']).columns.tolist())
    
    # Customization options
    bins = st.slider('Select Number of Bins:', 5, 50, 15) # Default is 15 bins
    kde = st.checkbox('Include Kernel Density Estimate (KDE)?', value=True) # Default is to include KDE
    color = st.color_picker('Pick a color for the bars:', '#3498db') # Default is a shade of blue
    
    # Plotting the histogram using Seaborn
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column], bins=bins, kde=kde, color=color)
    
    # Rendering the plot in Streamlit
    st.pyplot(plt)

# Function to display a box plot
def show_box_plot(df):
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    y_column = st.selectbox('Select a Numeric Column for Y-axis:', numeric_columns)
    x_column = st.selectbox('Select a Categorical Column for X-axis (Optional):', [None] + categorical_columns)

    show_swarm = st.checkbox('Show Swarm Plot?')
    color_option = st.color_picker('Pick a Color for Box Plot')
    cat_color_option = st.color_picker('Pick a Color for Categorical Box Plot', '#95a5a6') if x_column else None

    fig, ax = plt.subplots(figsize=(10, 6))

    if x_column:
        sns.boxplot(x=x_column, y=y_column, data=df, ax=ax, color=cat_color_option)
    else:
        sns.boxplot(x=x_column, y=y_column, data=df, ax=ax, color=color_option)

    if show_swarm:
        sns.swarmplot(x=x_column, y=y_column, data=df, ax=ax, color='black', size=3)

    sns.despine(left=True, bottom=True)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    plt.title('Box Plot of ' + y_column, fontsize=16, fontweight="bold")
    plt.ylabel(y_column, fontsize=12)
    plt.xlabel(x_column if x_column else '', fontsize=12)

    if x_column:
        categories = df[x_column].unique()
        for category in categories:
            subset = df[df[x_column] == category][y_column]
            median = subset.median()
            plt.annotate(f'Median: {median}', xy=(categories.tolist().index(category), median), xytext=(-20,20),
                         textcoords='offset points', arrowprops=dict(arrowstyle='->'), fontsize=10)
    else:
        median = df[y_column].median()
        plt.annotate(f'Median: {median}', xy=(0, median), xytext=(-20,20),
                     textcoords='offset points', arrowprops=dict(arrowstyle='->'), fontsize=10)

    st.pyplot(fig)



# Function to display scatter plot
def show_scatter_plot(df):
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    col1 = st.selectbox('Select the first Numeric Column:', numeric_columns)
    col2 = st.selectbox('Select the second Numeric Column:', numeric_columns)
    hue_col = st.selectbox('Select a Categorical Column for Coloring (Optional):', [None] + categorical_columns)
    size_option = st.slider('Select Point Size:', min_value=1, max_value=10, value=5)
    show_regression_line = st.checkbox('Show Regression Line?')
    annotate_points = st.checkbox('Annotate Points?')

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(x=col1, y=col2, hue=hue_col, data=df, s=size_option * 10, ax=ax, palette='viridis' if hue_col else None)

    if show_regression_line:
        sns.regplot(x=col1, y=col2, data=df, scatter=False, ax=ax, line_kws={'color': 'red'})

    if annotate_points:
        for i, txt in enumerate(df.index):
            ax.annotate(txt, (df[col1].iloc[i], df[col2].iloc[i]), fontsize=8)

    sns.despine(left=True, bottom=True)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    plt.title(f'Scatter Plot of {col1} vs {col2}', fontsize=16, fontweight="bold")
    plt.xlabel(col1, fontsize=12)
    plt.ylabel(col2, fontsize=12)

    st.pyplot(fig)


# Function to display correlation matrix
def show_correlation_matrix(df):
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=['number'])

    # Compute the correlation matrix
    corr = numerical_df.corr()

    # You can then display the correlation matrix using Streamlit as you desire
    st.write(corr)

    # Optionally, you may want to visualize it as a heatmap
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    st.pyplot(plt)

# Function to perform PCA
def perform_pca(df):
    numeric_df = df.select_dtypes(include=['number'])
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # User options
    n_components = st.slider('Select Number of Principal Components:', 1, min(numeric_df.shape[1], 10), 2)
    scaling_option = st.selectbox('Select Scaling Option:', ['StandardScaler', 'MinMaxScaler', 'RobustScaler'])
    color_by = st.selectbox('Color By Categorical Column:', [None] + categorical_columns)
    show_scree_plot = st.checkbox('Show Scree Plot', value=False)

    # Scaling
    if scaling_option == 'StandardScaler':
        scaler = StandardScaler()
    elif scaling_option == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaling_option == 'RobustScaler':
        scaler = RobustScaler()

    scaled_data = scaler.fit_transform(numeric_df)

    # PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)

    # Membuat DataFrame dengan hasil PCA dan data aktual
    pca_df = pd.DataFrame(data=pca_result, columns=[f'Principal Component {i}' for i in range(1, n_components + 1)])
    combined_df = pd.concat([df.reset_index(drop=True), pca_df], axis=1)

    # Tombol Unduh
    csv = combined_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Beberapa byte dance
    href = f'<a href="data:file/csv;base64,{b64}" download="pca_result.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)

    # Scatter plot
    fig, ax = plt.subplots()
    if color_by:
        sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=df[color_by], palette='viridis')
    else:
        plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    st.pyplot(fig)

    # Scree plot
    if show_scree_plot:
        fig, ax = plt.subplots()
        plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance')
        st.pyplot(fig)

    # Loadings
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components)], index=numeric_df.columns)
    st.write("Loadings:")
    st.write(loadings)


# Function to show missing data
def show_missing_data(df):
    st.write('Check Missing Values')
    missing_data = df.isnull().sum()
    st.write(missing_data[missing_data > 0])

# Function to show outliers using Z-score
def show_outliers(df):
    column = st.selectbox('Select a Numeric Column for Outlier Detection:', df.select_dtypes(include=['number']).columns.tolist())
    values = df[column].dropna()
    z_scores = np.abs(stats.zscore(values))
    threshold = 2
    outliers = np.where(z_scores > threshold)

    st.write(f"Outliers found at index positions: {outliers}")

    # Plotting the data
    plt.figure(figsize=(14, 6))
    plt.scatter(range(len(values)), values, label='Data')
    plt.axhline(y=np.mean(values) + threshold*np.std(values), color='r', linestyle='--', label='Upper bound')
    plt.axhline(y=np.mean(values) - threshold*np.std(values), color='r', linestyle='--', label='Lower bound')

    # Annotating the outliers
    for idx in outliers[0]:
        plt.annotate(f'Outlier\n{values.iloc[idx]}', (idx, values.iloc[idx]), 
                     textcoords="offset points", 
                     xytext=(-5,5),
                     ha='center',
                     arrowprops=dict(facecolor='red', arrowstyle='wedge,tail_width=0.7', alpha=0.5))
        
    plt.title(f'Outlier Detection in {column}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    st.pyplot(plt)

# Function to perform Shapiro-Wilk normality test
def perform_shapiro_wilk_test(df):
    column = st.selectbox('Select a Numeric Column for Normality Testing:', df.select_dtypes(include=['number']).columns.tolist())
    data = df[column].dropna()
    _, p_value = stats.shapiro(data)
    if p_value > 0.05:
        st.write(f"The data in the column '{column}' appears to be normally distributed (p-value = {p_value}).")
    else:
        st.write(f"The data in the column '{column}' does not appear to be normally distributed (p-value = {p_value}).")

    # Plotting the histogram
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(data, bins=15, kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Plotting the Q-Q plot
    plt.subplot(1, 2, 2)
    sm.qqplot(data, line='s', ax=plt.gca())
    plt.title(f'Q-Q Plot of {column}')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')

    st.pyplot(plt)

def show_bar_plot(df):
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    column = st.selectbox('Select a Categorical Column for Bar Plot:', categorical_columns)
    chart_type = st.selectbox('Select Chart Type:', ['Single', 'Grouped', 'Stacked', '100% Stacked'])
    y_column = None
    aggregation_method = None

    if chart_type != 'Single':
        y_column = st.selectbox('Select a Numeric Column:', numeric_columns)
        aggregation_method = st.selectbox('Select Aggregation Method:', ['sum', 'mean', 'count', 'max', 'min'])
    
    aggregation_methods = {
        'sum': np.sum,
        'mean': np.mean,
        'count': 'count',
        'max': np.max,
        'min': np.min,
    }
    aggregation_func = aggregation_methods[aggregation_method] if aggregation_method else None

    orientation = st.selectbox('Select Orientation:', ['Vertical', 'Horizontal'])
    color_option = st.selectbox('Select Bar Color:', sns.color_palette().as_hex())
    sort_option = st.selectbox('Sort By:', ['None', 'Value', 'Category'])
    order = None
    if sort_option == 'Value' and y_column:
        order = df.groupby(column).agg({y_column: aggregation_method}).sort_values(by=y_column, ascending=False).index
    elif sort_option == 'Category':
        order = sorted(df[column].unique())

    if not y_column and chart_type != 'Single':
        st.warning('Please select a Numerical Column for chart types other than Single.')
        return

    if chart_type == 'Single':
        # Handle Single chart type
        if orientation == 'Vertical':
            sns.countplot(x=column, data=df, order=order, color=color_option)
        elif orientation == 'Horizontal':
            sns.countplot(y=column, data=df, order=order, color=color_option)

    # Continue with other chart types
    else:
        if aggregation_method == 'count':
            data_to_plot = df.groupby(column).size().reset_index(name=y_column)
        else:
            data_to_plot = df.groupby(column)[y_column].agg(aggregation_func).reset_index()
        y_value = y_column

        if chart_type == 'Grouped':
            if orientation == 'Vertical':
                sns.barplot(x=column, y=y_value, data=data_to_plot, order=order, color=color_option)
            elif orientation == 'Horizontal':
                sns.barplot(y=column, x=y_value, data=data_to_plot, order=order, color=color_option)
        elif chart_type == 'Stacked':
            data_to_plot.plot(kind='bar', x=column, y=y_value, stacked=True, color=color_option)
        elif chart_type == '100% Stacked':
            df_stacked = data_to_plot.groupby(column).apply(lambda x: 100 * x / x.sum()).reset_index()
            df_stacked.plot(kind='bar', x=column, y=y_value, stacked=True, color=color_option)

    plt.title(f'Bar Plot of {column}', fontsize=16, fontweight="bold")
    plt.xlabel('Value' if y_column else 'Count', fontsize=12)
    plt.ylabel(column, fontsize=12)
    sns.despine(left=True, bottom=True)
    st.pyplot()

# Function to perform pie chart for categorical data
def show_pie_chart(df):
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    column = st.selectbox('Select a Categorical Column for Pie Chart:', categorical_columns)
    color_palette = st.selectbox('Choose a Color Palette:', sns.palettes.SEABORN_PALETTES.keys())
    show_percentage = st.checkbox('Show Percentage', value=True)
    show_labels = st.checkbox('Show Labels', value=True)
    explode_option = st.slider('Explode Segments:', 0.0, 0.5, 0.0)
    figsize_option = st.slider('Size of Pie Chart:', 5, 20, 10)

    labels = df[column].value_counts().index
    sizes = df[column].value_counts().values
    colors = sns.color_palette(color_palette, len(labels))
    explode = [explode_option] * len(labels)
    autopct = '%1.1f%%' if show_percentage else None

    fig, ax = plt.subplots(figsize=(figsize_option, figsize_option))
    ax.pie(sizes, explode=explode, labels=labels if show_labels else None, colors=colors, autopct=autopct, shadow=True, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig)


# Function to perform Linear Regression
def perform_linear_regression(df):
    X_columns = st.multiselect('Select Feature Columns:', df.select_dtypes(include=['number']).columns.tolist())
    if not X_columns:  # If no features are selected
        st.warning('Please select feature columns.')
        return
    
    y_column = st.selectbox('Select Target Column:', df.select_dtypes(include=['number']).columns.tolist())
    test_size = st.slider('Select Test Size for Train-Test Split:', 0.1, 0.5, 0.2)
    fit_intercept = st.checkbox('Fit Intercept?', value=True)

    X = df[X_columns]
    y = df[y_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("Model Coefficients:", model.coef_)
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))

    # Scatter plot of predicted vs actual values
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.title('Predicted vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Identity line
    st.pyplot()
    plt.clf()

    # Residual plot
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='red', linestyle='--')
    st.pyplot()


# Function to perform Logistic Regression
def perform_logistic_regression(df):
    X_columns = st.multiselect('Select Feature Columns for Logistic Regression:', df.select_dtypes(include=['number']).columns.tolist())
    y_column = st.selectbox('Select Target Column for Logistic Regression:', df.select_dtypes(include=['object']).columns.tolist())
    test_size = st.slider('Select Test Size for Train-Test Split for Logistic Regression:', 0.1, 0.5, 0.2)

    penalty_option = st.selectbox('Select Penalty Type:', ['l2', 'l1'])
    solver_option = 'saga' if penalty_option == 'l1' else 'newton-cg'

    X = df[X_columns]
    y = LabelEncoder().fit_transform(df[y_column])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = LogisticRegression(penalty=penalty_option, solver=solver_option)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    st.write("Classification Report:", classification_report(y_test, y_pred))

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    st.pyplot()

    # Confusion Matrix Heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot()

# Function to perform K-Means Clustering
def perform_k_means_clustering(df):
    # Pilih fitur numerik
    features = st.multiselect('Select features for K-Means clustering:', df.select_dtypes(include=['number']).columns.tolist())
    if not features or len(features) < 2:
        st.warning('Please select at least two numerical features.')
        return
    
    X = df[features]

    # Pra-pemrosesan: Skalakan fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Metode Elbow untuk menentukan jumlah klaster optimal
    distortions = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k).fit(X_scaled)
        distortions.append(sum(np.min(cdist(X_scaled, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X_scaled.shape[0])

    plt.plot(range(1, 11), distortions, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    st.pyplot(plt)

    optimal_clusters = np.argmin(np.diff(np.diff(distortions))) + 2
    st.write(f"The optimal number of clusters based on the Elbow Method is: {optimal_clusters}")

    num_clusters = st.slider('Select Number of Clusters for K-Means (recommended from Elbow Method):', 2, 10, optimal_clusters)

    # Lakukan klastering K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(X_scaled)
    df['Cluster'] = kmeans.labels_

    # Tampilkan pusat klaster dan label
    st.write('Cluster Centers (in scaled space):', kmeans.cluster_centers_)
    st.write(df)

    # Ambil statistik dari semua klaster
    cluster_stats = []
    for i in range(num_clusters):
        cluster_stat = df[df['Cluster'] == i].describe()
        cluster_stats.append(cluster_stat)

    # Analisis Klaster yang Mendetail dan Kesimpulan
    for i in range(num_clusters):
        st.write(f"Cluster {i} Statistics:")
        st.write(cluster_stats[i])

        conclusions = []
        for j in range(num_clusters):
            if i != j:
                conclusion = f"Compared to Cluster {j}, Cluster {i} has "
                
                # Contoh perbandingan berdasarkan rata-rata fitur pertama (gantikan dengan analisis yang relevan)
                if cluster_stats[i][features[0]]['mean'] > cluster_stats[j][features[0]]['mean']:
                    conclusion += f"a higher average of {features[0]}."
                else:
                    conclusion += f"a lower average of {features[0]}."

                conclusions.append(conclusion)

        st.write(conclusions)

    # Visualisasi 2D (gunakan dua fitur pertama)
    for i in range(num_clusters):
        subset = df[df['Cluster'] == i]
        plt.scatter(subset[features[0]], subset[features[1]], label=f"Cluster {i}", alpha=0.6)
        plt.scatter(kmeans.cluster_centers_[i][0], kmeans.cluster_centers_[i][1], marker='x', color='red')
    
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.legend()
    plt.title(f'K-Means Clustering with {num_clusters} clusters')
    st.pyplot(plt)

    # Penilaian Kualitas Klaster
    silhouette_avg = silhouette_score(X_scaled, df['Cluster'])
    st.write('Silhouette Score:', silhouette_avg)
    for i in range(num_clusters):
        st.write(f"Cluster {i} Statistics:")
        st.write(df[df['Cluster'] == i].describe())

        # Opsional: Pairplot untuk fitur yang dipilih dalam masing-masing klaster
        sns.pairplot(df[df['Cluster'] == i][features + ['Cluster']], hue='Cluster')
        st.pyplot(plt)


# Function to perform Time-Series Analysis
def perform_time_series_analysis(df):
    time_column = st.selectbox('Select Time Column:', df.select_dtypes(include=['datetime']).columns.tolist())
    if not time_column:  # If no features are selected
        st.warning('Please select column for time.')
        return
    target_column = st.selectbox('Select Target Column for Time-Series Analysis:', df.select_dtypes(include=['number']).columns.tolist())
    window_size = st.slider('Select Window Size for Moving Average:', 3, 30)

    df[time_column] = pd.to_datetime(df[time_column])
    df.set_index(time_column, inplace=True)
    moving_avg = df[target_column].rolling(window=window_size).mean()

    fig, ax = plt.subplots()
    ax.plot(df[target_column], label='Original')
    ax.plot(moving_avg, label=f'Moving Average (window={window_size})')
    ax.legend()
    st.pyplot(fig)

# Function to perform Hierarchical Clustering
def perform_hierarchical_clustering(df):
    # Select numerical columns or appropriate features
    X = df.select_dtypes(include=['number'])
    
    # Convert to float if not already
    X = X.astype(float)

    # Perform hierarchical clustering
    linkage_matrix = linkage(X, method='ward') # You can choose different linkage methods
    
    # Check the data type
    if linkage_matrix.dtype != 'float64':
        st.error("Unexpected data type for linkage matrix")
        return

    # Plot dendrogram
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram')
    st.pyplot(plt)

# Function to perform Text Analysis using Word Cloud
def perform_text_analysis(df):
    text_column = st.selectbox('Select a Text Column for Word Cloud:', df.select_dtypes(include=['object']).columns.tolist())
    text_data = " ".join(text for text in df[text_column])
    wordcloud = WordCloud().generate(text_data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# Ekstrak semua deskripsi statistik data
def analyze_dataframe(df):
    # Analisis Data Numerik
    numerical_columns = df.select_dtypes(include=['number']).columns
    numerical_summary = df[numerical_columns].describe().transpose().to_dict()
    numerical_summary['skewness'] = df[numerical_columns].skew().to_dict()
    numerical_summary['kurtosis'] = df[numerical_columns].kurt().to_dict()

    # Analisis Data Kategorikal
    categorical_columns = df.select_dtypes(include=['object']).columns
    categorical_summary = {col: {
            'unique_categories': df[col].nunique(),
            'mode': df[col].mode().iloc[0],
            'frequency': df[col].value_counts().iloc[0]
        } for col in categorical_columns}

    # Analisis Missing Values
    missing_values = df.isnull().sum().to_dict()
    missing_percentage = {col: (missing_values[col] / len(df) * 100) for col in df.columns}
    missing_summary = {
        "Missing Values": missing_values,
        "Percentage": missing_percentage
    }

    # Analisis Korelasi
    correlation_matrix = df.corr().to_dict()

    # Analisis Outliers
    z_scores = df[numerical_columns].apply(zscore)
    outliers = (z_scores.abs() > 2).sum().to_dict()  # Agregat jumlah outliers

    # Agregasi Lengkap untuk Semua Kolom yang Mungkin
    all_aggregations = df.agg(['mean', 'median', 'sum', 'min', 'max', 'std', 'var', 'skew', 'kurt']).transpose().to_dict()

    # Kuantil
    quantiles = df.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).transpose().to_dict()

    # Agregasi Group By untuk Semua Kombinasi Kolom Kategorikal
    groupby_aggregations = {}
    for r in range(1, len(categorical_columns) + 1):
        for subset in combinations(categorical_columns, r):
            group_key = ', '.join(subset)
            group_data = df.groupby(list(subset)).agg(['mean', 'count', 'sum', 'min', 'max'])
            groupby_aggregations[group_key] = group_data.to_dict()

    return {
        'Numerical Summary': numerical_summary,
        'Categorical Summary': categorical_summary,
        'Missing Values': missing_summary,
        'Correlation Matrix': correlation_matrix,
        'Outliers': outliers,
        'All Possible Aggregations': all_aggregations,
        'Quantiles': quantiles,
        'Group By Aggregations': groupby_aggregations
    }
    
def main():
    st.set_page_config(
    layout="wide",
    )
    
    import warnings
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.image('https://drive.google.com/uc?export=view&id=1uqJad2S9rcwPt-yVvrGIhKbUX-0Gvz0N', use_column_width=True)

    # st.title('Personal Data Analysis by Datasans')
    # st.write('Beta Access.')
    st.write('Beta access diberikan kepada beberapa user sebelum perilisan resmi, mohon digunakan dan berikan input melalui DM akun IG @datasans.book jika ada error atau fitur yang kurang sempurna.')
    st.subheader('Upload your CSV / Excel data:')
    file = st.file_uploader("Upload file", type=['csv', 'xls', 'xlsx'])

    # user_api = st.text_input("Masukkan OpenAI API Key anda: ")
    
    # os.environ['user_api'] = st.secrets['user_api']
    openai.api_key = st.secrets['user_api']

    # try:
    if file is not None:
        df = load_file_auto_delimiter(file)
        st.dataframe(df.head())

        # Extract df schema
        schema_dict = df.dtypes.apply(lambda x: x.name).to_dict()
        schema_str = json.dumps(schema_dict)
        st.write("\nDataframe schema : ", schema_str)

        # Extract the first two rows into a dictionary
        rows_dict = df.head(2).to_dict('records')
        rows_str = json.dumps(rows_dict, default=str)

        
        
        st.sidebar.subheader('Ada 4 opsi untuk mengeksplorasi data:')
        # Tombol 1
        if st.sidebar.button('1. Eksplorasi data secara manual (menggunakan PyGWalker)'):
            st.session_state.manual_exploration = True
            st.session_state.auto_exploration = False
            st.session_state.show_analisis_lanjutan = False
            st.session_state.show_natural_language_exploration = False
            st.session_state.story_telling = False

        
        # Tombol 2
        if st.sidebar.button('2. Eksplorasi data otomatis (menggunakan Pandas Profiling)'):
            st.session_state.manual_exploration = False
            st.session_state.auto_exploration = True
            st.session_state.show_analisis_lanjutan = False
            st.session_state.show_natural_language_exploration = False
            st.session_state.story_telling = False

        
        # Tombol 3
        if st.sidebar.button('3. Analisa tingkat lanjutan.'):
            st.session_state.manual_exploration = False
            st.session_state.auto_exploration = False
            st.session_state.show_analisis_lanjutan = True
            st.session_state.show_natural_language_exploration = False
            st.session_state.story_telling = False

        
        # Tombol 4
        if st.sidebar.button('4. Eksplorasi data dengan bahasa natural (disupport oleh ChatGPT)'):
            st.session_state.manual_exploration = False
            st.session_state.auto_exploration = False
            st.session_state.show_analisis_lanjutan = False
            st.session_state.show_natural_language_exploration = True
            st.session_state.story_telling = False

        # Tombol 5
        if st.sidebar.button('5. Buat laporan story telling secara otomatis.'):
            st.session_state.manual_exploration = False
            st.session_state.auto_exploration = False
            st.session_state.show_analisis_lanjutan = False
            st.session_state.show_natural_language_exploration = False
            st.session_state.story_telling = True

        
        if st.session_state.get('manual_exploration', False):
            st.subheader("PyGWalker")
            st.write("PyGWalker adalah pustaka Python untuk analisis visual dengan stye mirip Tableau, memungkinkan eksplorasi data dengan drag and drop seperti Tableau.")
            st.markdown("[Klik di sini untuk mempelajari lebih lanjut.](https://github.com/Kanaries/pygwalker)")
            # Jika tombol diklik, gunakan PyGWalker
            walker = pyg.walk(df, env='Streamlit')

        if st.session_state.get('auto_exploration', False):
            st.subheader("Pandas Profiling Report")
            # Create Pandas Profiling Report
            pr = ProfileReport(df, explorative=True)
        
            # Display the report
            st_profile_report(pr)

        if st.session_state.get('show_analisis_lanjutan', False):
            st.subheader("Analisis Lanjutan")
            analysis_option = st.sidebar.selectbox('Choose an analysis:', 
                                                   ('Descriptive Statistics', 'Histogram', 'Box Plot', 'Scatter Plot', 'Bar Plot', 'Pie Chart', 'Missing Data', 'Correlation Matrix',
                                                    'Principal Component Analysis', 'Outlier Detection',
                                                    'Normality Test', 'Linear Regression', 'Logistic Regression',
                                                    'K-Means Clustering', 'Time-Series Analysis', 'Hierarchical Clustering',
                                                    'Text Analysis'))

            if analysis_option == 'Hierarchical Clustering':
                perform_hierarchical_clustering(df)
            elif analysis_option == 'Handle Imbalance Classes':
                handle_imbalance_classes(df)
            elif analysis_option == 'Text Analysis':
                perform_text_analysis(df)
            elif analysis_option == 'Logistic Regression':
                perform_logistic_regression(df)
            elif analysis_option == 'K-Means Clustering':
                perform_k_means_clustering(df)
            elif analysis_option == 'Time-Series Analysis':
                perform_time_series_analysis(df)
            elif analysis_option == 'Bar Plot':
                show_bar_plot(df)
            elif analysis_option == 'Pie Chart':
                show_pie_chart(df)
            elif analysis_option == 'Linear Regression':
                perform_linear_regression(df)
            elif analysis_option == 'Missing Data':
                show_missing_data(df)
            elif analysis_option == 'Outlier Detection':
                show_outliers(df)
            elif analysis_option == 'Normality Test':
                perform_shapiro_wilk_test(df)
            elif analysis_option == 'Descriptive Statistics':
                show_descriptive_statistics(df)
            elif analysis_option == 'Histogram':
                show_histogram(df)
            elif analysis_option == 'Box Plot':
                show_box_plot(df)
            elif analysis_option == 'Scatter Plot':
                show_scatter_plot(df)
            elif analysis_option == 'Correlation Matrix':
                show_correlation_matrix(df)
            elif analysis_option == 'Principal Component Analysis':
                perform_pca(df)

        if st.session_state.get('show_natural_language_exploration', False):
            st.subheader("Natural Language Exploration")
            # input_pengguna = ""
            # User Input
            input_pengguna = st.text_input("""Masukkan perintah anda untuk mengolah data tersebut: (ex: 'Lakukan EDA.', 'Buat 5 visualisasi insightful.', 'Lakukan metode2 statistika pada data tersebut.' """)
            if (input_pengguna != "") & (input_pengguna != None) :
                error_message = None
                previous_script = None
                retry_count = 0
                script = request_prompt(input_pengguna, schema_str, rows_str, error_message, previous_script, retry_count)
                exec(str(script))
                st.write("The Script:")
                st.text(script)
                input_pengguna = ""

        if st.session_state.get('story_telling', False):
            st.subheader("Laporan Statistika")
            st.write(request_story_prompt(analyze_dataframe(df)))

if __name__ == "__main__":
    main()
