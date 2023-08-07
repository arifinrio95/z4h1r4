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
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from wordcloud import WordCloud

st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
    messages = [
        {"role": "system", "content": "I only response with python syntax streamlit version, no other text explanation."},
        {"role": "user", "content": f"""I have a dataframe name df with the following column schema: {schema_str}, and 2 sample rows: {rows_str}. 
        1. Respon pertanyaan atau pernyataan ini: {input_pengguna}. 
        2. My dataframe already load previously, named df, use it, do not reload the dataframe.
        3. Only respond with python scripts in streamlit version without any text. 
        4. Start your response with “import”.
        5. Show all your response to streamlit apps.
        6. Use Try and Except.
        7. Pay attention to the column type before creating the script."""}
    ]

    # messages = [
    #     {"role": "system", "content": "I only response with python syntax streamlit version, no other text explanation."},
    #     {"role": "user", "content": f"""I have a dataframe name df with the following column schema: {schema_str}, and 2 sample rows: {rows_str}. 
    #     1. {input_pengguna}. 
    #     2. My dataframe already load previously, named df, use it, do not reload the dataframe.
    #     3. Respond with scripts without any text. 
    #     4. Only code in a single cell. 
    #     5. Don’t start your response with “Sure, here are”. 
    #     6. Start your response with “import” inside the python block. 
    #     7. Give and show with streamlit the title for every steps.
    #     8. Print with st.write the explanation for every syntax.
    #     9. Don’t give me any explanation about the script. Response only with python block.
    #     10. Do not reload the dataframe.
    #     11. Use Try and Except for each syntax.
    #     12. Print and show the detail step you did.
    #     13. Dont forget to show the steps with st.write."""}
    # ]
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
    column = st.selectbox('Select a Numeric Column for Box Plot:', df.select_dtypes(include=['number']).columns.tolist())
    sns.boxplot(x=df[column])
    st.pyplot()

# Function to display scatter plot
def show_scatter_plot(df):
    col1 = st.selectbox('Select the first Numeric Column:', df.select_dtypes(include=['number']).columns.tolist())
    col2 = st.selectbox('Select the second Numeric Column:', df.select_dtypes(include=['number']).columns.tolist())
    sns.scatterplot(x=col1, y=col2, data=df)
    st.pyplot()

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
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    st.pyplot()

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
    plt.plot(values, label='Data')
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

# Function to perform bar plot for categorical data
def show_bar_plot(df):
    column = st.selectbox('Select a Categorical Column for Bar Plot:', df.select_dtypes(include=['object']).columns.tolist())
    sns.countplot(x=column, data=df)
    st.pyplot()

# Function to perform pie chart for categorical data
def show_pie_chart(df):
    column = st.selectbox('Select a Categorical Column for Pie Chart:', df.select_dtypes(include=['object']).columns.tolist())
    df[column].value_counts().plot.pie(autopct='%1.1f%%')
    st.pyplot()

# Function to perform Linear Regression
def perform_linear_regression(df):
    X_columns = st.multiselect('Select Feature Columns:', df.select_dtypes(include=['number']).columns.tolist())
    y_column = st.selectbox('Select Target Column:', df.select_dtypes(include=['number']).columns.tolist())
    test_size = st.slider('Select Test Size for Train-Test Split:', 0.1, 0.5, 0.2)

    X = df[X_columns]
    y = df[y_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("Model Coefficients:", model.coef_)
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Function to perform Logistic Regression
def perform_logistic_regression(df):
    X_columns = st.multiselect('Select Feature Columns for Logistic Regression:', df.select_dtypes(include=['number']).columns.tolist())
    y_column = st.selectbox('Select Target Column for Logistic Regression:', df.select_dtypes(include=['object']).columns.tolist())
    test_size = st.slider('Select Test Size for Train-Test Split for Logistic Regression:', 0.1, 0.5, 0.2)

    X = df[X_columns]
    y = LabelEncoder().fit_transform(df[y_column])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Confusion Matrix:", confusion_matrix(y_test, y_pred))

# Function to perform K-Means Clustering
def perform_k_means_clustering(df):
    # Choose features
    features = st.multiselect('Select features for K-Means clustering:', df.select_dtypes(include=['number']).columns.tolist())
    
    if not features:  # If no features are selected
        st.warning('Please select at least one feature.')
        return
    
    X = df[features]

    # Choose number of clusters
    num_clusters = st.slider('Select Number of Clusters for K-Means:', 2, 10)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(X)
    labels = kmeans.labels_
    df['Cluster'] = labels

    # Display results
    st.write('Cluster Centers:', kmeans.cluster_centers_)
    st.write(df)

    # Optional: Visualize clusters if a target is chosen
    target = st.selectbox('Select a target column (Optional for visualization):', [None] + df.columns.tolist())
    
    if target:
        import matplotlib.pyplot as plt

        for i in range(num_clusters):
            subset = df[df['Cluster'] == i]
            plt.scatter(subset[features[0]], subset[features[1]], label=f"Cluster {i}", alpha=0.6)
            plt.scatter(kmeans.cluster_centers_[i][0], kmeans.cluster_centers_[i][1], marker='x', color='red')
        
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.legend()
        plt.title(f'K-Means Clustering with {num_clusters} clusters')
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

def main():
    input_pengguna = ""
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
        input_pengguna = ""
        # df = pd.read_csv(file)
        df = load_file_auto_delimiter(file)
        st.dataframe(df.head())

        # Extract df schema
        schema_dict = df.dtypes.apply(lambda x: x.name).to_dict()
        schema_str = json.dumps(schema_dict)
        # st.write("\nDataframe schema : ", schema_str)
        
        # Extract the first two rows into a dictionary
        rows_dict = df.head(2).to_dict('records')
        rows_str = json.dumps(rows_dict, default=str)

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
        
        # Create a button in the Streamlit app
        if st.button('Or automatically Explore the Data with Pandas Profiling'):
            # Create Pandas Profiling Report
            pr = ProfileReport(df, explorative=True)
        
            # Display the report
            st.title('Pandas Profiling Report')
            st_profile_report(pr)
            
        input_pengguna = ""
        # User Input
        input_pengguna = st.text_input("""Masukkan perintah anda untuk mengolah data tersebut: (ex: 'Lakukan EDA.', 'Buat 5 visualisasi insightful.', 'Lakukan metode2 statistika pada data tersebut.' """)
        if (input_pengguna != "") & (input_pengguna != None) :
            if st.button('Eksekusi!'):
                # schema_dict = {col: str(dtype) for col, dtype in df.dtypes.iteritems()}
                
    
                # Membuat text input dan menyimpan hasilnya ke dalam variabel
                
                # response = openai.ChatCompletion.create(
                #     # model="gpt-3.5-turbo-16k",
                #     model="gpt-4",
                #     messages=[
                #         {"role": "system", "content": "I only response with syntax, no other text explanation."},
                #         {"role": "user", "content": f"""I have a dataframe name df with the following column schema: {schema_str}, and 2 sample rows: {rows_str}. 
                #                                         1. {input_pengguna}. 
                #                                         2. My dataframe already load previously, named df, use it, do not reload the dataframe.
                #                                         3. Respond with scripts without any text. 
                #                                         4. Only code in a single cell. 
                #                                         5. Don’t start your response with “Sure, here are”. 
                #                                         6. Start your response with “import” inside the python block. 
                #                                         7. Give and show with streamlit the title for every steps.
                #                                         8. Give an explanation for every syntax.
                #                                         9. Don’t give me any explanation about the script. Response only with python block.
                #                                         10. Do not reload the dataframe.
                #                                         11. Use Try Except for each syntax.
                #                                         12. Gunakan st.write untuk selain visualisasi, dan st.pyplot untuk visualisasi."""}
                #     ],
                #     max_tokens=14000,
                #     temperature=0
                # )
                
                # script = response.choices[0].message['content']
                error_message = None
                previous_script = None
                retry_count = 0
                script = request_prompt(input_pengguna, schema_str, rows_str, error_message, previous_script, retry_count)
                exec(str(script))
                st.write("The Script:")
                st.text(script)
            
                # retry_count = 0
                # error_message = None
                # previous_script = None
                # while retry_count < 5:
                #     try:
                #         script = request_prompt(input_pengguna, schema_str, rows_str, error_message, previous_script, retry_count)
                #         exec(str(script))
    
                #         # error_message = None
                #         # previous_script = None
                #         # input_pengguna = ""
                #         # if st.button('Lihat Script.'):
                #         # st.write("")
                #         # # st.write("The Script:")
                #         # st.text(script)
                #         break
                #     except Exception as e:
                #         error_message = str(e)
                #         # previous_script = str(script)
                #         retry_count += 1
                #         # # st.write("Previous script:")
                #         # # st.text(previous_script)
                #         # st.write("Error: ",error_message)
                #         # st.write("Trying to solving...")

                #         if retry_count == 5:
                #             st.write("Maaf saya tidak bisa menyelesaikan perintah tersebut, coba perintah lain, atau modifikasi dan perjelas perintahnya.")
                #             retry_count = 0
                    # if (script!='') & st.button('Lihat Script.'):
                    #     st.write("")
                    #     # st.write("The Script:")
                    #     st.text(script)
            # error_message = None
            # previous_script = None
            input_pengguna = ""

            # Mengevaluasi string sebagai kode Python
            # exec(str(script))
            # if st.button('Lihat Script.'):
            #     st.write("The Script:")
            #     st.text(script)
            
            # Menyimpan plot sebagai file sementara dan menampilkan dengan Streamlit
            # plt.savefig("plot.png")
            # st.image("plot.png")

    # except:
    #     st.write("Mohon maaf error ges, coba perintah lain, atau modifikasi dan perjelas perintahnya.")


if __name__ == "__main__":
    main()
