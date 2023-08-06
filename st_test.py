import openai
import streamlit as st
import pandas as pd
import requests
import json
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import pandas as pd
from io import StringIO
import os

warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

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
        {"role": "system", "content": "I only response with syntax, no other text explanation."},
        {"role": "user", "content": f"""I have a dataframe name df with the following column schema: {schema_str}, and 2 sample rows: {rows_str}. 
                                                    1. {input_pengguna}. 
                                                    2. My dataframe already load previously, named df, use it, do not reload the dataframe.
                                                    3. Respond with scripts without any text. 
                                                    4. Only code in a single cell. 
                                                    5. Don’t start your response with “Sure, here are”. 
                                                    6. Start your response with “import” inside the python block. 
                                                    7. Give and show with streamlit the title for every steps.
                                                    8. Give an explanation for every syntax.
                                                    9. Don’t give me any explanation about the script. Response only with python block.
                                                    10. Do not reload the dataframe.
                                                    11. Gunakan st.write untuk selain visualisasi, dan st.pyplot untuk visualisasi.
                                                    12. Pastikan semua library yang dibutuhkan telah diimport"""}
    ]
    
    if error_message and previous_script:
        messages.append({"role": "user", "content": f"Solve this error: {error_message} in previous Script : {previous_script} to "})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        max_tokens=12000,
        temperature=0
    )
    script = response.choices[0].message['content']

    return script


def main():
    input_pengguna = ""
    import warnings
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title('Personal Data Analysis by Datasans')
    st.write('Beta Access.')
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
        
        if st.button('Klik disini jika kamu ingin saya melakukan data cleansing secara otomatis.'):
            # st.subheader('Data cleansing...')
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                # model="gpt-4",
                messages=[
                    {"role": "system", "content": "I only response with syntax, no other text explanation."},
                    {"role": "user", "content": f"""I have a dataframe name df with the following column schema: {schema_str}, and 2 sample rows: {rows_str}. 
                                                    1. Do a data cleansing and update the df. 
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
                                                    12. Print with st.write the detail step of data cleansing you did.
                                                    13. Export clean df with st.download_button("Press to Download",df.to_csv(index=False).encode('utf-8'),"file.csv","text/csv",key='download-csv')
                                                    """}
                ],
                max_tokens=14000,
                temperature=0
            )
            
            script = response.choices[0].message['content']
            exec(str(script))

        # Cari kolom yang memiliki missing values
        missing_columns = [col for col in df.columns if df[col].isnull().any()]
        
        if missing_columns:
            st.write("Kolom dengan missing values:")
            
            selected_methods = {}
            for column in missing_columns:
                # Tentukan opsi berdasarkan tipe data
                if df[column].dtype == 'float64' or df[column].dtype == 'int64':
                    options = ['0', 'Average']
                else:
                    options = ['Modus', 'Unknown']
        
                # Dropdown untuk memilih metode pengisian untuk setiap kolom
                selected_methods[column] = st.selectbox(f"Pilih metode pengisian untuk kolom {column}:", options)
        
            # Tombol untuk mengisi semua missing values
            if st.button("Handling missing values"):
                for column, method in selected_methods.items():
                    fill_missing_values(df, column, method)
                st.write('Missing values telah dihandle.')
        else:
            st.write("Tidak ada missing values dalam DataFrame.")
            
        # password = st.text_input("Masukkan Password: ")
        # if st.button('Submit'):
        #     if password != st.secrets['pass']:
        #         st.write('Password Salah.')
                
        #     if password == st.secrets['pass']:
                
        

        # EDA (automate sebelum 
        input_pengguna = ""
        # User Input
        input_pengguna = st.text_input("Masukkan perintah anda untuk mengolah data tersebut:")

        
                
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
                
                
                retry_count = 0
                error_message = None
                previous_script = None
                while retry_count < 5:
                    try:
                        script = request_prompt(input_pengguna, schema_str, rows_str, error_message, previous_script, retry_count)
                        exec(str(script))
    
                        error_message = None
                        previous_script = None
                        input_pengguna = ""
                        # if st.button('Lihat Script.'):
                        st.write("")
                        st.write("The Script:")
                        st.text(script)
                        break
                    except Exception as e:
                        error_message = str(e)
                        previous_script = str(script)
                        retry_count += 1
                        # st.write("Previous script:")
                        # st.text(previous_script)
                        st.write("Error: ",error_message)
                        st.write("Trying to solving...")

                        if retry_count == 5:
                            st.write("Maaf saya tidak bisa menyelesaikan perintah tersebut, coba perintah lain, atau modifikasi dan perjelas perintahnya.")
                error_message = None
                previous_script = None
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
