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

def detect_delimiter(file):
    file.seek(0)  # Reset file position to the beginning
    file_content = file.read(1024).decode()  # Convert bytes to string
    try:
        dialect = csv.Sniffer().sniff(file_content)
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","  # Fallback to comma if Sniffer fails
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

def main():
    import warnings
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title('Datasans Code Interpreter')
    st.subheader('Upload your CSV data:')
    file = st.file_uploader("Upload file", type=['csv', 'xls', 'xlsx'])

    # user_api = st.text_input("Masukkan OpenAI API Key anda: ")
    
    # os.environ['user_api'] = st.secrets['user_api']
    openai.api_key = st.secrets['user_api']
    
    if file is not None:
        # df = pd.read_csv(file)
        df = load_file_auto_delimiter(file)
        st.dataframe(df.head())

        # Extract df schema
        schema_dict = df.dtypes.apply(lambda x: x.name).to_dict()
        schema_str = json.dumps(schema_dict)
        st.write("\nDataframe schema : ", schema_str)

        # Extract the first two rows into a dictionary
        rows_dict = df.head(2).to_dict('records')
        rows_str = json.dumps(rows_dict, default=str)

        # User Input
        input_pengguna = st.text_input("Masukkan perintah anda untuk mengolah data tersebut:")

        if input_pengguna != "":
            # if st.button('Eksplorasi Dataset'):
            # schema_dict = {col: str(dtype) for col, dtype in df.dtypes.iteritems()}
            

            # Membuat text input dan menyimpan hasilnya ke dalam variabel
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                # model="gpt-4",
                messages=[
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
                                                    11. Use Try Except for each syntax.
                                                    12. Gunakan st.write untuk selain visualisasi, dan st.pyplot untuk visualisasi."""}
                ],
                max_tokens=14000,
                temperature=0
            )
            
            script = response.choices[0].message['content']
            
            
            


            # Mengevaluasi string sebagai kode Python
            exec(str(script))
            if st.button('Lihat Script.'):
                st.write("The Script:")
                st.text(script)
            
            # Menyimpan plot sebagai file sementara dan menampilkan dengan Streamlit
            # plt.savefig("plot.png")
            # st.image("plot.png")

if __name__ == "__main__":
    main()
