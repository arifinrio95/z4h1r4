import pandas as pd
import csv

class LoadDataframe():
  
    def __init__(self, file):
        self.file = file
        self.df = None

    def detect_delimiter(self):
        self.file.seek(0)  # Reset file position to the beginning
        file_content = self.file.read(1024).decode(errors='ignore')  # Convert bytes to string, ignoring errors
        delimiter = ","
        try:
            dialect = csv.Sniffer().sniff(file_content)
            delimiter = dialect.delimiter
        except csv.Error:
            pass  # keep the delimiter as ","
        self.file.seek(0)  # Reset file position to the beginning after reading
        return delimiter

    def load_file_auto_delimiter(self):
        # Get file extension
        file_extension = self.file.name.split('.')[-1]  # Assuming the file has an extension
        if file_extension == 'csv':
            delimiter = self.detect_delimiter()
            self.file.seek(0)  # Reset file position to the beginning
            self.df = pd.read_csv(self.file, delimiter=delimiter)
        elif file_extension in ['xls', 'xlsx']:
            self.df = pd.read_excel(self.file)
        else:
            raise ValueError(f'Unsupported file type: {file_extension}')
        return self.df
      
    # def load_csv_auto_delimiter(self):
    #     delimiter = detect_delimiter(file)
    #     file.seek(0)  # Reset file position to the beginning
    #     df = pd.read_csv(file, delimiter=delimiter)
    #     return df
