from Preprocessing import Preprocessing
import pandas as pd
import traceback
import sys

def main(data_path, data_name):
    
    try:
        
        preprocessing = Preprocessing(data_folder = data_path)
        data =  pd.read_csv(data_path + data_name)
        data['cleaned_report'] = data['توضیحات پزشک'].apply(lambda x: preprocessing.cleaner(x))
        data['cleaned_report'] = data['cleaned_report'].apply(lambda x: preprocessing.complete_normalization(x))
        data['tokenized_report'] = data['cleaned_report'].apply(lambda x: preprocessing.sentence_tokenizer(x))
        data.to_excel(data_path + 'cleaned_tokenized_report.xlsx')
        print('The Final File is Saved at: ' + data_path + 'cleaned_tokenized_report.xlsx')
        
    except Exception:
        
        traceback.print_exc()

if __name__ == "__main__":
    
    data_path = sys.argv[1]
    data_name = sys.argv[2]
    
    main(data_path, data_name)
