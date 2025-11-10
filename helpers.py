import re
import pandas as pd
import numpy as np
from zipfile import ZipFile


def unzip_chat(whatsapp_file_path):
    """
    Unzips the whatsapp.zip file and returns the loaded .txt file as a list
    Params: whatsapp_file_path= Path of the whatsapp zip file
    """
    with ZipFile(f"{whatsapp_file_path}.zip", "r") as f:
        f.extractall()
    with open(f"{whatsapp_file_path}.txt", "r", encoding='utf-8') as file:
        whatsapp_chat = file.readlines()
    return whatsapp_chat

def extract_chat_data(whatsapp_file_path):
    """
    Extracts important information like date, time, sender and message content from whatsapp.zip file 
    and returns a pandas dataframe with the columns: date, time, sender and message
    Params: whatsapp_file_path= Path of the whatsapp zip file
    """
    whatsapp_chat = unzip_chat(whatsapp_file_path=whatsapp_file_path)
    extracted_data_dict = {
        'date': [], 'time': [], 'sender': [], 'message': []
    }
    
    for chat in whatsapp_chat:
        try:
            date, time = re.search(r"\d{2}/\d{2}/\d{4}", chat), re.search(r"\d{2}:\d{2}", chat)
            sender, message = re.search(r"-\s*([^:]+):", chat), re.search(r": (.+)\n$", chat)

            extracted_data_dict['date'].append(date.group(0) if date else "")
            extracted_data_dict['time'].append(time.group(0) if time else "")
            extracted_data_dict['sender'].append(sender.group(1) if sender else "")
            extracted_data_dict['message'].append(message.group(1) if message else "")
        except:
            extracted_data_dict['date'].append("")
            extracted_data_dict['time'].append("")
            extracted_data_dict['sender'].append("")
            extracted_data_dict['message'].append("")
    assert len(extracted_data_dict['date'])== len(extracted_data_dict['time'])==\
          len(extracted_data_dict['sender'])== len(extracted_data_dict['message'])
    
    return pd.DataFrame({'date': extracted_data_dict['date'], 'time': extracted_data_dict['time'],\
               'sender': extracted_data_dict['sender'], 'message': extracted_data_dict['message']})


def preprocess_df(df: pd.DataFrame):
    df = df.iloc[1:, :]
    df['sender'].replace('tmak 3', 'Tmak', inplace=True)
    df = df[~df['message'].isin(['<Media omitted>', 'Waiting for this message'])].reset_index()
    df['date_and_time']=df['date']+ " " + df['time']
    df['date_and_time']= pd.to_datetime(df['date_and_time'], dayfirst=True, errors='coerce')
    df = df.dropna()
    df=df.drop(['date', 'time', 'index'], axis=1)
    df['month']= df['date_and_time'].dt.month_name()
    df['day'] = df['date_and_time'].dt.day_name()
    df['hour'] = df['date_and_time'].dt.hour
    df['date'] = df['date_and_time'].dt.date
    df['character_length']= df['message'].str.len()
    df['word_length'] = df['message'].apply((lambda x: len(x.split(" "))))
    time_bins= [-1, 6, 12, 16, 19, 24]
    time_labels= ['Midnight', 'Morning', 'Afternoon', 'Evening', 'Night']
    df['part_of_day']= pd.cut(df['hour'], bins=time_bins, labels=time_labels)
    df['hour']=df['hour'].apply(lambda x : f"0{x}:00" if x <10 else f"{x}:00")
    df['is_url']= df['message'].apply((lambda x: bool(re.search(r"(http:/|https:/)", str(x)))))
    df['response_time'] = df['date_and_time'].diff()
    df = df.reset_index().drop('index', axis=1)
    return df

# extracted_chat_data = extract_chat_data(whatsapp_file="WhatsApp Chat with The Boys")