import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
from pymystem3 import Mystem
nltk.download('stopwords')
from tqdm import tqdm
tqdm.pandas()

#Defines a function to preprocess texts and applies it to paper abstracts in a dataframe 
#Определяет функцию для предварительной обработки текстов и применяет ее к аннотациям статей в датафрейме

input_file=r'C:\Users\123\Desktop\Diplom\Program\test2.csv'
#input_file=r'C:\Users\123\Desktop\Diplom\Program\scopus.csv' #a dataframe with the columnn with abstracts / фрейм данных со столбцом с аннотациями
#output_file=input_file #where to store results / где хранить результаты
output_file=r'C:\Users\123\Desktop\Diplom\Program\test_change.csv'
source_columnn='Аннотация'
#source_columnn='abstracts' #the name of the column in input file containing texts to be preprocessed / имя колонки во входном файле, содержащей тексты, подлежащие предварительной обработке
target_column='Аннотация1' #the name of the column to store preprocessed texts in the output file / имя колонки для хранения препроцессированных текстов в выходном файле

#set stopwords for filtering
#установка стоп-слов для фильтрации

#stop_words = set(stopwords.words('english'))
stop_words = set(stopwords.words('russian'))

# And also some specific academia stopwords:
# А также несколько специфических научных стоп-слов:

#stop_addon={'paper','study','present','aim','propose','purpose','chapter','background','article','introduction','investigate','consider'}
stop_addon={'Статья', 'исследование', 'настоящее', 'цель', 'предложение', 'цель', 'глава', 'история', 'введение', 'исследовать', 'рассматривать'}
stop_words.update(stop_addon)

def preprocess_text(text):

    # Skip empty texts and the like
    # Пропускаем пустые тексты и т.п.
    if not isinstance(text, str):
        return text
    
    # Lowercase the text
    # Текст, набранный в нижнем регистре
    text = text.lower()

    # Remove copyrights and 'abstract:'
    # Удалите авторские права и "аннотацию:".


    #text = re.sub(r'©.*', '', text)
    #text = re.sub(r'abstract:', '', text)
    #text = re.sub(r'abstract:', '', text)
    #text = re.sub(r'^abstract', '', text)
    #text = re.sub(r'^abstract', '', text)
    #text = re.sub(r'copyright.*', '', text)



    # normalise dash
    # нормализовать тире
    text = re.sub(r'[–—‐]', '-', text)

    # Remove special characters and punctuation - could be detrimental for chemistry and the like
    # Удалите специальные символы и знаки препинания - это может навредить при работе с программой по химии и т.п.
    text = re.sub(r'[^-A-Za-z0-9\s-]+', '', text)

    # Remove all numbers except years (e.g., 2021)
    # text = re.sub(r'\b(?!(?:19|20)\d{2}\b)\d+\b', '', text) #useful for humanities
    # Удалите все числа, кроме годов (например, 2021)
    # text = re.sub(r'\b(?!(?:19|20)\d{2}\b)\d+\b', '', text) # полезно для гуманитарных наук


    # Lemmatize the text
    # Лемматизировать текст (привести слова в начальную форму)
    lemmatizer = WordNetLemmatizer()
    words = text.split()


    # Exclude specific words from lemmatization
    # Исключить конкретные слова из лемматизации
    preserved_words = ['discuss', 'has', 'assess']  # Add more words if needed / и другие слова при необходимости
    words = [lemmatizer.lemmatize(word) if word not in preserved_words else word for word in words]

    # Remove stopwords
    # Удалить стоп-слова
    text = ' '.join(word for word in words if word not in stop_words)

    return text


from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation

# Create lemmatizer and stopwords list
mystem = Mystem()
russian_stopwords = stopwords.words("russian")


# Preprocess function
def preprocess_text_rus(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords \
              and token != " " \
              and token.strip() not in punctuation]

    text = " ".join(tokens)

    return text


if __name__ == "__main__":

    #df=pd.read_csv(input_file, header=0, sep='\t', encoding='utf-8')
    df=pd.read_csv(input_file, header=0, sep=';', encoding='windows-1251')

    print (f'loaded file: {input_file}')
    #df['abstracts_pp']=df['abstracts'].progress_apply(preprocess_text)
    df['Аннотация1'] = df['Аннотация'].progress_apply(preprocess_text_rus)

    #df.to_csv(output_file, index=False, sep='\t', encoding='utf-8')
    df.to_csv(output_file, index=False, sep=';', encoding='windows-1251')

    print (f'exported to file: {input_file}')
