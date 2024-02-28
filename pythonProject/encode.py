import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle

# Encodes embeddings of text abstracts in a given dataframe and saves df and embeddings as a pickle
# Кодирует внедрение текстовых абстракций в данный фрейм данных и сохраняет df и внедрение в виде pickle

# Set file paths
# Установка путей к файлам
#input_file=r'C:\Users\123\Desktop\Diplom\Program\scopus.csv' #a path to dataframe with the abstracts to encode / путь к датафрейму с аннотациями для кодирования
#source_column='abstracts_pp' #set a df column name to look for abstracts / задайте имя столбца df для поиска аннотаций
#id_column='DOI' #set a df column with corresponding paper ids / установить столбец df с соответствующими идентификаторами статей
#output_file=r'C:\Users\123\Desktop\Diplom\Program\scopus_embeddings.pickle' #a path to store embeddings as a pickle object / путь для хранения данных о эмбдинге в виде объекта pickle


input_file=r'C:\Users\123\Desktop\Diplom\Program\test_change.csv' #a path to dataframe with the abstracts to encode / путь к датафрейму с аннотациями для кодирования
source_column='Аннотация1' #set a df column name to look for abstracts / задайте имя столбца df для поиска аннотаций
id_column='DOI' #set a df column with corresponding paper ids / установить столбец df с соответствующими идентификаторами статей
output_file=r'C:\Users\123\Desktop\Diplom\Program\scopus_embeddings_rus.pickle' #a path to store embeddings as a pickle object / путь для хранения данных о эмбдинге в виде объекта pickle



# Load the SentenceTransformer model
# Загрузите модель SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# The main func to do embedding
# Основная функция для выполнения внедрения
def encode_abstracts(model, abstracts):
    with tqdm(total=len(abstracts), desc='Encoding abstracts') as pbar_encode:
        encoded_abstracts = []
        for abstract in abstracts:
            encoded_abstract = model.encode([abstract], convert_to_tensor=True)[0]
            encoded_abstracts.append(encoded_abstract.cpu().numpy())
            pbar_encode.update(1)
    return encoded_abstracts

# Load dataframe
# Загрузить фрейм данных
#df=pd.read_csv(input_file, header=0, sep='\t', encoding='utf-8')


df=pd.read_csv(input_file, header=0, sep=';', encoding='windows-1251')


# Remove duplicates by id (not abstract!) and empty rows
# Удалите дубликаты по id (не аннотации!) и пустые строки



df.dropna(subset=[source_column], inplace=True)
df.drop_duplicates(subset=[id_column], keep='first', inplace=True)



# Load abstracts to a list
# Загрузите аннотации в список
preprocessed_abstracts = []
for abstract in df[source_column]:
    preprocessed_abstracts.append(abstract)

print (f'loaded file {input_file} with {len(preprocessed_abstracts)} abstracts')

# Encode
# Кодировать
encoded_abstracts = encode_abstracts(model, preprocessed_abstracts)
output_data=[df,encoded_abstracts]
with open(output_file, 'wb') as file:
    pickle.dump(output_data, file)
print ('saved dataframe (list item 0) and encoded abstracts (list item 1) as a list object to:',output_file)


