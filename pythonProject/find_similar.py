import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
from preprocess import preprocess_text
from preprocess import preprocess_text_rus

# A function that takes an id of a paper and returns a list of dicts with similar papers from a table (column names become dict keys)
# Функция, которая принимает идентификатор статьи и возвращает список массивов с похожими статьями из таблицы (имена столбцов становятся ключами для массивов)
# We need a pickle object - a list with dataframe with papers to search through (item 0) and embeddings of abstracts in this dataframe (item 1)
# Нам нужен объект pickle - список, содержащий датафрейм со статьями для поиска (элемент 0) и внедрения аннотаций в этот датафрейм (элемент 1)

#pickle_path=r'C:\Users\123\Desktop\Diplom\Program\scopus_embeddings.pickle' # path to pickled list with df and embeddings / путь к сортированному списку с df и внедрениями
#source_column='abstracts_pp' #set a df column name to look for abstracts / задайте имя столбца df для поиска аннотаций
#id_column='DOI' #point to a df column with corresponding paper ids / указывают на столбец df с соответствующими идентификаторами статей
# these are optional, only needed if you filter papers of the same authors who wrote the paper you query via id: / эти параметры необязательны, они нужны только в том случае, если вы фильтруете статьи тех же авторов, которые написали статью, которую вы запрашиваете по id:
#author_id_column='ids' #point to a df column / указать на столбец df
#author_id_delimiter=';' #specify author id delimiter / указать разделитель идентификаторов авторов





pickle_path=r'C:\Users\123\Desktop\Diplom\Program\scopus_embeddings_rus.pickle' # path to pickled list with df and embeddings / путь к сортированному списку с df и внедрениями
source_column='Аннотация1' #set a df column name to look for abstracts / задайте имя столбца df для поиска аннотаций
id_column='DOI' #point to a df column with corresponding paper ids / указывают на столбец df с соответствующими идентификаторами статей
# these are optional, only needed if you filter papers of the same authors who wrote the paper you query via id: / эти параметры необязательны, они нужны только в том случае, если вы фильтруете статьи тех же авторов, которые написали статью, которую вы запрашиваете по id:
author_id_column='ids' #point to a df column / указать на столбец df
author_id_delimiter=',' #specify author id delimiter / указать разделитель идентификаторов авторов





# Load the SentenceTransformer model
# Загрузите модель SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load df and embeddings
# Загрузите df и внедрения
with open(pickle_path, 'rb') as file:
    obj = pickle.load(file)
df=obj[0]
encoded_abstracts=obj[1]
print (f'loaded dataframe from {pickle_path} with {len(encoded_abstracts)} abstract embeddings')
'''
# The main function finds the paper by id in the df. adjust similarity score as needed (0)
# Основная функция находит статью по id в df. При необходимости корректируем оценку сходства (0)
def find_similar_papers(paper_id, sim_score=0.6, exclude_authored=False, verbose=False):

    #Print the paper data with the queried ID:
    # Вывести данные статей с запрошенным идентификатором:
    row = df[df[id_column] == paper_id]
    # Check if the row was found
    # Проверьте, найден ли ряд.
    if not row.empty:
        # Print the metadata of queried paper
        # Вывести метаданные запрашиваемой статьи
        if verbose:
            print ('queried paper:')
            print(row.iloc[0], '\n')  
    else:
        print (f'no paper with id {paper_id} found')
        return []

    # Locate and encode the abstract we are searching similars for
    # Найдите и закодируйте аннотацию, для которой мы ищем аналогии
    preprocessed_paper_abstract = row[source_column].values[0]    
    encoded_paper_abstract = model.encode([preprocessed_paper_abstract], convert_to_tensor=True)[0]

    # Compute the cosine similarity between the paper and other abstracts
    # Вычислите косинусоидальное сходство между статьей и другими аннотациями

    similarity_scores = cosine_similarity(
        encoded_paper_abstract.cpu().numpy().reshape(1, -1), np.vstack(encoded_abstracts)
    )[0]

    # Create a list of dictionaries containing the similar paper information
    # Создайте список словарей, содержащих информацию о похожих статьях.

    similar_papers = []
    for i, score in enumerate(similarity_scores):
        if score > sim_score and score < 0.98:  #Removes the paper itself from results. also somehow the random papers with very high similarity keep showing up... this filters them out  / Удаляет саму статью из результатов. Также почему-то появляются случайные статьи с очень высоким сходством... это отсеивает их
            selected_row = df.iloc[i]
            row_dict = selected_row.to_dict()
            row_dict['similarity_score']=score
            similar_papers.append(row_dict)
            if exclude_authored: # Remove papers with the same authors as the queried paper / # Удалить статьи с теми же авторами, что и запрашиваемая статья
                source_authors = df[df[id_column] == paper_id][author_id_column].values[0].split(author_id_delimiter)
                source_authors = [x.strip() for x in source_authors]
                for paper in similar_papers:
                    paper_authors = paper[author_id_column].split(author_id_delimiter)
                    paper_authors = [x.strip() for x in paper_authors]
                    if set(source_authors).intersection(set(paper_authors)):
                        similar_papers.remove(paper)
            similar_papers.sort(key=lambda x: x['similarity_score'], reverse=True) #Sort list of papers by sim score / Сортировка списка статей по степени sim

    return similar_papers
'''
#this function finds papers not by ID, but by text (any text)
# эта функция находит статьи не по ID, а по тексту (любому тексту)
def find_similar_papers_by_text(text, sim_score=0.6):

    # Locate and encode the abstract we are searching similars for
    # Найдите и закодируйте аннотацию, для которой мы ищем аналогии
    encoded_paper_abstract = model.encode([preprocess_text_rus(text)], convert_to_tensor=True)[0]

    # Compute the cosine similarity between the paper and other abstracts
    # Вычислите косинусоидальное сходство между статьей и другими аннотациями
    similarity_scores = cosine_similarity(
        encoded_paper_abstract.cpu().numpy().reshape(1, -1), np.vstack(encoded_abstracts)
    )[0]

    # Create a list of dictionaries containing the similar paper information
    # Создайте список словарей, содержащих информацию о похожих статьях.
    similar_papers = []
    for i, score in enumerate(similarity_scores):
        if score > sim_score and score < 0.98:  #Removes the paper itself from results. also somehow the random papers with very high similarity keep showing up... this filters them out / Удаляет саму статью из результатов. Также почему-то появляются случайные статьи с очень высоким сходством... это отсеивает их
            selected_row = df.iloc[i]
            row_dict = selected_row.to_dict()
            row_dict['similarity_score']=score
            similar_papers.append(row_dict)
            similar_papers.sort(key=lambda x: x['similarity_score'], reverse=True) #Sort list of papers by sim score / Сортировка списка статей по степени sim
    return similar_papers

# Here we can run the func to test it
# Здесь мы можем запустить func для проверки.

if __name__ == "__main__":
    #similar_papers = find_similar_papers(paper_id='10.1063/5.0124896', sim_score=0.6, exclude_authored=False)
    text_input=input()
    similar_papers = find_similar_papers_by_text(text=text_input, sim_score=0.4)

    for item in similar_papers:
        for key, value in item.items():
            print(key,":", value)
        print('---')


# Пример аннотации для теста
#A method is proposed for determining the operating speed of a reciprocating compressor with regenerative heat transfer in pump mode, when the operating speed of the machine in compressor mode is known. The mathematical models are based on fundamental laws of energy and mass conservation, the equations of motion, and the equations of state. The operating speed of the machine in pump mode is determined on the basis that the maximum pressure losses when the machine operates in pump and compressor modes are the same and the relative energy losses are equal.