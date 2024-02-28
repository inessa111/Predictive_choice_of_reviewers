from pythonProject import *
from pythonProject import find_similar
from pythonProject import encode
from pythonProject import preprocess








if __name__ == "__main__":
    #similar_papers = find_similar_papers(paper_id='10.1063/5.0124896', sim_score=0.6, exclude_authored=False)
    text_input=input()
    similar_papers = find_similar.find_similar_papers_by_text(text=text_input, sim_score=0.4)

    for item in similar_papers:
        for key, value in item.items():
            print(key,":", value)
        print('---')
