
from langchain_community.document_loaders import TextLoader

loader = TextLoader("./nvda_news_1.txt")
data = loader.load()
#print(data[0].page_content)

from langchain_community.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path='./movies.csv',source_column="title")
data = loader.load()
# print(len(data))
# print(data[0])


from langchain_community.document_loaders import UnstructuredURLLoader
loader = UnstructuredURLLoader(urls=[
    "https://laravel.com/docs/10.x/passport"
])
data = loader.load()
print(data)