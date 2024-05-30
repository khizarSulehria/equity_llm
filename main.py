import os
os.environ["OPENAI_API_KEY"] = "Enter your team"

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
import langchain

loader = UnstructuredURLLoader(urls=[
    "https://laravel.com/docs/10.x/passport",
    "https://www.toptal.com/laravel/passport-tutorial-auth-user-access"
])

data = loader.load()
dataText = data[0].page_content

llm = OpenAI(temperature=0.9, max_tokens=500)
text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ","
    ],
    chunk_size=1000,
    chunk_overlap=50
)
langchain.debug = True
docs = text_splitter.split_documents(data)
# embeddings = OpenAIEmbeddings()
# print("=====================Embading =================================")
# print(embeddings)
# print("=====================Embading =================================")

# vectorstore_openai = FAISS.from_documents(docs, embeddings)
#
# vectorstore_openai.save_local("faiss_store")

vectorstore = FAISS.load_local("faiss_store", OpenAIEmbeddings(),allow_dangerous_deserialization=True)
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
query = "how to get access token"

result = chain({"question": query}, return_only_outputs=True)
print(result)