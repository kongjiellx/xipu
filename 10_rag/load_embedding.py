# -*-coding: utf-8 -*-
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma


vectorstore = Chroma(
    embedding_function=OllamaEmbeddings(model="qwen:4b", embed_instruction="总结以下内容：",
        query_instruction="回答问题："),
    persist_directory="./laodongfa.emb",
)

# for d in vectorstore.similarity_search(query="用人单位非法招用未满十六周岁的未成年人的，怎么处罚？", k=10):
for d in vectorstore.similarity_search(query="什么情况劳动合同无效？", k=10):
    print(d)