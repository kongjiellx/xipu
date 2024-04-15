# -*-coding: utf-8 -*-
import re
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma

with open("laodongfa.txt", "r") as fp:
    text = fp.read().replace("\u3000", " ")
pattern = r"(  第.*?条.*?)(?=  第.*?条|第.*?章|$)"
matches = re.findall(pattern, text, re.DOTALL)

rules = []
for index, match in enumerate(matches):
    rules.append(match.strip())
vectorstore = Chroma.from_texts(
    texts=rules,
    embedding=OllamaEmbeddings(
        model="qwen:4b",
        embed_instruction="总结以下内容：",
        query_instruction="回答问题：",
    ), persist_directory="./laodongfa.emb")

vectorstore.similarity_search(query="什么情况劳动合同无效？")