from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

llm = ChatOllama(model="qwen:4b")

vectorstore = Chroma(
    embedding_function=OllamaEmbeddings(model="qwen:4b", embed_instruction="总结以下内容：",
        query_instruction="回答问题："),
    persist_directory="./laodongfa.emb",
)

retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
prompt = ChatPromptTemplate.from_messages([
    ("user", """你是一个用于回答问题的助手。使用以下检索到的上下文来回答问题。如果你不知道答案，就说你不知道。使用最多三个句子并保持回答简洁。如果Context和Question无关，则忽略Context内容。
```Question
{question} 
```
```Context
{context}
```""")
])

def format_docs(docs):
    print("相关法条如下：")
    for i, d in enumerate(docs):
        print(f"doc_{i}: {d.page_content}")
    print("=" * 60)
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
chat_history = []
while True:
    user_text = input("USER: ")
    ai_text = rag_chain.invoke(user_text)
    print(f"AI: {ai_text}")
    chat_history += [HumanMessage(content=user_text), AIMessage(content=ai_text)]
