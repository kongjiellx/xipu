FROM ollama/ollama:latest
RUN apt update && apt install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN DEBIAN_FRONTEND=noninteractive apt install python3.11 python3-pip -y -q
RUN pip install langchain chromadb
RUN rm -rf /root/.ollama/models
ADD models /root/.ollama/models
WORKDIR /workdir
ADD 10_rag/laodongfa.emb /workdir/laodongfa.emb
