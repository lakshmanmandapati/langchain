import gradio as gr
import os
os.environ["USER_AGENT"] = "lakshman-bot/1.0"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "FALSE"

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

def process_input(urls, question):
    model_local = ChatOllama(model="mistral")
    
    # Convert string of URLs to list
    urls_list = urls.strip().split("\n")
    docs = [WebBaseLoader(url.strip()).load() for url in urls_list if url.strip()]
    docs_list = [item for sublist in docs for item in sublist]
    
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()

    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question)

# Define Gradio interface
iface = gr.Interface(
    fn=process_input,
    inputs=[gr.Textbox(label="Enter URLs separated by new lines", lines=5), gr.Textbox(label="Your Question")],
    outputs="text",
    title="Document Query with Ollama",
    description="Enter URLs and a question. It will retrieve info using RAG and answer with a local LLM."
)

iface.launch()
