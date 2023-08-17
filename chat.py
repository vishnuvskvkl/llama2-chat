import os

import chainlit as cl
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.callbacks import StreamingStdOutCallbackHandler

prompt_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.
The example of your response should be:

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return prompt


def create_retrieval_qa_chain(llm, prompt, db):

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


def load_model():
    c = {"gpu_layers": 20, "max_new_tokens": 512, "temperature": 0,"context_length":4092}

    llm = CTransformers(
        model="./llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        callbacks=[StreamingStdOutCallbackHandler()],
        config=c

    )

    return llm


def create_retrieval_qa_bot(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        persist_dir="db_chroma",
        device="cpu",
):


    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"No directory found at {persist_dir}")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        )
    except Exception as e:
        raise Exception(
            f"Failed to load embeddings with model name {model_name}: {str(e)}"
        )

    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    try:
        llm = load_model()  # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

    qa_prompt = (
        set_custom_prompt()
    )  # Assuming this function exists and works as expected

    try:
        qa = create_retrieval_qa_chain(
            llm=llm, prompt=qa_prompt, db=db
        )  # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to create retrieval QA chain: {str(e)}")

    return qa


def retrieve_bot_answer(query):

    qa_bot_instance = create_retrieval_qa_bot()
    bot_response = qa_bot_instance({"query": query})
    return bot_response


@cl.on_chat_start
async def initialize_bot():

    qa_chain = create_retrieval_qa_bot()
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Hi, Welcome to iLE_Chat. How can I help you?"
    )
    await welcome_message.update()

    cl.user_session.set("chain", qa_chain)


@cl.on_message
async def process_chat_message(message):

    qa_chain = cl.user_session.get("chain")
    callback_handler = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    callback_handler.answer_reached = True
    response = await qa_chain.acall(message, callbacks=[callback_handler])
    bot_answer = response["result"]
    source_documents = response["source_documents"]

    if source_documents:
        bot_answer += f"\nSources:" + str(source_documents)
    else:
        bot_answer += "\nNo sources found"

    await cl.Message(content=bot_answer).send()
