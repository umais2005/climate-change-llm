from prompts import general_prompt, contextualize_q_system_prompt, book_assistant_prompt
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from  data_ingestion import DocumentProcessor
import os
from dotenv import load_dotenv
from warnings import filterwarnings
filterwarnings("ignore")
load_dotenv()

dl = DocumentProcessor("./data/")
groq_api_key =  os.getenv("GROQ_TOKEN")
openai_api_key =  os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HF_TOKEN")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
openai_llm = ChatOpenAI( model="gpt-4o",
                        temperature=0.2)
# groq_llm = ChatGroq(model="llama3-70b-8192", api_key=groq_api_key, temperature=0.2)
vector_store_name = "context_for_book"

# Load vector store
print("loading vector store")
try:
    # vector_store = FAISS.load_local(vector_store_name, allow_dangerous_deserialization=True, embeddings=embeddings)
    vector_store = dl.vector_store
except:
    print("Vector database not found. Creating vector database. This might take some time")

else:
    retriever = vector_store.as_retriever(search_kwargs={"k":40})
    print("loaded successfully")

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    openai_llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", book_assistant_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(openai_llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


store = {}
user = "user"

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

if __name__ == "__main__":

    while True:
        if user in store:
            for message in store[user].messages[1:]:
                if isinstance(message, AIMessage):
                    prefix = "AI"
                else:
                    prefix = "User"

                print(f"{prefix}: {message.content}\n")

        user_input = input("User --> ")
        response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={
            "configurable": {"session_id": user}
        }
    )
        sources = list(set([document.metadata['source'] for document in response["context"]]))
        print(sources)