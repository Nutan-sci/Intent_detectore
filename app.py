import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, retrieval_qa
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.agents import tools, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import os

print("libreries imported")

st.set_page_config(page_title=" Intent Detectore", layout="wide")
st.title("Intent Detection Bot with RAG")

intent_examples = [
    ("I want to buy a new smartphone", "buy_product"),
    ("Can you help me reset my password?", "technical_support"),
    ("Tell me about your refund policy", "ask_policy"),
    ("What's the weather like in Mumbai?", "out_of_scope"),
    ("Schedule a meeting with the sales team", "schedule_meeting"),
    ("How do I track my order?", "track_order"),
]

few_shot_example = "\n".join(
    [f"user: {text}\n intent: {intent}" for text, intent in intent_examples]
)

intent_prompt = PromptTemplate(
    input_variables=["query"],
    template=f""" You are AI assistent train to classify user intent into catogories.
                               Here are some examples:
                               {few_shot_example}
                               user: {{query}}
                               intent:""",
)

llm = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
intent_chain = LLMChain(llm=llm, prompt=intent_prompt)
model_name = "sentence-transformers/all-MiniLM-L6-v2"


def detect_intent(query):
    return intent_chain.run(query=query).strip()


# RAG setup


@st.cache_resource
def create_qa_chain():
    loader = TextLoader
    docs = loader.load()
    embeddings = SentenceTransformer(model_name)

    vectorstores = FAISS.from_documents(docs, embeddings)
    retriver = vectorstores.as_retriever(vectorstores)

    return retrieval_qa.from_chain_type(llm=llm, retriver=retriver)


qa_chain = create_qa_chain()
# Agent setup
tool = [
    tools(
        name="Product_qa",
        func=qa_chain.run(),
        discription="Use this for answering Product related and FAQ Questions",
    )
]

memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(
    tools=tool, llm=llm, agent="Chat-Converstional-react-description", memory=memory
)

if "messages" not in st.session_state:
    st.session_state.message = []

user_input = st.chat_input("Ask me anything")

if user_input:
    # display user input
    st.session_state.message.append(("user", user_input))

    # detect intent
    intent = detect_intent(user_input)

    if intent == "out of focus":
        bot_response = "Sorry, I am not trained to handel that query"

    elif intent in ["ask_policy", "track_order", "buy_that_product"]:
        bot_response = agent.run(user_input)
    else:
        bot_response = (
            f" Inten: {intent}. you can connect with our support team for help"
        )

    st.session_state.messages.append("bot", bot_response)

    for rol, msg in st.session_state.messages:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(msg)
