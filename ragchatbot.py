from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


def make_pipeline():
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=400,
    )
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    memory = ConversationBufferWindowMemory(
        k=3,
        memory_key="history",
        return_messages=True
    )
    prompt = load_prompt('ethos_bot/prompt_socrates.txt')
    retriever = load_database('ethos_bot/kb_index')

    return prompt, memory, llm, retriever


def load_prompt(path):
    with open(path, 'r') as file:
        prompt_text = file.read()
    prompt = ChatPromptTemplate.from_template(prompt_text)
    return prompt
    
    
def load_database(index_path):
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    db = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity")
    return retriever


def reindex_database():
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    loader = DirectoryLoader('ethos_bot/data_source', glob='*.txt', loader_cls=TextLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
    chunked_docs = splitter.split_documents(docs)
    db = FAISS.from_documents(chunked_docs, embedding_model)
    db.save_local('ethos_bot/kb_index')
    #db.save_local('ethos_bot/kb_index', index_name='kb-index')


class RagChatbot:

    def __init__(self, prompt, memory, llm, retriever):
        self.prompt = prompt
        self.memory = memory
        self.llm = llm
        self.retriever = retriever


    def format_docs(docs):
        return "\n".join(f"{i+1}. {doc[0].page_content.strip()}" for i, doc in enumerate(docs))
    
    
    def converse(self, user_input):

        if user_input.lower() == 'clear memory':
            self.memory.clear()
            return {'response': "Memory cleared.", 'docs': []}
    
        # Pull chat history
        messages = self.memory.chat_memory.messages if hasattr(self.memory, "chat_memory") else []
        chat_history = "\n".join(
            f"User: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}"
            for m in messages
        )

        if user_input.lower() == 'summarize memory':
            summary = self.memory.get_summary()
            return {'response': summary, 'docs': []}

        # Retrieve relevant docs
        docs = self.retriever.vectorstore.similarity_search_with_score(user_input, k=6)
        matching_docs = [d for d in docs if d[1] < 0.8]
        if not matching_docs:
            return {'response': "AI: I don't know the answer to that question.", 'docs': []}
        context = RagChatbot.format_docs(matching_docs)

        # Format the prompt
        prompt_text = self.prompt.format(
            context=context,
            history=chat_history,
            input=user_input
        )

        # Generate the response
        response = self.llm.invoke(prompt_text)
        cleaned = response.split('<|im_start|>assistant\n')[-1].strip()
        if 'AI: ' in cleaned:
            cleaned = cleaned.split('AI: ')[-1].strip()

        # Save the context and response to memory
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(cleaned)

        return {'response': cleaned, 'docs': [{'doc': d[0].metadata['source'], 'score': d[1]} for d in matching_docs]}
