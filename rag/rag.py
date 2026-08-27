import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# export GROQ_API_KEY=xxxx  avant de lancer le script, ou décommente la ligne suivante :
# os.environ["GROQ_API_KEY"] = "ta-cle-ici"

# 1. Charger le document
pdf_path = "../corpus/381_dolipranesuspnot.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# 2. Découper le texte
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# 3. Embeddings 
print("Indexation du document en cours...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 4. Modèles via Groq 
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
llm_fast = ChatGroq(model="openai/gpt-oss-20b", temperature=0)  # reformulation : modèle plus léger

# 5. Prompt avec historique + consigne de citation des sources
prompt = ChatPromptTemplate.from_messages([
    ("system", """Tu es un assistant qui répond aux questions sur une notice de médicament.
Utilise UNIQUEMENT le contexte fourni pour répondre, ne complète jamais avec tes propres connaissances.
Si la question utilise des pronoms comme 'il', 'elle', 'ses', 'ces',
réfère-toi à l'historique de conversation pour comprendre de qui on parle.
Si la réponse ne se trouve pas dans le contexte, dis-le clairement ("Je ne trouve pas cette information dans le document"), ne l'invente jamais.
Quand tu t'appuies sur un passage, cite sa source entre crochets, par exemple [Source 1].

Contexte du document :
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def get_context_aware_query(query: str, chat_history: list) -> str:
    """Reformule la question en tenant compte de l'historique."""
    if not chat_history:
        return query
    history_text = "\n".join([
        f"Q: {m.content}" if isinstance(m, HumanMessage) else f"R: {m.content[:100]}..."
        for m in chat_history[-4:]  # 2 derniers échanges
    ])
    reformulation_prompt = f"""Historique: {history_text}
Nouvelle question: {query}

Reformule la nouvelle question de façon autonome et complète en une phrase,
en remplaçant les pronoms par les entités auxquelles ils font référence.
Réponds UNIQUEMENT avec la question reformulée, rien d'autre."""
    reformulated = llm_fast.invoke(reformulation_prompt)
    return reformulated.content


def format_docs(docs) -> str:
    """Formate les chunks récupérés avec une référence de source citable."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "document"))
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source {i} - {source}, page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def answer_question(query: str, chat_history: list) -> str:
    """Point d'entrée réutilisable : à exposer plus tard comme outil `search_corpus`
    dans l'agent avec tool-calling (RAG traité comme un outil parmi les outils MCP)."""
    smart_query = get_context_aware_query(query, chat_history)
    context_docs = retriever.invoke(smart_query)
    context = format_docs(context_docs)

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "context": context,
        "question": query,
        "chat_history": chat_history,
    })
    return response


if __name__ == "__main__":
    print("\n--- IA prête ! Posez vos questions sur le document (tapez 'quit' pour sortir) ---")
    chat_history = []

    while True:
        query = input("\nVotre question : ")
        if query.lower() == "quit":
            break

        response = answer_question(query, chat_history)

        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=response))

        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

        print(f"\nRéponse : {response}")
