from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import pandas as pd

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

class QuestionRequest(BaseModel):
    question: str

embeddings = OpenAIEmbeddings()

if not os.path.exists("chroma_db"):

    df = pd.read_csv(r"C:\Users\hp\Documents\Local_documents\RESUME_PROJECT_FOLDER\RAG_Knowledge_Search\dataset.csv")

    numeric_cols = [
        "price_numeric",
        "built_up_area_numeric_in_sq_ft",
        "bathrooms",
        "age_numeric"
    ]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["price_numeric", "built_up_area_numeric_in_sq_ft"])

    avg_price = df["price_numeric"].mean()
    price_by_lift = df.groupby("Lift")["price_numeric"].mean().to_dict()
    price_by_parking = df.groupby("Parking")["price_numeric"].mean().to_dict()
    price_by_furnishing = df.groupby("furnishing")["price_numeric"].mean().to_dict()
    area_price_corr = df["price_numeric"].corr(df["built_up_area_numeric_in_sq_ft"])

    documents = []

    analytics = [
        f"The average house price is approximately {avg_price:.2f}.",
        f"Lift availability affects price. Average prices by lift: {price_by_lift}.",
        f"Parking availability affects price. Average prices by parking: {price_by_parking}.",
        f"Furnishing impacts price. Average prices by furnishing: {price_by_furnishing}.",
        f"Correlation between built-up area and price is {area_price_corr:.2f}."
    ]

    for insight in analytics:
        documents.append(
            Document(
                page_content=insight,
                metadata={"source": "analytics"}
            )
        )

    for _, row in df.iterrows():
        text = (
            f"This property has {row['bathrooms']} bathrooms, "
            f"a built-up area of {row['built_up_area_numeric_in_sq_ft']} sq ft, "
            f"is {row['furnishing']} furnished, "
            f"priced at {row['price_numeric']}. "
            f"Lift availability is {row['Lift']} and parking is {row['Parking']}."
        )

        documents.append(
            Document(
                page_content=text,
                metadata={"source": "dataset"}
            )
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

else:
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

app = FastAPI(title="House Pricing RAG System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Backend running"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    docs = retriever.invoke(request.question)

    if not docs:
        return {"question": request.question, "answer": "I don't know"}

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
Use ONLY the information below to answer the question.
If the answer is not present, say "I don't know".

Information:
{context}

Question:
{request.question}
"""

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return {
        "question": request.question,
        "answer": response.choices[0].message.content.strip(),
        "sources": list(set(doc.metadata["source"] for doc in docs))
    }
