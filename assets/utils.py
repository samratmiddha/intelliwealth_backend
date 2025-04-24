try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen
import certifi
import json
import pandas as pd
from datetime import datetime

def get_stock_data(ticker, api_key):
    """Get comprehensive stock data from multiple endpoints"""
    all_data = {}
    
    quote_url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={api_key}"
    response = urlopen(quote_url, cafile=certifi.where())
    print(f"quote_url: {quote_url}")
    quote_data = json.loads(response.read().decode("utf-8"))
    print(f"quote_data: {quote_data}")
    if quote_data and isinstance(quote_data, list):
        all_data["quote"] = quote_data[0]
    
    try:
        ratios_url = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}?apikey={api_key}&limit=1"
        response = urlopen(ratios_url, cafile=certifi.where())
        ratios_data = json.loads(response.read().decode("utf-8"))
        print(f"ratios_url: {ratios_url}")
        print(f"ratios_data: {ratios_data}")
        if ratios_data and isinstance(ratios_data, list):
            all_data["ratios"] = ratios_data[0]
    except Exception:
        print("Error fetching ratios data")
    
    try:
        metrics_url = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?apikey={api_key}&limit=1"
        response = urlopen(metrics_url, cafile=certifi.where())
        metrics_data = json.loads(response.read().decode("utf-8"))
        print(f"metrics_url: {metrics_url}")
        print(f"metrics_data: {metrics_data}")
        if metrics_data and isinstance(metrics_data, list):
            all_data["metrics"] = metrics_data[0]
    except Exception:
        print("Error fetching metrics data")
    
    try:
        profile_url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
        response = urlopen(profile_url, cafile=certifi.where())
        profile_data = json.loads(response.read().decode("utf-8"))
        print(f"profile_url: {profile_url}")
        print(f"profile_data: {profile_data}")
        if profile_data and isinstance(profile_data, list):
            all_data["profile"] = profile_data[0]
    except Exception:
        print("Error fetching profile data")
    
    return all_data

def prepare_financial_data(data):
    """Transform nested data into flattened DataFrame"""
    flattened_data = {}
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    flattened_data["data_retrieved_date"] = current_date
    flattened_data["analysis_date"] = current_date
    
    for section, section_data in data.items():
        if isinstance(section_data, dict):
            for key, value in section_data.items():
                flattened_data[f"{section}_{key}"] = value
    
    df = pd.DataFrame([flattened_data])
    df["data_description"] = (
        f"This is current financial data for the stock as of {current_date}. "
        "It includes quote information, financial ratios, key metrics, and company profile data. "
        "All monetary values are in USD unless otherwise specified."
    )
    
    return df

import re

def run_langchain_query(df, question):
    """Process the DataFrame with LangChain to answer the financial query using the asset's JSON data as intermediary"""
    import json
    from langchain_core.documents import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    from langchain_community.llms import Ollama

    json_data = df.to_dict(orient="records")[0]
    doc_content = json.dumps(json_data, indent=2)
    # documents = [Document(page_content=doc_content)]
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # texts = text_splitter.split_documents(documents)
    # hg_embeddings = HuggingFaceEmbeddings()
    # PERSIST_DIR = "docs/chroma_rag/"
    # vectorstore = Chroma.from_documents(
    #     documents=texts,
    #     collection_name="stock_data",
    #     embedding=hg_embeddings,
    #     persist_directory=PERSIST_DIR
    # )
    # retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    template = """
            IMPORTANT INSTRUCTION: You are a Financial Analyst providing information SOLELY based on the data provided below. 
            Do NOT refer to any other knowledge about this company or defer to external sources. Give the result in a markdown format.
            
            CURRENT DATA: 
            {context}
            
            USER QUERY: {question}
            
            YOUR TASK:
            1. Analyze ONLY the data provided above
            2. Answer the query directly using ONLY this data
            3. If specific information isn't in the data, clearly state what IS available and provide that information instead
            4. Format your response in a professional, easy-to-read format
            5. Do NOT suggest visiting websites or getting data elsewhere
            6. Do NOT apologize for limitations - just work with what you have
            7. ASSUME ALL DATA IS CURRENT AND ACCURATE
            
            RESPONSE (using only the data shown above):
            """
    PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    llm = Ollama(model="mistral", temperature=0.01)
    prompt_str = PROMPT.format(context=doc_content, question=question)
    markdown_report = llm.invoke(prompt_str)
    # Parse the result string into a dict before returning
    # parsed = parse_financial_report(result)
    return {"raw": markdown_report}