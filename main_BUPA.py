# # Import necessary libraries
# import os, re
# from flask import Flask, render_template, request
# from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
# import openai
# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from PyPDF2 import PdfReader
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from langchain_community.document_loaders import DirectoryLoader
# from langchain.chains import LLMChain
# import json

# # Get the OpenAI API key from the environment variable

# api_key = "YOUR KEY HERE"

# if api_key is None or api_key == "":
#     print("OpenAI API key not set or empty. Please set the environment variable.")
#     exit()  # Terminate the program if the API key is not set.

# # Initialize the OpenAI client with the API key
# os.environ['OPENAI_API_KEY'] = api_key
# FAISS_PATH = "/faiss"

# # Flask App
# app = Flask(__name__)

# vectorstore = None
# conversation_chain = None
# chat_history = []
# general_exclusion_list = ["HIV/AIDS", "Parkinson's disease", "Alzheimer's disease","pregnancy", "substance abuse", "self-inflicted injuries", "sexually transmitted diseases(std)", "pre-existing conditions"]

# def get_document_loader():
#     loader = DirectoryLoader('documents', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
#     docs = loader.load()
#     return docs

# def get_text_chunks(documents: list[Document]):
#     text_splitter = RecursiveCharacterTextSplitter(
        
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
        
#     )
#     chunks = text_splitter.split_documents(documents)
#     return chunks

# def get_embeddings():
#     documents = get_document_loader()
#     chunks = get_text_chunks(documents)
#     db = FAISS.from_documents(
#         chunks, OpenAIEmbeddings()
#     )
#     return db


# def get_retriever():
#     db = get_embeddings()
#     retriever = db.as_retriever()
#     return retriever

# def get_claim_approval_context():
#     db = get_embeddings()
#     context = db.similarity_search("What are the documents required for claim approval?")
#     claim_approval_context = ""
#     for x in context:
#         claim_approval_context += x.page_content

#     return claim_approval_context

# def get_general_exclusion_context():
#     db = get_embeddings()
#     context = db.similarity_search("Give a list of all general exclusions")
#     general_exclusion_context = ""
#     for x in context:
#         general_exclusion_context += x.page_content

#     return general_exclusion_context

# def get_file_content(file):
#     text = ""
#     if file.filename.endswith(".pdf"):
#         pdf = PdfReader(file)
#         for page_num in range(len(pdf.pages)):
#             page = pdf.pages[page_num]
#             text += page.extract_text()

#     return text

# def get_bill_info(data):
#     prompt = "Act as an expert in extracting information from medical invoices. You are given with the invoice details of a patient. Go through the given document carefully and extract the 'disease' and the 'expense amount' from the data. Return the data in json format = {'disease':"",'expense':""}"
#     messages=[
#         {"role": "system", 
#         "content": prompt}
#         ]
    
#     user_content = f"INVOICE DETAILS: {data}"

#     messages.append({"role": "user", "content": user_content})

#     response = openai.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=messages,
#                 temperature=0.4,
#                 max_tokens=2500)
        
#     data = json.loads(response.choices[0].message.content)

#     return data



# PROMPT = """You are an AI assistant for verifying health insurance claims. You are given with the references for approving the claim and the patient details. Analyse the given data and predict if the claim should be accepted or not. Use the following guidelines for your analysis.

# 1.Verify if the patient has provided all necessary information and all necessary documents
# and if you find any incomplete information or required documents are not provided then set INFORMATION criteria as FALSE and REJECT the claim.
# if patient has provided all required documents then set INFORMATION criteria as TRUE. 

# 2. If any disease mentioned in the medical bill of the patient is in the general exclusions list, set EXCLUSION criteria as FALSE and REJECT the claim.

# Use this information to verify if the application is valid and to accept or reject the application.

# DOCUMENTS FOR CLAIM APPROVAL: {claim_approval_context}
# EXCLUSION LIST : {general_exclusion_context}
# PATIENT INFO : {patient_info}
# MEDICAL BILL : {medical_bill_info}

# Use the above information to verify if the application is valid and decide if the application has to be accepted or rejected keeping the guidelines into consideration. 

# Generate a detailed report about the claim and procedures you followed for accepting or rejecting the claim and the write the information you used for creating the report. 
# Create a report in the following format

# Write whether INFORMATION AND EXCLUSION are TRUE or FALSE 
# Reject the claim if any of them is FALSE.
# Write whether claim is accepted or not. If the claim has been accepted, the maximum amount which can be approved will be {max_amount}

# Executive Summary
# [Provide a Summary of the report.]

# Introduction
# [Write a paragraph about the aim of this report, and the state of the approval.]

# Claim Details
# [Provide details about the submitted claim]

# Claim Description
# [Write a short description about claim]

# Document Verification
# [Mentions which documents are submitted and if they are verified.] 

# Document Summary
# [Give a summary of everything here including the medical reports of the patient]

# Please verify for any signs of fraud in the submitted claim if you find the documents required for accepting the claim for the medical treatment.
# """


# prompt = PromptTemplate(input_variables=["claim_approval_context", "general_exclusion_context", "patient_info","max_amount"], template=PROMPT)

# def check_claim_rejection(claim_reason, general_exclusion_list, prompt_template, threshold=0.4):
#     vectorizer = CountVectorizer()
#     patient_info_vector = vectorizer.fit_transform([claim_reason])

#     for disease in general_exclusion_list:
#         disease_vector = vectorizer.transform([disease])
#         similarity = cosine_similarity(patient_info_vector, disease_vector)[0][0]
#         if float(similarity) > float(threshold):
            
#             prompt_template = """You are an AI assistant for verifying health insurance claims. You are given with the references for approving the claim and the patient details. Analyse the given data and give a good rejection. You the following guidelines for your analysis.
#             PATIENT INFO : {patient_info}

#             Executive Summary
#                 [Provide a Summary of the report.]

#                 Introduction
#                 [Write a paragraph about the aim of this report, and the state of the approval.]

#                 Claim Details
#                 [Provide details about the submitted claim]

#                 Claim Description
#                 [Write a short description about claim]

#                 Document Verification
#                 [Mentions which documents are submitted and if they are verified.] 

#                 Document Summary
#                 [Give a summary of everything here including the medical reports of the patient]
            
#             CLAIM MUST BE REJECTED: Patient has {disease} which is present in the general exclusion list."""
#             return prompt_template
    
#     return prompt_template

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/', methods=['GET', 'POST'])
# def msg():
#     claim_validation_message = ""
#     name = request.form['name']
#     address = request.form['address']
#     claim_type = request.form['claim_type']
#     claim_reason = request.form['claim_reason']
#     date = request.form['date']
#     medical_facility = request.form['medical_facility']
#     medical_bill = request.files['medical_bill']
#     total_claim_amount = request.form['total_claim_amount']
#     description = request.form['description']

#     bill = get_file_content(medical_bill)

#     bill_info = get_bill_info(bill)
#     # If input amount is more than the bill amount - REJECT
#     if bill_info['expense'] != None and int(bill_info['expense']) < int(total_claim_amount) :
#         claim_validation_message = "The amount mentioned for claiming is more than the billed amount. Claim Rejected."
        
#         return render_template("result.html", name=name, address=address, claim_type=claim_type, claim_reason=claim_reason, date=date, medical_facility=medical_facility, total_claim_amount=total_claim_amount, description=description, output=claim_validation_message)
        
#     elif bill_info['expense'] != None and int(bill_info['expense']) > int(total_claim_amount) :
#         #Check if the disease is in the exclusion list or not, update the prompt accordingly
#         patient_info = f"Name: {name} " + f"\nAddress: {address} " + f"\nClaim type: {claim_type} " + f"\nClaim reason: {claim_reason}" + f"\nMedical facility: {medical_facility} " + f"\nDate : {date} " + f"\nTotal claim amount: {total_claim_amount}" + f"\nDescription: {description}"
#         medical_bill_info = f"Medical Bill: {bill}"
        
#         validated_prompt = check_claim_rejection(bill_info["disease"], general_exclusion_list,PROMPT)
    
#         prompt_template = PromptTemplate(input_variables=["claim_approval_context","patient_info"],template=validated_prompt)
#         llm = ChatOpenAI(model="gpt-3.5-turbo")
#         llmchain = LLMChain(llm=llm, prompt= prompt_template)
#         output = llmchain.run({"claim_approval_context": get_claim_approval_context(), "general_exclusion_context": get_general_exclusion_context(), "patient_info": patient_info, "medical_bill_info":medical_bill_info,"max_amount":total_claim_amount, "disease":bill_info["disease"]})
        
#         output = re.sub(r'\n', '<br>', output)
        
#         return render_template("result.html", name=name, address=address, claim_type=claim_type, claim_reason=claim_reason, date=date, medical_facility=medical_facility, total_claim_amount=total_claim_amount, description=description, output=output)
        
#     else:
#         #If no expense value has been extracted
#         output = "Please enter a valid Consultation Receipt."
        
#         return render_template("result.html", name=name, address=address, claim_type=claim_type, claim_reason=claim_reason, date=date, medical_facility=medical_facility, total_claim_amount=total_claim_amount, description=description, output=output)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8081)



# main_BUPA.py

import os
import re
import json
from flask import Flask, render_template, request
from dotenv import load_dotenv

# LangChain / Vector store / loaders
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document

# PDF and basic NLP
from PyPDF2 import PdfReader

import joblib
import difflib
import numpy as np
import base64
import fitz  # PyMuPDF

# Known diseases from training data
KNOWN_DISEASES = [
    "Flu", "Heart Attack", "Broken Leg", "Diabetes", "Hypertension", 
    "Appendicitis", "Cataract", "Pneumonia", "Covid-19", "Migraine"
]

# Models removed

# -----------------------------
# Env & OpenAI client
# -----------------------------
load_dotenv()  # reads OPENAI_API_KEY from .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env like OPENAI_API_KEY=\"sk-...\"")

# ChatOpenAI, OpenAIEmbeddings, etc. read the key from env automatically.
# No need to set os.environ again.

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

# Globals
general_exclusion_list = [
    "HIV/AIDS",
    "Parkinson's disease",
    "Alzheimer's disease",
    "pregnancy",
    "substance abuse",
    "self-inflicted injuries",
    "sexually transmitted diseases",
    "pre-existing conditions",
]

# -----------------------------
# Document helpers (RAG)
# -----------------------------
def get_document_loader() -> list[Document]:
    # Loads all PDFs in ./documents recursively
    loader = DirectoryLoader(
        "documents",
        glob="**/*.pdf",
        show_progress=True,
        loader_cls=PyPDFLoader,
    )
    docs = loader.load()
    return docs

def get_text_chunks(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_documents(documents)

def get_embeddings_db():
    documents = get_document_loader()
    chunks = get_text_chunks(documents)
    db = FAISS.from_documents(chunks, OpenAIEmbeddings())
    return db

def get_retriever():
    return get_embeddings_db().as_retriever()

def _join_search_results(results) -> str:
    return "".join(doc.page_content for doc in results)

def get_claim_approval_context() -> str:
    db = get_embeddings_db()
    return _join_search_results(db.similarity_search("What are the documents required for claim approval?"))

def get_general_exclusion_context() -> str:
    db = get_embeddings_db()
    return _join_search_results(db.similarity_search("Give a list of all general exclusions"))

# -----------------------------
# PDF handling
# -----------------------------
# -----------------------------
# PDF handling
# -----------------------------
# -----------------------------
# PDF handling
# -----------------------------
def get_pdf_text_via_vision(file_bytes) -> str:
    """
    Fallback: Convert PDF bytes to images (PyMuPDF) and use OpenAI Vision (GPT-4o) to extract text.
    """
    print("DEBUG: Starting Vision OCR...")
    text_out = ""
    try:
        # 1. Open PDF from bytes
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        # 2. Iterate pages
        for i, page in enumerate(doc):
            print(f"DEBUG: Vision Processing page {i+1}...")
            # Render page to image (PNG)
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            
            # Encode to base64
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            
            # 3. Call OpenAI Vision
            # We use a direct chat completion call here
            llm_vision = ChatOpenAI(model="gpt-4o", max_tokens=2000)
            
            response = llm_vision.invoke(
                [
                    {"role": "system", "content": "You are an OCR machine. Extract all text from this image exactly as it appears. Do not summarize."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Extract text from this medical bill page."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ]}
                ]
            )
            text_out += response.content + "\n"
            
    except Exception as e:
        print(f"DEBUG: Vision OCR Failed: {e}")
        return f" [Vision OCR Failed: {str(e)}]"

    return text_out

def get_file_content(file_storage) -> str:
    text = ""
    if file_storage and file_storage.filename.lower().endswith(".pdf"):
        # Read all bytes first to allow multiple passes.
        file_bytes = file_storage.read()
        file_storage.seek(0) # Reset for safety if needed elsewhere
        
        # 1. Try Standard PyPDF2 extraction
        try:
            from io import BytesIO
            reader = PdfReader(BytesIO(file_bytes))
            for page in reader.pages:
                text += (page.extract_text() or "")
        except Exception as e:
            print(f"DEBUG: PyPDF2 failed: {e}")

        # 2. If text is too short, try Vision OCR Fallback
        if len(text.strip()) < 50:
            print(f"DEBUG: Text too short ({len(text.strip())} chars). Attempting Vision OCR...")
            ocr_text = get_pdf_text_via_vision(file_bytes)
            
            if len(ocr_text.strip()) > 10 and "Failed" not in ocr_text:
                print(f"DEBUG: Vision OCR successful. Extracted {len(ocr_text)} chars.")
                text = ocr_text # Replace entirely as the original was garbage/empty
            else:
                print(f"DEBUG: Vision OCR result: {ocr_text}")
                # If explicit failure, we might want to show it or leave empty to trigger 'Unreadable' error

    return text

# -----------------------------
# Bill parsing via LLM
# -----------------------------
def safe_json_loads(s: str):
    # Try strict JSON first, then try to recover { ... } substring
    try:
        return json.loads(s)
    except Exception:
        # Extract first {...} block
        import re as _re
        m = _re.search(r"\{.*\}", s, flags=_re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}

def get_bill_info(data: str) -> dict:
    """
    Ask the LLM to extract 'disease' and 'expense' from invoice text.
    Returns a dict like {'disease': '...', 'expense': '12345'} (values may be missing).
    """
    system_prompt = (
        "Act as an expert in extracting information from medical invoices. "
        "You are given the invoice details of a patient. Carefully extract the "
        "'disease', 'expense amount', 'patient_name', 'date', and 'medical_facility' from the data. "
        "Return ONLY valid JSON in the exact format: "
        "{\"disease\": \"...\", \"expense\": \"...\", \"patient_name\": \"...\", \"date\": \"...\", \"medical_facility\": \"...\"}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"INVOICE DETAILS:\n{data}"},
    ]

    # Use LangChain's ChatOpenAI (reads key from env)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    resp = llm.invoke(messages)  # returns an AIMessage with .content

    parsed = safe_json_loads(resp.content or "")
    # Normalize keys
    disease = parsed.get("disease")
    expense = parsed.get("expense")
    patient_name = parsed.get("patient_name")
    bill_date = parsed.get("date")
    medical_facility = parsed.get("medical_facility")
    
    return {
        "disease": disease, 
        "expense": expense,
        "patient_name": patient_name,
        "date": bill_date,
        "medical_facility": medical_facility
    }

def verify_claim_details(form: dict, bill: dict) -> tuple[bool, str]:
    """
    Verifies if the form details match the extracted bill details.
    Returns (True, "") if valid, or (False, "reason") if invalid.
    """
    # 1. Verify Name (Fuzzy Match)
    form_name = form.get("name", "").lower().strip()
    bill_name = (bill.get("patient_name") or "").lower().strip()
    
    if bill_name:
        # Simple similarity check
        matcher = difflib.SequenceMatcher(None, form_name, bill_name)
        if matcher.ratio() < 0.6: # Allow some OCR/Typo flexibility
            return False, f"Patient Name Mismatch: Form says '{form.get('name')}', Bill says '{bill.get('patient_name')}'"

    # 2. Verify Hospital (Fuzzy Match)
    form_hospital = form.get("medical_facility", "").lower().strip()
    bill_hospital = (bill.get("medical_facility") or "").lower().strip()
    
    if bill_hospital:
        matcher = difflib.SequenceMatcher(None, form_hospital, bill_hospital)
        if matcher.ratio() < 0.4: # Hospital names can be tricky (e.g. "Apollo" vs "Apollo Hospitals")
            return False, f"Medical Facility Mismatch: Form says '{form.get('medical_facility')}', Bill says '{bill.get('medical_facility')}'"

    # 3. Verify Amount (Hard check - already existed but moving logic here is cleaner, 
    #    but we'll keep the existing amount check in the main route to minimize refactor risk)
    
    return True, ""

# -----------------------------
# Claim prompt(s)
# -----------------------------
MAIN_PROMPT = """You are an AI assistant for verifying health insurance claims. You are given references for approving the claim and the patient details. Analyze the data and decide if the claim should be accepted or rejected.

Rules:
1) Verify if all required information and documents are present.
   - If any required info/docs are missing => set INFORMATION = FALSE and REJECT the claim.
   - Else set INFORMATION = TRUE.
2) If any disease in the medical bill appears in the general exclusions list => set EXCLUSION = FALSE and REJECT the claim.
   - Else set EXCLUSION = TRUE.

DOCUMENTS FOR CLAIM APPROVAL:
{claim_approval_context}

EXCLUSION LIST:
{general_exclusion_context}

PATIENT INFO:
{patient_info}

MEDICAL BILL:
{medical_bill_info}

If the claim is accepted, the maximum approvable amount is {max_amount}.

Output a detailed report in this structure:

INFORMATION: [TRUE/FALSE]
EXCLUSION: [TRUE/FALSE]
CLAIM DECISION: [ACCEPTED or REJECTED]
MAX APPROVABLE AMOUNT (if accepted): {max_amount}

Executive Summary
[Summary]

Introduction
[Aim of the report and state of approval]

Claim Details
[Submitted claim details]

Claim Description
[Short description]

Document Verification
[Which documents were submitted and verified]

Document Summary
[Summary including medical reports]
"""

REJECTION_PROMPT = """You are an AI assistant for verifying health insurance claims. Based on the patient details, produce a clear rejection report.

PATIENT INFO:
{patient_info}

Claim must be REJECTED because the patient has a general-exclusion condition: {disease}.

Executive Summary
[Summary]

Introduction
[Aim of the report and state of approval]

Claim Details
[Submitted claim details]

Claim Description
[Short description]

Document Verification
[Which documents were submitted and verified]

Document Summary
[Summary including medical reports]
"""

# Full set of variables used by MAIN_PROMPT:
MAIN_TEMPLATE = PromptTemplate(
    input_variables=[
        "claim_approval_context",
        "general_exclusion_context",
        "patient_info",
        "medical_bill_info",
        "max_amount",
    ],
    template=MAIN_PROMPT,
)

REJECTION_TEMPLATE = PromptTemplate(
    input_variables=["patient_info", "disease"],
    template=REJECTION_PROMPT,
)

def check_claim_rejection(claim_reason: str, exclusions: list[str], threshold: float = 0.4):
    """
    Simple similarity check: if claim_reason is similar to any disease in exclusions
    beyond the threshold, return True and the matched disease.
    """
    vectorizer = CountVectorizer().fit([claim_reason] + exclusions)
    claim_vec = vectorizer.transform([claim_reason])
    for disease in exclusions:
        sim = cosine_similarity(claim_vec, vectorizer.transform([disease]))[0][0]
        if float(sim) > float(threshold):
            return True, disease
    return False, None

# -----------------------------
# Routes
# -----------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    # POST: process form
    form = request.form
    files = request.files

    name = form.get("name", "")
    address = form.get("address", "")
    claim_type = form.get("claim_type", "")
    claim_reason = form.get("claim_reason", "")
    date = form.get("date", "")
    medical_facility = form.get("medical_facility", "")
    total_claim_amount_str = form.get("total_claim_amount", "0")
    description = form.get("description", "")
    medical_bill_file = files.get("medical_bill")

    # Safely parse amount
    try:
        total_claim_amount = int(float(total_claim_amount_str))
    except Exception:
        total_claim_amount = 0

    print(f"DEBUG: Processing file: {medical_bill_file.filename if medical_bill_file else 'None'}")
    
    # Pass the file object (FileStorage) directly to our smart function
    bill_text = get_file_content(medical_bill_file) if medical_bill_file else ""

    print(f"DEBUG: Total extracted extracted len: {len(bill_text)}")

    bill_info = get_bill_info(bill_text) if bill_text else {"disease": None, "expense": None}

    # --- Enforce Bill Readability ---
    if not bill_text or len(bill_text.strip()) < 10:
        return render_template(
            "result.html",
            name=name,
            address=address,
            claim_type=claim_type,
            claim_reason=claim_reason,
            date=date,
            medical_facility=medical_facility,
            total_claim_amount=total_claim_amount,
            description=description,
            output=f"Error: Could not read text (len={len(bill_text)}). Ensure PDF is text-based."
        )
            
    # --- Verify Identity Details ---
    print(f"DEBUG: Extracted Bill Info: {bill_info}")
    is_valid, validation_msg = verify_claim_details(form, bill_info)
    print(f"DEBUG: Verification Result: {is_valid}, Msg: {validation_msg}")
    
    if not is_valid:
            return render_template(
            "result.html",
            name=name,
            address=address,
            claim_type=claim_type,
            claim_reason=claim_reason,
            date=date,
            medical_facility=medical_facility,
            total_claim_amount=total_claim_amount,
            description=description,
            output=f"Verification Failed: {validation_msg}"
        )
    # ------------------------------------

    # If input amount > bill amount => reject
    try:
        bill_expense = int(float(bill_info["expense"])) if bill_info.get("expense") else None
    except Exception:
        bill_expense = None

    if bill_expense is not None and bill_expense < total_claim_amount:
        msg = "The amount claimed is more than the billed amount. Claim Rejected."
        return render_template(
            "result.html",
            name=name,
            address=address,
            claim_type=claim_type,
            claim_reason=claim_reason,
            date=date,
            medical_facility=medical_facility,
            total_claim_amount=total_claim_amount,
            description=description,
            output=msg,
        )

    # Build patient/bill info strings
    patient_info = (
        f"Name: {name}\n"
        f"Address: {address}\n"
        f"Claim type: {claim_type}\n"
        f"Claim reason: {claim_reason}\n"
        f"Medical facility: {medical_facility}\n"
        f"Date: {date}\n"
        f"Total claim amount: {total_claim_amount}\n"
        f"Description: {description}"
    )
    medical_bill_info = f"{bill_text[:15000]}"  # avoid massive prompts

    # Check for exclusion-based rejection shortcut
    disease_text = bill_info.get("disease") or ""
    must_reject, matched_disease = check_claim_rejection(disease_text, general_exclusion_list)

    if must_reject and matched_disease:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
        chain = LLMChain(llm=llm, prompt=REJECTION_TEMPLATE)
        output = chain.run({"patient_info": patient_info, "disease": matched_disease})
    else:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
        chain = LLMChain(llm=llm, prompt=MAIN_TEMPLATE)
        output = chain.run(
            {
                "claim_approval_context": get_claim_approval_context(),
                "general_exclusion_context": get_general_exclusion_context(),
                "patient_info": patient_info,
                "medical_bill_info": medical_bill_info,
                "max_amount": str(total_claim_amount),
            }
        )

    # Predict Fair Price
    # Features removed.

    # Render
    output_html = re.sub(r"\n", "<br>", output or "No output generated.")
    return render_template(
        "result.html",
        name=name,
        address=address,
        claim_type=claim_type,
        claim_reason=claim_reason,
        date=date,
        medical_facility=medical_facility,
        total_claim_amount=total_claim_amount,
        description=description,
        output=output_html,
    )

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    # Use a common dev port; change if needed
    app.run(host="0.0.0.0", port=8081, debug=True)
