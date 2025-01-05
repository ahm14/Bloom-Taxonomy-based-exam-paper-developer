import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import pytesseract
from PIL import Image
import pdfplumber
import docx
from io import BytesIO
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import logging

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize LLM
llm = ChatGroq(temperature=0.5, groq_api_key="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", model_name="llama3-8b-8192")

# Initialize Pinecone for vector storage
PINECONE_API_KEY = "pcsk_6PtxDh_6tortuWyNhXdmVrAjx1ZSv8bQRcbgbE7j3JtwwcpMCkFfdsp6VC925WxmqpNYQC"
pc = Pinecone(api_key=PINECONE_API_KEY)

cloud = os.getenv('PINECONE_CLOUD', 'aws')
region = os.getenv('PINECONE_REGION', 'us-east-1')

spec = ServerlessSpec(cloud=cloud, region=region)

index_name = "syllabus-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        spec=spec
    )

index = pc.Index(index_name)

# Initialize embedding model
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# OCR Configuration for Pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Adjust to your system's path

# Function to extract text, images, tables, and formulas from PDF
def extract_pdf_data(pdf_path):
    data = {"text": "", "tables": [], "images": []}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract Text
                data["text"] += page.extract_text() or ""
                # Extract Tables
                tables = page.extract_tables()
                for table in tables:
                    data["tables"].append(table)
                # Extract Images
                for image in page.images:
                    base_image = pdf.extract_image(image["object_number"])
                    image_obj = Image.open(BytesIO(base_image["image"]))
                    data["images"].append(image_obj)
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
    return data

# Function to extract text from DOCX files
def extract_docx_data(docx_file):
    doc = docx.Document(docx_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to extract text from plain text files
def extract_text_file_data(text_file):
    return text_file.read().decode('utf-8')

# Function to extract text from images using OCR
def extract_text_from_images(images):
    ocr_text = ""
    for image in images:
        ocr_text += pytesseract.image_to_string(image) + "\n"
    return ocr_text

# Function to process extracted content (PDF, DOCX, etc.)
def process_content(file_data, file_type="pdf"):
    text = ""
    images = []
    if file_type == "pdf":
        pdf_data = extract_pdf_data(file_data)
        text = process_pdf_content(pdf_data)
        images = pdf_data["images"]
    elif file_type == "docx":
        text = extract_docx_data(file_data)
    elif file_type == "txt":
        text = extract_text_file_data(file_data)

    ocr_text = extract_text_from_images(images)
    return text + "\n" + ocr_text

# Function to process PDF content
def process_pdf_content(pdf_data):
    # Process OCR text from images
    ocr_text = extract_text_from_images(pdf_data["images"])
    combined_text = pdf_data["text"] + ocr_text

    # Process tables into readable text
    table_text = ""
    for table in pdf_data["tables"]:
        table_rows = [" | ".join(row) for row in table]
        table_text += "\n".join(table_rows) + "\n"

    return combined_text + "\n" + table_text

# Function to add syllabus to vector database
def add_syllabus_to_index(syllabus_text):
    sentences = syllabus_text.split(". ")
    embeddings = embedder.encode(sentences, batch_size=32, show_progress_bar=True)
    for i, sentence in enumerate(sentences):
        index.upsert([(f"sentence-{i}", embeddings[i].tolist(), {"text": sentence})])

# Function to retrieve relevant syllabus content
def retrieve_relevant_content(query):
    try:
        query_embedding = embedder.encode([query])
        results = index.query(vector=query_embedding.tolist(), top_k=5, include_metadata=True)
        relevant_content = "\n".join([match["metadata"]["text"] for match in results["matches"]])
        return relevant_content
    except Exception as e:
        logging.error(f"Error retrieving content: {e}")
        return ""

# Function to generate questions
def generate_questions(question_type, subject_name, syllabus_context, num_questions, difficulty_level):
    prompt_template = f"""
    Based on the following syllabus content, generate {num_questions} {question_type} questions. Ensure the questions are directly derived from the provided syllabus content.

    Subject: {subject_name}
    Syllabus Content: {syllabus_context}

    Difficulty Levels:
    - Remember: {difficulty_level.get('Remember', 0)}
    - Understand: {difficulty_level.get('Understand', 0)}
    - Apply: {difficulty_level.get('Apply', 0)}
    - Analyze: {difficulty_level.get('Analyze', 0)}
    - Evaluate: {difficulty_level.get('Evaluate', 0)}
    - Create: {difficulty_level.get('Create', 0)}

    Format questions as follows:
    Q1. ________________

    Q2. ________________

    ...
    """
    chain = (ChatPromptTemplate.from_template(prompt_template) | llm | StrOutputParser())
    try:
        return chain.invoke({})
    except Exception as e:
        logging.error(f"Error generating {question_type} questions: {e}")
        return ""

# Function to generate answers
def generate_answers(questions, syllabus_context):
    prompt = f"""
    Based on the provided syllabus content, generate detailed answers for the following questions. The answers must only be based on the syllabus content.

    Syllabus Content: {syllabus_context}

    Questions:
    {questions}

    Format answers as follows:
    Answer 1: ________________
    Answer 2: ________________
    ...
    """
    chain = (ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser())
    try:
        return chain.invoke({})
    except Exception as e:
        logging.error(f"Error generating answers: {e}")
        return ""

# Streamlit app
st.title("Bloom Taxonomy Based Exam Paper Developer")

# Sidebar inputs
instructor_name = st.sidebar.text_input("Instructor")
class_name = st.sidebar.text_input("Class")
institution_name = st.sidebar.text_input("Institution")
subject_name = st.sidebar.text_input("Subject")

# Syllabus Upload
uploaded_file = st.sidebar.file_uploader("Upload Syllabus (PDF, DOCX, TXT, Image)", type=["pdf", "docx", "txt", "png", "jpg"])
syllabus_text = None
if uploaded_file:
    file_type = uploaded_file.type.split("/")[1]
    st.sidebar.markdown("✅ Syllabus uploaded")
    syllabus_text = process_content(uploaded_file, file_type)
    add_syllabus_to_index(syllabus_text)

# Preview of Syllabus
if syllabus_text:
    st.subheader("Syllabus Preview:")
    st.text_area("Extracted Content", syllabus_text[:1000], height=300)

# Question Type Selection
question_type = st.sidebar.radio("Select Question Type", ("MCQs", "Short Questions", "Long Questions", "Fill in the Blanks", "Case Studies", "Diagram-based"))
difficulty_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
difficulty = {level: st.sidebar.slider(level, 0, 5, 1) for level in difficulty_levels}
num_questions = st.sidebar.number_input("Number of Questions", min_value=1, max_value=50, value=10)

# Instructor Feedback Option
feedback = st.sidebar.text_area("Instructor Feedback (Optional)")

# Generate Questions
if st.sidebar.button("Generate Questions"):
    if syllabus_text:
        with st.spinner(f"Generating {question_type}..."):
            syllabus_context = retrieve_relevant_content(f"Generate {question_type} based on syllabus")
            st.session_state.generated_questions = generate_questions(question_type, subject_name, syllabus_context, num_questions, difficulty)
        st.text_area(f"Generated {question_type}", value=st.session_state.generated_questions, height=400)
    else:
        st.error("Please upload a syllabus before generating questions.")

# Generate Answers
if st.sidebar.button("Generate Answers for Questions"):
    if "generated_questions" in st.session_state and st.session_state.generated_questions:
        with st.spinner("Generating answers..."):
            syllabus_context = retrieve_relevant_content("Generate answers from syllabus")
            st.session_state.generated_answers = generate_answers(st.session_state.generated_questions, syllabus_context)
        st.text_area("Generated Answers", value=st.session_state.generated_answers, height=400)
    else:
        st.error("Generate questions first before generating answers.")

# Download Options
if "generated_questions" in st.session_state and st.session_state.generated_questions:
    st.sidebar.download_button(
        label="Download Questions",
        data=st.session_state.generated_questions,
        file_name=f"{subject_name}_questions.txt",
        mime="text/plain",
    )

if "generated_answers" in st.session_state and st.session_state.generated_answers:
    st.sidebar.download_button(
        label="Download Answers",
        data=st.session_state.generated_answers,
        file_name=f"{subject_name}_answers.txt",
        mime="text/plain",
    )

# Application Footer
st.markdown("""
---
**Advanced Test Paper Generator** - powered by LangChain, Pinecone, and Streamlit.
""")
