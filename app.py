from flask import Flask, render_template, request, redirect, url_for
import os
from pathlib import Path as p
from PIL import Image, ImageChops, ImageStat
import re
import textwrap
from transformers import AutoTokenizer
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
import warnings

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

def clean_component_name(name):
    return re.sub(r'[^\w\s]', '', name).strip()

def clean_title_name(name):
    return re.sub(r'[^\w\s]', '', name).strip()

def analyze_image(image, api_key, data):
    try:
        data_str = ", ".join(data)
        prompt = f"""
        You need to access the following list of components: {data_str}.
        Detect whether the image contains any of these components.
        List each component found in the image on a new line.
        Do not include descriptions or any extra information, only the component names.
        The names should be presented as a list, each on a separate line.
        """

        # Configure the API
        genai.configure(api_key=api_key)
        image_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

        # Generate content using the API
        response = image_model.generate_content([image, prompt])

        # Print the response for debugging
        print(f"Response: {response}")

        generated_text = response.text
        components = [line.strip() for line in generated_text.strip().splitlines() if line.strip()]
        print(f"Detected Components: {components}")
        return components
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"

def load_image(image_file):
    img = Image.open(image_file)
    return img

def find_image_in_folders(uploaded_image, folders):
    uploaded_image = uploaded_image.convert("L")
    for folder in folders:
        if not os.path.exists(folder):
            print(f"Warning: Directory {folder} does not exist.")
            continue
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                folder_image = Image.open(file_path).convert("L")
                folder_image = folder_image.resize(uploaded_image.size)
                diff = ImageChops.difference(uploaded_image, folder_image)
                stat = ImageStat.Stat(diff)
                mse = stat.mean[0] ** 2
                if mse == 0:
                    return folder
    return "No matching image found"

default_folders = [
    'C:\\Users\\chend\\Desktop\\New folder\\Mech\\Backend\\kb\\Aircraft',
    'C:\\Users\\chend\\Desktop\\New folder\\Mech\\Backend\\kb\\dataset',
    'C:\\Users\\chend\\Desktop\\New folder\\Mech\\Backend\\kb\\Diesel'
]

pdf_files = {
    'C:\\Users\\chend\\Desktop\\New folder\\Mech\\Backend\\kb\\Aircraft': 'C:\\Users\\chend\\Desktop\\New folder\\Mech\\Backend\\kb\\Aircraft\\Aircraft.pdf',
    'C:\\Users\\chend\\Desktop\\New folder\\Mech\\Backend\\kb\\dataset': 'C:\\Users\\chend\\Desktop\\New folder\\Mech\\Backend\\kb\\dataset\\ENJOMOR V12 Engine Model updated.pdf',
    'C:\\Users\\chend\\Desktop\\New folder\\Mech\\Backend\\kb\\Diesel': 'C:\\Users\\chend\\Desktop\\New folder\\Mech\\Backend\\kb\\Diesel\\Diesel KB.pdf'
}

@app.route('/uploads', methods=['GET', 'POST'])
def index():
    result = None
    title_data = []
    detected_components = []
    image_url = None  # To store the URL of the uploaded image
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            image = load_image(file_path)
            image_url = url_for('static', filename='uploads/' + file.filename)  # URL for displaying the image
            folder_name = find_image_in_folders(image, default_folders)
            print(f"Detected folder: {folder_name}")

            # Find the relevant PDF file based on the detected folder
            default_pdf_path = pdf_files.get(folder_name)
            if not default_pdf_path:
                result = "No matching PDF found for the detected folder."
                return render_template('index.html', result=result, image_url=image_url)

            pdf_loader = PyPDFLoader(default_pdf_path)
            pages = pdf_loader.load_and_split()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
            context = "\n\n".join(str(p.page_content) for p in pages)
            texts = text_splitter.split_text(context)

            num_texts = len(texts)
            k_value = min(5, num_texts)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", clean_up_tokenization_spaces=True)

            vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": k_value})

            text_model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key='AIzaSyDf_h2YY8IbuNmFaRj4DJSDUnTxXt2og6c',
                temperature=0.2,
                convert_system_message_to_human=True
            )

            qa_chain = RetrievalQA.from_chain_type(
                text_model,
                retriever=vector_index,
                return_source_documents=True
            )

            prompt = "List all the mechanical components available in the engine along with their names and I don't want any description like: Here are the mechanical components available in the ENJOMOR V12 engine. Then in each and every string you don't want to mention integers."
            result = qa_chain({"query": prompt})
            components_list = result["result"].split('\n')

            prompt_title = "What is the name of this mechanical object?"
            result_title = qa_chain({"query": prompt_title})
            title_list = result_title["result"].split('\n')

            clean_components = [clean_component_name(comp) for comp in components_list if clean_component_name(comp)]
            clean_titles = [clean_title_name(title) for title in title_list if clean_title_name(title)]

            stored_components = clean_components
            title_data = clean_titles
            print(title_data)
            print("\n")
            data = stored_components
            print(data)

            detected_components = analyze_image(image, 'AIzaSyDdRNV7YuLzJ8n-nYkIzmGMZPR8l8wVlks', data)
            result = {
                'title_data': to_markdown("\n".join(title_data)),
                'detected_components': detected_components
            }
    return result

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
