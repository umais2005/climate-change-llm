DIRECTORY_PATH = "./data/"
INDEX_NAME = "test"

import io
import json
import os
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from warnings import filterwarnings
filterwarnings("ignore")
# from google.auth.transport.requests import Request
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import InstalledAppFlow
import time
from langchain_pinecone import PineconeVectorStore
import re
from dotenv import load_dotenv


SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
TEMP_DOWNLOAD_DIR = './temp_downloads'


class DocumentProcessor:
    def __init__(self, directory_path="./data/", index_name="test",drive_folder_id ="1-50Qfs8g8DELeudumJNarNrS4WRA6-F6"):
        load_dotenv()
        self.drive_folder_id = drive_folder_id
        self.directory = directory_path
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")
        self.index_name = index_name
        self.vector_store = self.load_pinecone_vector_store()
        print("Document Processor initialized.")

    def load_pinecone_vector_store(self):
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("No Pinecone API key found in environment variables.")
        
        pc = Pinecone(api_key=pinecone_api_key)

        if self.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
                
        self.index = pc.Index(self.index_name)
        print(f"Pinecone vector store '{self.index_name}' loaded.")
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)
        return self.vector_store
    

    def get_file_paths_from_directory_and_subdirectories(self):
        """
        Get all PDF and DOCX file paths from the directory,
        including files in nested directories.
        """
        file_paths = []
        for root, _, files in os.walk(self.directory):
            for filename in files:
                if filename.endswith((".pdf", ".docx")):
                    # Construct the full file path and add it to the list
                    file_paths.append(os.path.join(root, filename))
        return file_paths


    def check_existing_docs_by_id(self, doc_ids):
        """
        Check if the document IDs (filenames) exist in the Pinecone index.
        """
        # print(doc_ids)
        existing_ids = set()
        for doc_id in doc_ids:
            try:
                response = self.index.fetch(ids=[doc_id])
                # print(response)
                if response["vectors"]:
                    existing_ids.add(doc_id)
            except Exception as e:
                print(f"Error checking document ID {doc_id}: {str(e)}")
        return existing_ids
    
    
    def authenticate_drive_with_service_account(self):
        service_account_info = st.secrets["google_drive"]["service_account_info"]
        service_account_info = json.loads(service_account_info)
        creds = service_account.Credentials.from_service_account_info(
            service_account_info, scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)


    def download_file_from_drive(self, service, file_id, file_name):
        os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)
        destination = os.path.join(TEMP_DOWNLOAD_DIR, file_name)
        request = service.files().get_media(fileId=file_id)
        with io.FileIO(destination, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"Fetching {int(status.progress() * 100)}% of {file_name}.")
        return destination
    

    def list_files_in_drive(self, service, folder_id):
        """Returns list of all files in a given folder and its subfolders each
        as dict with values:
        1.mimeType
        2. id
        3. name
        """
        files = []
        folder_queue = [self.drive_folder_id]
        while folder_queue:
            current_folder_id = folder_queue.pop(0)
            query = f"'{current_folder_id}' in parents"
            results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
            items = results.get('files', [])
            for item in items:
                if item['mimeType'] == 'application/vnd.google-apps.folder':
                    folder_queue.append(item['id'])
                else:
                    files.append(item)
        return files

    def process_and_add_documents_from_drive(self):
        service = self.authenticate_drive_with_service_account()
        files = self.list_files_in_drive(service, self.drive_folder_id)

        if not files:
            print("No documents found in the specified folder.")
            st.warning("No documents found in the specified folder.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)

        # Step 1: Extract filenames (without extensions) to use as IDs
        filenames = [os.path.splitext(file['name'])[0] for file in files]
        
        # Step 2: Check if the document IDs already exist in Pinecone
        existing_ids = self.check_existing_docs_by_id(filenames)

        # Step 3: Filter out files that already exist in Pinecone
        new_files = [file for file in files if os.path.splitext(file['name'])[0] not in existing_ids]

        if not new_files:
            print("No new documents to add.")
            st.success("No new documents to add.")
            return
        else:
            st.success("New files found. Fetching:")
        # Step 4: Process and add only the new files to Pinecone
        for file in new_files:
            file_id = file['id']
            file_name = file['name']
            st.write(f"Fetching file from Drive: {file_name}")

            # Download the file temporarily
            downloaded_path = self.download_file_from_drive(service, file_id, file_name)

            # Load the file using the appropriate loader
            if file_name.endswith(".pdf"):
                loader = PyMuPDFLoader(file_path=downloaded_path)
            elif file_name.endswith(".docx"):
                loader = Docx2txtLoader(file_path=downloaded_path)
            else:
                print(f"Unsupported file format: {file_name}")
                continue

            filename = os.path.splitext(file_name)[0]
            # print(f"Processing document: {filename}")
            st.write(f"Processing document: {filename}")
            
            # Add document-level dummy vector
            dummy_vector = [1.0] * 1536  # Ensure vector values are floats
            self.index.upsert(vectors=[(filename, dummy_vector)])  # Add document-level vector for tracking

            # Load and split the document into chunks
            file_docs = loader.load()
            chunks = text_splitter.split_documents(file_docs)
            print(f"Processed {len(chunks)} chunks from document {filename}")

            # Generate IDs for the chunks
            ids = [f"{filename}_chunk_{i}" for i, _ in enumerate(chunks)]
            
            # Add chunk-level vectors using add_documents
            self.vector_store.add_documents(documents=chunks, ids=ids)
            st.write(f"Document processing and vector store update complete for {filename}.")
            print(f"Document processing and vector store update complete for {filename}.")

            # Clean up the downloaded file
            os.remove(downloaded_path)



    def process_and_add_documents_from_local(self):
        """
        Process new PDF and DOCX documents from the directory and add them to Pinecone.
        Only new documents (not already in Pinecone) are processed.
        """
        # Step 1: Get all PDF and DOCX file paths from the directory and its subdirectories
        file_paths = self.get_file_paths_from_directory_and_subdirectories()

        # Step 2: Extract filenames (without extensions) to use as IDs
        filenames = [os.path.splitext(os.path.basename(path))[0] for path in file_paths]

        # Step 3: Check if the document IDs already exist in Pinecone
        existing_ids = self.check_existing_docs_by_id(filenames)

        # Step 4: Filter out paths of files that already exist
        new_file_paths = [path for path in file_paths if os.path.splitext(os.path.basename(path))[0] not in existing_ids]

        if not new_file_paths:
            print("No new documents to add.")
            return

        # Initialize text splitter for chunking documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)

        # Step 5: Process new files
        for file_path in new_file_paths:
            # Load PDF or DOCX file
            if file_path.endswith(".pdf"):
                loader = PyMuPDFLoader(file_path=file_path)
                print(f"Processing PDF: {file_path}")
            elif file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path=file_path)
                print(f"Processing DOCX: {file_path}")
            else:
                print(f"Unsupported file format: {file_path}")
                continue
            filename = os.path.splitext(os.path.basename(file_path))[0]
            print(filename)
            # Step 5: Add document-level dummy vector
            dummy_vector = [1.0] * 1536  # Ensure vector values are floats
            self.index.upsert(vectors=[(filename, dummy_vector)])  # Add document-level vector for tracking

            # Step 6: Load and split the document into chunks
            file_docs = loader.load()
            chunks = text_splitter.split_documents(file_docs)
            print(f"Processed {len(chunks)} chunks from document {filename}")

            # Generate IDs for the chunks
            ids = [f"{filename}_chunk_{i}" for i, _ in enumerate(chunks)]
            
            # Step 7: Add chunk-level vectors using add_documents
            self.vector_store.add_documents(documents=chunks, ids=ids)
            print("Document processing and vector store update complete.")

    
# Example usage:
if __name__ == "__main__": ## TESTING ##
    # folder_id = "1ptf4zFnw0Lu5sJMMDIqwY2WxDVj29kz1"
    processor = DocumentProcessor(INDEX_NAME)
    processor.process_and_add_documents_from_drive()
    # existing_ids = processor.check_existing_docs_by_id(files)

    # processor.process_and_add_documents_from_local()
    # filepaths = processor.get_file_paths_from_directory_and_subdirectories()
    # print(filepaths)
    # processor.extract_filenames_from_vector_ids(filenames)
    #     print(processor.check_existing_docs_by_id(filenames))
