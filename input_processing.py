from PyPDF2 import PdfReader
import docx
from openai import OpenAI
import base64
import streamlit as st
from openai import OpenAI
import base64
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_image(image_file):
    try:
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Construct the image data for vision input
        image_data = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}"
            }
        }

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful medical assistant. Extract the text exactly from the image provided."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please extract the text from this image."},
                        image_data
                    ]
                }
            ],
            max_tokens=2048
        )

        return response.choices[0].message.content

    except Exception as e:
        import streamlit as st
        st.error(f"❌ OCR failed: {e}")
        return None


    except Exception as e:
        import streamlit as st
        st.error(f"❌ OCR failed: {e}")
        return None

def extract_text_from_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8")
        elif uploaded_file.name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif uploaded_file.name.endswith(".docx"):
            doc = docx.Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
    return None
