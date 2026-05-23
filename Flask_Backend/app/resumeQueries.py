from flask import Flask, request, jsonify
import fitz
import cloudinary
import cloudinary.uploader
import os
import tempfile
import nltk

from dotenv import load_dotenv
from pyresparser import ResumeParser
from groq import Groq

# Load environment variables
load_dotenv()

# Download required nltk data
nltk.download('stopwords')

# Flask app
app = Flask(__name__)

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
)

# Function to get ATS score
def get_ATS(img_link):
    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    completion = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Give an ATS score to this resume. Give the score out of 100 and output only the score."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_link
                        }
                    }
                ]
            }
        ],
        temperature=0.7,
        max_tokens=100,
        top_p=1,
        stream=False
    )

    return completion.choices[0].message.content


# Upload PDF route
@app.route('/upload', methods=['POST'])
def upload_pdf():
    pdf_file = request.files.get('pdf')

    if not pdf_file:
        return jsonify({"error": "No file provided"}), 400

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, pdf_file.filename)

        # Save uploaded PDF
        pdf_file.save(pdf_path)

        try:
            # Open PDF
            doc = fitz.open(pdf_path)

            # Convert first page to image
            page = doc.load_page(0)
            pix = page.get_pixmap()

            image_path = os.path.join(temp_dir, "converted_image.png")
            pix.save(image_path)

            doc.close()

            # Upload image to Cloudinary
            upload_result = cloudinary.uploader.upload(
                image_path,
                transformation={
                    'width': 3000,
                    'height': 3000,
                    'crop': 'limit'
                }
            )

            image_url = upload_result['secure_url']

            # Get ATS score
            ats_score = get_ATS(image_url)

            return jsonify({
                "ats_score": ats_score
            }), 200

        except Exception as e:
            return jsonify({
                "error": str(e)
            }), 500


# Run app
if __name__ == '__main__':
    app.run(debug=True)