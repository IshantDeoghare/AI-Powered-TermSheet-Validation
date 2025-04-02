import os
import json
import re
import tempfile
import PyPDF2
from docx import Document
import pytesseract
from PIL import Image
import pandas as pd
from groq import Groq

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production use
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

class TermSheetValidator:
    def __init__(self):
        # Verify Tesseract installation
        self._verify_tesseract()
        
        # Replace with your actual Groq API key
        self.client = Groq(api_key="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")
        
        self.mandatory_fields = [
            'company_name',
            'investor_names',
            'investment_amount',
            'valuation',
            'security_type'
        ]
        self.current_model = "llama3-70b-8192"

    def _verify_tesseract(self):
        """Check Tesseract OCR installation"""
        try:
            pytesseract.get_tesseract_version()
        except EnvironmentError:
            print("""
            [ERROR] Tesseract OCR not found!
            Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
            Then add it to your PATH or set TESSERACT_CMD environment variable
            """)
            raise

    def read_file(self, file_path):
        """Extract text from various file formats"""
        try:
            if file_path.endswith('.pdf'):
                return self._read_pdf(file_path)
            elif file_path.endswith('.docx'):
                return self._read_docx(file_path)
            elif file_path.endswith('.xlsx'):
                return self._read_excel(file_path)
            elif file_path.endswith('.txt'):
                return self._read_text(file_path)
            elif file_path.endswith(('.png', '.jpg', '.jpeg')):
                return self._read_image(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            return ""

    def _read_pdf(self, file_path):
        """Handle PDF text extraction"""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ''.join([page.extract_text() for page in reader.pages])
            if not text.strip():
                raise ValueError("PDF appears to be image-based - use OCR")
            return text

    def _read_docx(self, file_path):
        """Handle DOCX files"""
        doc = Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])

    def _read_excel(self, file_path):
        """Handle Excel files"""
        df = pd.read_excel(file_path)
        return '\n'.join(f"{col}: {val}" for col, val in zip(df.columns, df.iloc[0]))

    def _read_text(self, file_path):
        """Handle plain text files"""
        with open(file_path, 'r') as f:
            return f.read()

    def _read_image(self, file_path):
        """Handle image OCR with better error handling"""
        try:
            # Adjust the tesseract_cmd path if necessary for your OS
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            return pytesseract.image_to_string(Image.open(file_path))
        except Exception as e:
            print(f"OCR Failed: {str(e)}")
            return ""

    def extract_entities(self, text):
        """Use Groq API to extract structured entities from text"""
        prompt = f"""Extract financial entities from this term sheet text. 
Return JSON with: company_name, investor_names (array), investment_amount, 
valuation (with type and amount), security_type, and optional_fields.

Text: {text}

Format:
{{
    "company_name": "string",
    "investor_names": ["string"],
    "investment_amount": "currency",
    "valuation": {{
        "type": "pre-money|post-money",
        "amount": "currency"
    }},
    "security_type": "string",
    "optional_fields": {{
        "capitalization_table": "boolean",
        "liquidation_preference": "string"
    }}
}}"""
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.current_model,
                temperature=0.2,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error in Groq API: {e}")
            return {}

    def validate_entities(self, entities):
        """Validate extracted entities against business rules"""
        results = {}
        
        for field in self.mandatory_fields:
            field_value = entities.get(field)
            results[field] = {
                'present': field_value is not None,
                'valid': self.validate_field(field, field_value)
            }
        
        return {
            'field_validation': results,
            'all_valid': all([v['present'] and v['valid'] for v in results.values()])
        }

    def validate_field(self, field, value):
        """Field-specific validation rules"""
        if value is None:
            return False
            
        if field == 'investment_amount':
            return isinstance(value, str) and self.validate_currency(value)
            
        if field == 'valuation':
            return (
                isinstance(value, dict) and
                value.get('type') in ['pre-money', 'post-money'] and
                isinstance(value.get('amount'), str) and
                self.validate_currency(value.get('amount', ''))
            )
            
        if field == 'security_type':
            return isinstance(value, str) and value.lower() in [
                'common stock', 'preferred stock', 
                'convertible note', 'safe', 'warrant'
            ]
            
        return True

    def validate_currency(self, value):
        """Validate currency format"""
        if not isinstance(value, str):
            return False
        return re.match(r'^\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?$', value) is not None

    def process(self, file_path):
        """End-to-end processing pipeline"""
        try:
            text = self.read_file(file_path)
            if not text:
                return {"error": "Failed to extract text"}
            
            entities = self.extract_entities(text)
            if not entities:
                return {"error": "Failed to extract entities"}
            
            validation = self.validate_entities(entities)
            
            # Convert all objects to JSON-serializable format
            return {
                "extracted_text": (text[:500] + "...") if isinstance(text, str) else str(text),
                "entities": self.make_json_serializable(entities),
                "validation": self.make_json_serializable(validation)
            }
        
        except Exception as e:
            return {"error": str(e)}

    def make_json_serializable(self, data):
        """Convert non-serializable objects to serializable formats"""
        if isinstance(data, (str, int, float, bool)):
            return data
        elif isinstance(data, dict):
            return {k: self.make_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self.make_json_serializable(item) for item in data]
        elif hasattr(data, '__dict__'):
            return self.make_json_serializable(data.__dict__)
        else:
            return str(data)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Render the interactive front-end
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Endpoint to handle file uploads and run the validator."""
    try:
        # Save the uploaded file temporarily
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process the file using TermSheetValidator
        validator = TermSheetValidator()
        result = validator.process(file_path)
        
        # Clean up the temporary file
        os.remove(file_path)
        
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
