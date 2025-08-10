import os
import sqlite3
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from PIL import Image
import pytesseract
import re
from transformers import BartForConditionalGeneration, BartTokenizer
import assemblyai as aai
from extractor import extract_information_from_text, create_pdf

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
app.secret_key = ""  

USERS = {"admin": "12345"}
aai.settings.api_key = ""  

tokenizer = BartTokenizer.from_pretrained("./models/bart-fine-tuned-mts")
model = BartForConditionalGeneration.from_pretrained("./models/bart-fine-tuned-mts")

# SQLite database initialization
DB_PATH = "patients.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patients 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  name TEXT, 
                  age TEXT, 
                  gender TEXT, 
                  history TEXT, 
                  summary TEXT)''')
    conn.commit()
    conn.close()

# Call this when the app starts
init_db()

def clean_ocr_text(ocr_text):
    lines = ocr_text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line: 
            continue
        if re.match(r"^\d{5,}$", line): 
            continue
        if re.search(r"@|\.com|www", line, re.IGNORECASE):  
            continue
        if re.match(r"^Page\s\d+", line, re.IGNORECASE):  
            continue
        if len(line.split()) <= 2 and re.search(r"\d{2,4}", line):  
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if username in USERS and USERS[username] == password:
        session['user'] = username
        return jsonify({"success": True})
    return jsonify({"success": False})

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login_page'))

@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login_page'))
    return render_template('index.html')

@app.route('/ocr', methods=['POST'])
def ocr():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized, please log in"}), 401
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']
    extracted_text = pytesseract.image_to_string(Image.open(image))
    cleaned_text = clean_ocr_text(extracted_text)
    
    return jsonify({"extracted_text": cleaned_text})

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized, please log in"}), 401
    ocr_text = request.json.get("ocr_text", "")
    additional_text = request.json.get("additional_text", "")
    audio_text = request.json.get("transcription", "")

    if not ocr_text and not additional_text and not audio_text:
        return jsonify({"error": "No input provided"}), 400

    extracted_info = extract_information_from_text(ocr_text, additional_text, audio_text)

    combined_info = {
        "Name": extracted_info.get("Name", "Not specified"),
        "Age": extracted_info.get("Age", "Not specified"),
        "Gender": extracted_info.get("Gender", "Not specified"),
        "Disease": extracted_info.get("Disease", "Not specified"),
        "Medications": extracted_info.get("Medications", "None specified") if extracted_info.get("Medications") != "Not Available" else "None specified",
        "Discharge Date": extracted_info.get("Discharge Date")
    }

    combined_text = (
        "Generate a discharge summary based on the following clinical information:\n"
        f"Patient Name: {combined_info['Name']}\n"
        f"Age: {combined_info['Age']}\n"
        f"Gender: {combined_info['Gender']}\n"
        f"Disease/Condition: {combined_info['Disease']}\n"
        f"Medications: {', '.join(combined_info['Medications']) if isinstance(combined_info['Medications'], list) else combined_info['Medications']}\n"
        f"Discharge Date: {combined_info['Discharge Date']}\n"
        "Additional Notes from Image Report:\n" + (ocr_text if ocr_text else "None provided") + "\n"
        "Additional Notes from Text Input:\n" + (additional_text if additional_text else "None provided") + "\n"
        "Additional Notes from Doctor-Patient Conversation:\n" + (audio_text if audio_text else "None provided") + "\n"
        "Provide a concise discharge summary incorporating all relevant details."
    ).strip()

    inputs = tokenizer(combined_text, return_tensors="pt", max_length=4096, truncation=True)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=1000,
        min_length=100,
        num_beams=4,
        early_stopping=True,
        length_penalty=1.0
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = re.sub(r'\s+', ' ', summary).strip()

    # Pass the summary to create_pdf
    create_pdf(extracted_info, summary)

    # Store data in SQLite database using extracted_info
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO patients (name, age, gender, history, summary) 
                 VALUES (?, ?, ?, ?, ?)''',
              (extracted_info.get('Name', 'Not specified'),
               extracted_info.get('Age', 'Not specified'),
               extracted_info.get('Gender', 'Not specified'),  
               extracted_info.get('Medical History', 'Not specified'),
               summary))
    conn.commit()
    conn.close()

    print("\n--- Consolidated Extracted Information ---")
    for key, value in extracted_info.items():
        print(f"{key}: {value}")
    print("-----------------------------\n")

    return jsonify({"summary": summary})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized, please log in"}), 401
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if audio_file:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        audio_path = os.path.join('uploads', audio_file.filename)
        audio_file.save(audio_path)

        try:
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(audio_path)
            if transcript.status == aai.TranscriptStatus.error:
                return jsonify({"error": transcript.error}), 500
            else:
                return jsonify({"transcription": transcript.text})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            try:
                os.remove(audio_path)
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}")

@app.route('/view_data')
def view_data():
    if 'user' not in session:
        return redirect(url_for('login_page'))
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, age, gender, history, summary FROM patients")
    data = c.fetchall()
    conn.close()
    return render_template('view_data.html', patients=data)

if __name__ == '__main__':
    app.run(debug=True)