import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from twilio.rest import Client
from transformers import pipeline
import spacy
import re

# Load spaCy's named entity recognition model
nlp = spacy.load('en_core_web_sm')

# Initialize the QA model
qa_model = pipeline("question-answering", model="ktrapeznikov/biobert_v1.1_pubmed_squad_v2")

def extract_information_from_text(ocr_text="", additional_text="", audio_text="", summary=""):
    """
    Extracts clinical information from combined OCR, additional text, and audio inputs.
    Args:
        ocr_text (str): Text from image report.
        additional_text (str): Manually entered text.
        audio_text (str): Transcribed audio text.
        summary (str): Summary text to be included in the PDF.
    Returns:
        dict: Consolidated clinical information.
    """
    # Combine all inputs into one context with markers
    combined_text = f"[OCR] {ocr_text}\n[Additional] {additional_text}\n[Audio] {audio_text}".strip()
    combined_text = re.sub(r'[¢«§]', '', combined_text)  # Clean OCR artifacts
    combined_text = re.sub(r'\s+', ' ', combined_text).strip()  # Normalize whitespace

    # Define questions for extracting clinical details
    questions = {
        "Name": "What is the patient's name?",
        "Age": "What is the patient's age?",
        "Gender": "What is the patient's gender?",
        "Medical History": "What relevant medical history does the patient have?",
        "Examination Findings": "What are the examination findings?",
        "Medications": "What medications were administered or prescribed to the patient?",
        "Procedures": "What medical procedures were performed, including surgeries or interventions?",
    }

    # Extract answers for each question from combined text
    extracted_info = {}
    # Load scispaCy's biomedical model for NER
    nlp_med = spacy.load("en_ner_bc5cdr_md")

    # Extract answers for each question from combined text
    extracted_info = {}
    for key, question in questions.items():
        if key == "Medical History":
            # Step 1: Use scispaCy to extract DISEASE entities, focusing on patient-specific history
            doc = nlp_med(combined_text)
            conditions = []
            for ent in doc.ents:
                if ent.label_ == "DISEASE":
                    condition_text = ent.text
                    # Find the sentence containing the entity
                    sentence = None
                    sentence_idx = -1
                    for idx, sent in enumerate(doc.sents):
                        if sent.start_char <= ent.start_char <= sent.end_char:
                            sentence = sent
                            sentence_idx = idx
                            break
                    if sentence:
                        sentence_text = sentence.text.lower()
                        # Check for negation in the same sentence
                        negation_patterns = [
                            r'\bno history of\b',
                            r'\bdoes not have\b',
                            r'\bnot have\b',
                            r'\bwithout\b',
                            r'\bno evidence of\b',
                            r'\bnot diagnosed with\b',
                            r'\b(not that i\'m aware of|not that i know of)\b',
                            r'\bno\b',
                            r'\bnot\b',
                            r'\bnever had\b',
                            r'\babsence of\b',
                            r'\bnegative for\b'
                        ]
                        is_negated = False
                        for pattern in negation_patterns:
                            if re.search(pattern, sentence_text):
                                is_negated = True
                                break
                        # If the sentence is a question, check the next sentence for negation
                        if not is_negated:
                            is_question = bool(re.search(r'\b(do you|have you|are there|is there|any|what|when|where|how|did you|can you)\b.*\?|[\?\!]', sentence_text))
                            if is_question:
                                # Look for negation in the next sentence
                                next_sentence = None
                                if sentence_idx + 1 < len(list(doc.sents)):
                                    next_sentence = list(doc.sents)[sentence_idx + 1]
                                if next_sentence:
                                    next_sentence_text = next_sentence.text.lower()
                                    for pattern in negation_patterns:
                                        if re.search(pattern, next_sentence_text):
                                            is_negated = True
                                            break
                        # Check if the condition pertains to the patient or family
                        is_family_history = bool(re.search(r'\b(in your family|family history|parents|siblings|family|relatives)\b', sentence_text))
                        # Check if the condition is a potential diagnosis (e.g., "suggestive of", "likely")
                        is_potential_diagnosis = bool(re.search(r'\b(suggestive of|likely|possible|probable|may be|consistent with|indicative of)\b', sentence_text))
                        # Check if the condition is explicitly attributed to the patient
                        is_patient_specific = bool(re.search(r'\b(the patient|patient|i|he|she)\b.*\b(has|had|history of|diagnosed with)\b', sentence_text))
                        if not is_negated and not is_family_history and not is_question and not is_potential_diagnosis and is_patient_specific:
                            conditions.append(condition_text)

            # Step 2: Rule-based extraction for "history of" phrases, focusing on patient-specific history
            history_phrases = re.findall(r'\bhistory of\s+([A-Za-z\s\d]+?)(?:,|\.|including|$)', combined_text, re.IGNORECASE)
            for phrase in history_phrases:
                phrase = phrase.strip()
                if phrase and "including" not in phrase.lower():
                    # Find the sentence containing the "history of" phrase
                    phrase_start = combined_text.lower().find(f"history of {phrase.lower()}")
                    if phrase_start != -1:
                        sentence = None
                        sentence_idx = -1
                        for idx, sent in enumerate(doc.sents):
                            if sent.start_char <= phrase_start <= sent.end_char:
                                sentence = sent
                                sentence_idx = idx
                                break
                        if sentence:
                            sentence_text = sentence.text.lower()
                            # Check for negation in the same sentence
                            negation_patterns = [
                                r'\bno history of\b',
                                r'\bdoes not have\b',
                                r'\bnot have\b',
                                r'\bwithout\b',
                                r'\bno evidence of\b',
                                r'\bnot diagnosed with\b',
                                r'\b(not that i\'m aware of|not that i know of)\b',
                                r'\bno\b',
                                r'\bnot\b',
                                r'\bnever had\b',
                                r'\babsence of\b',
                                r'\bnegative for\b'
                            ]
                            is_negated = False
                            for pattern in negation_patterns:
                                if re.search(pattern, sentence_text):
                                    is_negated = True
                                    break
                            # If the sentence is a question, check the next sentence for negation
                            if not is_negated:
                                is_question = bool(re.search(r'\b(do you|have you|are there|is there|any|what|when|where|how|did you|can you)\b.*\?|[\?\!]', sentence_text))
                                if is_question:
                                    # Look for negation in the next sentence
                                    next_sentence = None
                                    if sentence_idx + 1 < len(list(doc.sents)):
                                        next_sentence = list(doc.sents)[sentence_idx + 1]
                                    if next_sentence:
                                        next_sentence_text = next_sentence.text.lower()
                                        for pattern in negation_patterns:
                                            if re.search(pattern, next_sentence_text):
                                                is_negated = True
                                                break
                            # Check if the condition pertains to the patient or family
                            is_family_history = bool(re.search(r'\b(in your family|family history|parents|siblings|family|relatives)\b', sentence_text))
                            # Check if the condition is a potential diagnosis
                            is_potential_diagnosis = bool(re.search(r'\b(suggestive of|likely|possible|probable|may be|consistent with|indicative of)\b', sentence_text))
                            # Check if the condition is explicitly attributed to the patient
                            is_patient_specific = bool(re.search(r'\b(the patient|patient|i|he|she)\b.*\b(has|had|history of|diagnosed with)\b', sentence_text))
                            if not is_negated and not is_family_history and not is_question and not is_potential_diagnosis and is_patient_specific:
                                conditions.append(phrase)

            # Step 3: Remove duplicates while preserving order
            seen = set()
            conditions = [cond for cond in conditions if not (cond in seen or seen.add(cond))]
            
            # Convert to a comma-separated string
            answer = ", ".join(conditions) if conditions else "Not Available"
        elif key == "Medications":
            doc = nlp_med(combined_text)
            # Extract entities labeled as "CHEMICAL" (which includes drugs/medications)
            medications = []
            for ent in doc.ents:
                if ent.label_ == "CHEMICAL":
                    medications.append(ent.text)
            # Remove duplicates while preserving order
            seen = set()
            medications = [med for med in medications if not (med in seen or seen.add(med))]
            # Convert to a comma-separated string
            answer = ", ".join(medications) if medications else "Not Available"
        elif key == "Procedures":
            # Use BioBERT to extract procedures as a list
            procedures = []
            remaining_context = combined_text
            max_iterations = 5  # Prevent infinite loops, max 5 procedures
            for _ in range(max_iterations):
                result = qa_model(question=question, context=remaining_context)
                if result['score'] < 0.01:
                    break
                procedure = result['answer'].strip()
                # Clean the procedure name 
                procedure = procedure.lower().replace(" procedure", "").strip()
                if procedure:
                    procedures.append(procedure)
                remaining_context = remaining_context.replace(procedure, "", 1)
                # If the context becomes too short or no more procedures 
                if len(remaining_context.strip()) < 10:
                    break
            seen = set()
            procedures = [proc for proc in procedures if not (proc in seen or seen.add(proc))]
            answer = ", ".join(procedures) if procedures else "Not Available"
        else:
            result = qa_model(question=question, context=combined_text)
            answer = result['answer'].strip() if result['score'] > 0.01 else "Not Available"
        
        # Post-process Gender
        if key == "Gender":
            if answer == "Not Available":
                # for "male" or "female" in the text
                match = re.search(r'\b(male|female)\b', combined_text, re.IGNORECASE)
                if match:
                    answer = match.group(0).capitalize()
            else:
                match = re.search(r'\b(male|female)\b', answer, re.IGNORECASE)
                if match:
                    answer = match.group(0).capitalize()
                else:
                    answer = "Not Available"
    
        # Post-process Age to extract only the numerics
        if key == "Age":
            if answer == "Not Available":
                # conversational patterns like "I'm 29" or "29 years old"
                match = re.search(r'(?:I\'m|I am|age is)\s*(\d+)|(\d+)\s*(?:years old|years)', combined_text, re.IGNORECASE)
                if match:
                    answer = match.group(1) if match.group(1) else match.group(2)
                else:
                    match = re.search(r'\bAge\s*:\s*(\d+)\b', combined_text, re.IGNORECASE)
                    if match:
                        answer = match.group(1)
            else:                
                match = re.search(r'\b\d+\b', answer)
                if match:
                    answer = match.group(0)
                else:
                    answer = "Not Available"
        
        # Post-process Name 
        if key == "Name" and answer != "Not Available":
            if "Dr." in answer:
                answer = "Not Available"
            else:
                answer = answer.replace(" PID", "")
        
        extracted_info[key] = answer

    # If name is not found by QA, use SpaCy on combined text
    if extracted_info['Name'] == "Not Available":
        doc = nlp(combined_text)
        for ent in doc.ents:
            if ent.label_ == 'PERSON' and "Dr." not in ent.text and "Radiologist" not in ent.text:
                extracted_info['Name'] = ent.text.strip()
                extracted_info['Name'] = extracted_info['Name'].replace(" PID", "")
                break

    # Add discharge date and time
    today_date = datetime.today().strftime('%Y-%m-%d')
    current_time = datetime.today().strftime('%H:%M:%S')
    extracted_info["Discharge Date"] = today_date
    extracted_info["Discharge Time"] = current_time

    return extracted_info

def create_pdf(extracted_info, summary):
    """
    Create a PDF for the patient's discharge summary.
    Args:
        extracted_info (dict): Dictionary containing extracted information from the text inputs.
        summary (str): Summary text to be included in the PDF.
    """
    # PDF generation
    pdf_path = os.path.join("static", "pdfs", "discharge_summary.pdf")
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 40, "Patient Discharge Summary")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, height - 80, f"Patient Name: {extracted_info['Name']}")
    c.drawString(100, height - 100, f"Discharge Date: {extracted_info['Discharge Date']}")
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.5)
    c.line(100, height - 110, width - 100, height - 110)
    c.setFont("Helvetica", 10)
    c.drawString(100, height - 140, "Discharge Summary Details:")

    y = height - 160
    for key, value in extracted_info.items():
        if y < 40:
            c.showPage()
            y = height - 40
            c.setFont("Helvetica", 10)
        c.drawString(100, y, f"{key}: {value}")
        y -= 20

    c.setFont("Helvetica-Bold", 12)
    y -= 20
    c.drawString(100, y, "SUMMARY")
    c.setFont("Helvetica", 10)
    y -= 20



    current_line = ""
    char_count = 0

    for char in summary:
        current_line += char
        char_count += 1
        
        if char_count >= 100:
            # If we end in the middle of a word 
            if char != " " and char != summary[-1]:
                # Find the last space
                last_space = current_line.rfind(" ")
                if last_space != -1 and last_space < len(current_line) - 1:
                    # If there's a space in the line break there
                    to_print = current_line[:last_space]
                    current_line = current_line[last_space + 1:]
                    c.drawString(100, y, to_print.strip())
                else:
                    to_print = current_line[:-1] + "-"
                    current_line = current_line[-1:]
                    c.drawString(100, y, to_print)
            else:
                c.drawString(100, y, current_line.strip())
                current_line = ""
            
            y -= 20
            char_count = len(current_line)
            

            if y < 40:
                c.showPage()
                y = height - 40
                c.setFont("Helvetica", 10)


    if current_line:
        c.drawString(100, y, current_line.strip())

    c.setFont("Helvetica-Bold", 10)
    c.drawString(100, y - 20, "Approved by: ABC Hospital")
    c.setFont("Helvetica-Bold", 8)
    c.drawString(100, y - 40, "Additional Notes")
    c.drawString(100, y - 60, "In case of Emergency, contact:")
    c.drawString(100, y - 80, "0494-2763225")
    c.save()
    


    account_sid = ""  
    auth_token = ""  
    client = Client(account_sid, auth_token)

    pdf_url = "https://.ngrok-free.app/static/pdfs/discharge_summary.pdf"

    recipient_phone_number = ""
    message = client.messages.create(
        from_='whatsapp:',
        body="Here is your discharge summary PDF.",
        media_url=[pdf_url],
        to=f"whatsapp:{recipient_phone_number}"
    )

    print(f"Message sent to {recipient_phone_number}: {message.sid}")
    print(f"Discharge summary PDF has been saved at {pdf_path}")
