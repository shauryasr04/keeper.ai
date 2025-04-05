# /Users/shauryasr/Documents/textchat/Channing Way.m4a
from openai import OpenAI
from fpdf import FPDF
import speech_recognition as sr
import asyncio
import aiohttp
import openai
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.units import inch
import streamlit as st
import PyPDF2
import io
from pdf2image import convert_from_path
import os


def voice_input():
    def save_transcription(text):
        with open("/Users/shauryasr/streamlit-keeper/transcription2.txt", "a", encoding="utf-8") as f:
            f.write(text + "\n")

    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Use the microphone as the audio source
    with sr.Microphone() as source:
        print("Please speak something...")

        # Adjust for ambient noise and start listening
        recognizer.adjust_for_ambient_noise(source)
        while True:
            try:
                print("Listening...")
                audio_data = recognizer.listen(source)
                print("Recognizing...")

                # Recognize speech using Google Web Speech API
                text = recognizer.recognize_google(audio_data)
                print(f"You said: {text}")

                # Save transcription to file
                save_transcription(text)
                if text.lower() == "terminate":
                    st.write("Complete!")
                    break

            except sr.UnknownValueError:
                print("Sorry, I did not understand that.")
            except sr.RequestError as e:
                print(f"Could not request results from Google Web Speech API; {e}")
            except KeyboardInterrupt:
                print("Exiting...")
                
                break


def read_text_file(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text


# Replace with your OpenAI API key
api_key = "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"
openai.api_key = api_key


async def gpt_call_history(text):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Pretend as if you are a travel nurse and you've had the following conversation: {text}. You are going to be given a conversation between a travel nurse and a patient. I want you to summarize this conversation in exactly 1k words. Include a 200 to 300 word analysis for each finding. This detailed analysis should include the possible side effects if this were to increase, possible symptomps, and preventative measures. Have explanations for any medically technical words. Also, for any medical measurements, please compare those vitals with what is a normal range for people of different ages, etc. Please only return the response. Please do not use any special characters in your response. When writing new sections of this report, can you bold and underline those new sections? Also please dont use the # or * characters AT ALL. Keep it to plain text and letters a-z only.",
                    }
                ],
            },
        ) as response:
            data = await response.json()
            return data["choices"][0]["message"]["content"]


async def gpt_call_vitals(text):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Pretend as if you are a travel nurse and you're reading the summary of the following conversation: {text}. Include ONLY metrics and/or summaries of any vitals that may have been tracked in this conversation. Examples would include blood pressure, heart rate, oxygen level, etc. ONLY include these metrics, no explanations, no special characters in your response",
                    }
                ],
            },
        ) as response:
            data = await response.json()
            return data["choices"][0]["message"]["content"]


async def gpt_call_docNotes(text):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Pretend as if you are a travel nurse and you're reading the summary of the following conversation: {text}. Read this carefully, and analyze for anything concerning or important that may be worth to include if you were to send this to a doctor. Think about it as if you are summarizing this for a doctor so that he can know anything relevant and important for the patient. Keep this between 100-200 words. Only include your answer, do NOT use any special characters in your response.",
                    }
                ],
            },
        ) as response:
            data = await response.json()
            return data["choices"][0]["message"]["content"]


async def gpt_call_patientNotes(text):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Pretend as if you are a travel nurse and you're reading a summary of the following conversation: {text}. Read it carefully, and only output what you may believe would be important for the patient to know. This could be things such as acceptable ranges of health related vitals, reasons for why we track these vitals, tips to stay healthy, etc. I want you to summarize this conversation in exactly 100 - 200 words. Have explanations for any medically technical words. Please only return the response. Please do not use any special characters in your response. When writing new sections of this report, can you bold and underline those new sections? Also please dont use the # or * characters AT ALL. Keep it to plain text and letters a-z only.",
                    }
                ],
            },
        ) as response:
            data = await response.json()
            return data["choices"][0]["message"]["content"]


async def run_vitals(file_path):
    text = read_text_file(file_path)
    return await gpt_call_vitals(text)
    result_patientnotes = await gpt_call_patientNotes(text)

    
async def run_docNotes(file_path):
    text = read_text_file(file_path)
    return await gpt_call_docNotes(text)

async def run_patNotes(file_path):
    text = read_text_file(file_path)
    return await gpt_call_patientNotes(text)

def create_nursing_report_pdf(
        filename,
        nurse_name,
        date,
        patient_name,
        history,
        vitals,
        doctor_notes,
        patient_notes,
    ):
        doc = SimpleDocTemplate(filename, pagesize=letter)
        elements = []

        styles = getSampleStyleSheet()

        # Create a custom style for the wrapped text
        wrapped_style = ParagraphStyle(
            "WrappedStyle",
            parent=styles["Normal"],
            fontSize=10,
            leading=12,
            alignment=TA_LEFT,
        )

        # Create a custom style for the header text
        header_style = ParagraphStyle(
            "HeaderStyle",
            parent=styles["Normal"],
            fontSize=14,
            leading=16,
            alignment=TA_LEFT,
        )

        # Create a custom style for the nursing report title
        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Normal"],
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            underline=True,
        )

        # Header
        header_data = [
            [
                Paragraph("Nurse Name:", header_style),
                Paragraph(nurse_name, header_style),
            ],
            [Paragraph("Date:", header_style), Paragraph(date, header_style)],
        ]
        header_table = Table(header_data, colWidths=[100, 400])
        header_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        elements.append(header_table)

        # Add space after the header table
        elements.append(Spacer(1, 0.5 * inch))  # 0.5 inch space

        # Nursing Report Title
        elements.append(Paragraph(f"{patient_name} Nursing Report", title_style))
        elements.append(Spacer(1, 0.25 * inch))  # 0.25 inch space after the title

        # Rest of the code remains the same...

        # Patient History & Context
        elements.append(Paragraph("Patient History & Context:", styles["Heading2"]))
        history_table = Table([[Paragraph(history, wrapped_style)]], colWidths=[500])
        history_table.setStyle(
            TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 1, colors.black),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        elements.append(history_table)

        # Vitals
        elements.append(Paragraph("Vitals:", styles["Heading2"]))
        vitals_table = Table([[Paragraph(vitals, wrapped_style)]], colWidths=[500])
        vitals_table.setStyle(
            TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 1, colors.black),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        elements.append(vitals_table)

        # Notes for Doctor
        elements.append(Paragraph("Notes for Doctor:", styles["Heading2"]))
        doctor_notes_table = Table(
            [[Paragraph(doctor_notes, wrapped_style)]], colWidths=[500]
        )
        doctor_notes_table.setStyle(
            TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 1, colors.black),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        elements.append(doctor_notes_table)

        # Notes for Patient
        elements.append(Paragraph("Notes for Patient:", styles["Heading2"]))
        patient_notes_table = Table(
            [[Paragraph(patient_notes, wrapped_style)]], colWidths=[500]
        )
        patient_notes_table.setStyle(
            TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 1, colors.black),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        elements.append(patient_notes_table)

        doc.build(elements)

    

    


async def main():
    st.title("Keeper-Patient")
    st.subheader(
        "Convey new data to new patients like never before.."
    )
    if st.button("Start Recording"):
        voice_input()
    st.write("Click the button and start speaking...")

    file_path = "/Users/shauryasr/streamlit-keeper/transcription.txt"
    # Example usage remains the same
    # Example usage:
    patient_name = 'Shaurya Srivastava'
    nurse_name = 'Jack Jones'
    date = "2024-06-22"
    history = "Patient has a history of diabetes in his family. Patient has previously injured femur in March 2020. Patient has enormous penis syndrome"
    vitals = await run_vitals(file_path)
    doctor_notes = await run_docNotes(file_path)
    patient_notes = await run_patNotes(file_path)
    
    create_nursing_report_pdf(
        "nursing_report_summary.pdf",
        nurse_name,
        date,
        patient_name,
        history,
        vitals,
        doctor_notes,
        patient_notes,
    )
    st.title("Post Patient Information:")
    pdf_path = "/Users/shauryasr/streamlit-keeper/nursing_report_summary.pdf"

    # Convert PDF to images
    images = convert_from_path(pdf_path)

    # Save images to disk
    output_folder = "/Users/shauryasr/streamlit-keeper/pdf_imaged"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f"page_{i + 1}.png")
        image.save(image_path, 'PNG')
    
    st.image('/Users/shauryasr/streamlit-keeper/pdf_imaged/page_1.png')

if __name__ == "__main__":
    asyncio.run(main())
