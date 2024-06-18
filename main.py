from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import fitz  # PyMuPDF

app = FastAPI()

# Load DistilBERT model and tokenizer for question-answering
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)


class QuestionInput(BaseModel):
    question: str


@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()
    with open("uploaded.pdf", "wb") as f:
        f.write(content)
    return {"filename": file.filename}


@app.post("/ask_question/")
def ask_question(input: QuestionInput):
    # Read text from the uploaded PDF
    with fitz.open("uploaded.pdf") as doc:
        pdf_text = ""
        for page in doc:
            pdf_text += page.get_text()

    # Answer the question using the QA pipeline
    result = qa_pipeline(question=input.question, context=pdf_text)

    return {"answer": result['answer']}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
