# Multilingual-Text-Summarizer
# It is an NLP project where we can summarize the user given text in any language(Multilingual) and 
# also crosslingual text to understand quickly and easily. And we use BERT2BERT and SimpleT5 model for summarization.

from flask import Flask, render_template, request
from langdetect import detect
from googletrans import Translator
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(device)


def generate_summary(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="longest", max_length=1024).to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=5,
        max_length=150,
        min_length=50,
        length_penalty=1.5,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def detect_language(input_text):
    try:
        return detect(input_text)
    except:
        return "Could not detect language"
        
    
app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST','GET'])
def summarize():
    if request.method=='POST': 
        dest_lng= request.form['lang']  
        input_text = request.form.get('text1')
        translator = Translator()
        translation = translator.translate(input_text, dest="en")
        text1 = generate_summary(translation.text)
        translation = translator.translate(text1, dest= dest_lng)
        output_text = translation.text
        return render_template('index.html', input_text = input_text, output_text = output_text)
    elif request.method == 'GET':
        return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)
    