# Load the libraries
from fastapi import FastAPI, HTTPException
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from fastapi.responses import HTMLResponse

# Load the model
model_path = r"models/roberta-base/checkpoint-30/"

# Load the roberta tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route 
@app.get("/")
def root():
    return HTMLResponse(content="""<html>
    <body>
    <h1>Welcome to the Fake News Detector API</h1>
    <a href="/input_text/">Enter your text message here</a>
    </body>
    </html>""")

# Define the route for the input text prompt
@app.get("/input_text/")
def input_text():
    return HTMLResponse(content="""<html>
    <body>
    <h1>Enter your text message here:</h1>
    <form action="/predict_if_fake_news/" method="get">
    <input type="text" name="text_message">
    <input type="submit" value="Submit">
    </form>
    </body>
    </html>""")

# Define the route to the text predictor
@app.get("/predict_if_fake_news/")
def predict(text_message):
    # make the text message a string
    text_message = str(text_message)
    # remove all question marks and percentage signs
    text_message = text_message.replace("%", " ")
    text_message = text_message.replace("?", "")
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path)
    classify = pipeline("text-classification", model=model_path, tokenizer=tokenizer)
    # return htmlresponse that presents both the text message and the prediction
    return HTMLResponse(content="""<html>
    <body>
    <h1>Text Message: {}</h1>
    <h1>Prediction: {}</h1>
    </body>
    </html>""".format(text_message, classify(text_message)[0]['label']))