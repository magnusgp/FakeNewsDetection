# Load the libraries
from fastapi import FastAPI, HTTPException
from joblib import load

# Load the model
clf = load(open('src/models/model_placeholder.pkl','rb'))

# Process the input


# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to the Fake News Detector API"}

# Define the route to the sentiment predictor
@app.post("/predict_if_fake_news")
def predict(text_message):

    polarity = ""

    if(not(text_message)):
        raise HTTPException(status_code=400, 
                            detail = "Please Provide a valid text message")

    prediction = clf.predict(([text_message]))

    if(prediction[0] == 0):
        polarity = "True"

    elif(prediction[0] == 1):
        polarity = "False"
        
    return {
            "text_message": text_message, 
            "sentiment_polarity": polarity
           }