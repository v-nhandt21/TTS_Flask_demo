#!/usr/bin/env python3
from flask import Flask, render_template, request,json
import sys
sys.path.append('Service/')
sys.path.append('Service/waveglow/')
import warnings
warnings.filterwarnings("ignore")
from inference import getAudio
app = Flask(__name__, static_url_path='/static')



@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template('index.html')



@app.route("/demo" , methods=['GET', 'POST'])
def demo():
    if request.method == 'POST':
        if "back" in request.form:
            return render_template('demo_input.html')
        text = request.form['text']
        #text = getAudio(text)
        text_hashed=abs(hash(text)) % (10 ** 8)
        audio="static/audio/"+str(text_hashed)+'.wav'
        return render_template('demo_audio.html', text=text,audio=audio)
    else:
        return render_template('demo_input.html')
        #return "Nhan"



if __name__ == "__main__":
    app.run(port=5018)

