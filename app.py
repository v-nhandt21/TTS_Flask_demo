#!/usr/bin/env python3
DEV=0
from flask import Flask, render_template, request,json
import sys

from vinorm import TTSrawUpper

sys.path.append('Service/')
sys.path.append('Service/waveglow/')
import warnings
warnings.filterwarnings("ignore")
if DEV:
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


        text = TTSrawUpper(text)
        if DEV:
            text = getAudio(text)

        text_hashed=abs(hash(text)) % (10 ** 8)
        audio="static/audio/"+str(text_hashed)+'.wav'
        return render_template('demo_audio.html', text=text,audio=audio)
    else:
        return render_template('demo_input.html')
        #return "Nhan"


@app.route("/sample" , methods=['GET', 'POST'])
def sample():
    with open("static/sample.txt", "r" ,encoding="utf-8") as f:
        lines = f.readlines()
        print(lines)
    return render_template('sample.html',lines=lines)

@app.route("/record" , methods=['GET', 'POST'])
def record():
    with open("static/record.txt", "r" ,encoding="utf-8") as f:
        lines = f.read().splitlines()
        print(lines)
        lines = lines[0:100]
    return render_template('record.html',lines=lines)

if __name__ == "__main__":
    app.run(port=5018)

