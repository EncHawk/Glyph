import os
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS 
from llm import inferModel
app = Flask(__name__)
CORS(app, resources={
    r'/*':{'origin':'http://localhost:5500'}})


upload_folder = os.path.join(os.path.dirname(__file__),'uploads')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_files

@app.route('/status')
def home():
    return '<p>200,SERVER UP AND RUNNING.</p>'

allowed_files = ['pdf','xlsx','docx']
@app.route('/rag', methods=['GET','POST']) # post req to receive the files (1) for rag with the prompt, and send the response back.
def rag():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'no file found, try again', 401
        file = request.files['file']
        print(file.filename)
        filename = file.filename
        if filename and allowed_file(filename):
            content = file.read()
            file.save(os.path.join(upload_folder,filename))
            print('file saved')
            filepath = os.path.join(os.path.dirname(__file__),'reads',filename)
            print("content: ",len(content))

            with open(filepath,'wb') as pd:
                pd.write(content)
        
        if filename=='':
            return 'no file found, try again', 401
        return 'file was successfully saved, shd be followed with a mini summary', 200
    else:
        return 'GET the fuck out', 405

if __name__ =="__main__":
    app.run(host="0.0.0.0", port="8080",debug=True)