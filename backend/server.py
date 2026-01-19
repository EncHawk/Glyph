import os
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS 
from llm import inferModel
from rag import RagAgent
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
@app.route('/upload', methods=['GET','POST']) # post req to receive the files (1) for rag with the prompt, and send the response back.
def rag():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'no file found, try again', 401

        file = request.files['file']
        filename = file.filename

        if not filename or not allowed_file(filename):
            return 'invalid file', 400

        # the file path to storethe pdf locally.
        filepath = os.path.join(upload_folder, filename)

        file.save(filepath)

        print('file saved at:', filepath)

        stores(filepath)

        return 'upload successful', 200
    else:
        return 'GET the fuck out, invalid request method.', 403


@app.route('/response', methods=['GET'])
def rag_response(): 
    """
        returns the response that is generated from RAG class 
        todo : add special methods in rag class that generates the response, 
        then return that response form this endpoint  
    """
    llm = HuggingFaceEndpoint( # add your huggingface token, this shit free and good heck yeah!
        repo_id= "Qwen/Qwen2.5-7B-Instruct",
        temperature = 0.7,
    )
    model = ChatHuggingFace(llm=llm)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    rag = RagAgent(embeddings=embeddings, model=model)
    rag.store()
    
      

if __name__ =="__main__":
    app.run(host="0.0.0.0", port="8080",debug=True)