import os
import sys
import subprocess
from manim import Manim
from flowchart import Flowchart
from rag import RagAgent
from langchain_hub import InferenceClient
from pydantic import BaseModel, ValidationError, constr
from langchain.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelResponse, ModelRequest
# this is essentially only gonna return the output of any of the tools RAG , manim or flowchart
class Agent:
    # @tool
    # def MANIM(self,query):
    #     instance = Manim()

    def __init__(self, session_id, attempts, query):
        self.session_id = session_id
        self.attempts = attempts
        self.query = query
        
        self.hf_llm = HuggingFaceEndpoint(
            repo_id="zai-org/GLM-4.7",
            task="text-generation",
            max_new_tokens=512,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        self.basic_llm = HuggingFaceEndpoint(
            repo_id="google-bert/bert-base-uncased",
            task="text-generation",
            max_new_tokens=512,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        self.advanced_model = ChatHuggingFace(llm=self.hf_llm)
        self.basic_model = ChatHuggingFace(llm=self.basic_llm)
        self.model = self.basic_model

    class ModelResponse(BaseModel):
        code:constr(min_length=1, strip_whitespace=True)
        className:str
    response_format = {
        "type":"json_schema",
        "json_schema":{
            "name" : "Model_Response",
            "schema":ModelResponse.model_json_schema(),
            "strict":True
        }
    }

    @tool
    def manim_tool(self,prompt):
        """
        Docstring for Manim
        
        :param prompt: the input prompt to generate the illustration code
        """
        class ModelResponse(BaseModel):
            code:constr(min_length=1, strip_whitespace=True)
            className:str
        response_format = {
            "type":"json_schema",
            "json_schema":{
                "name" : "Model_Response",
                "schema":ModelResponse.model_json_schema(),
                "strict":True
            }
        }
        try:
            instance = Manim(response_format=response_format, prompt=prompt)
            response_text = instance.inferModel(model=ModelResponse)
            file = os.path.abspath(f"generated-scripts/{response_text.className}.py")

            subprocess.run(
                ["manim", "-pql", file, response_text.className],
                check=True
            )
            return {"ok":True, "msg":"File Ran successfully, return mp4 string only for manim", "string":"AWS_HOST_STRING"}
        except Exception as e:
            return {'ok':False,"error":str(e)}

    @tool
    def contextual_text_tool(self,prompt):
        """
        Docstring for flowchart
        
        :param prompt: the prompt to generate the code for graphviz flowchart
        """
        class ModelResponse(BaseModel):
            code:constr(min_length=1, strip_whitespace=True)
            className:str
        response_format = {
            "type":"json_schema",
            "json_schema":{
                "name" : "Model_Response",
                "schema":ModelResponse.model_json_schema(),
                "strict":True
            }
        }
        llm = HuggingFaceEndpoint( # add your huggingface token, this shit free and good heck yeah!
            repo_id= "Qwen/Qwen2.5-7B-Instruct",
            temperature = 0.7,
        )
        model = ChatHuggingFace(llm=llm)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        instance = RagAgent(embeddings=embeddings,model=model)
        response_code = instance.inferModel(input=prompt)
        return response_code
    
    @tool
    def flowchart_tool(self,prompt):
            """
            Docstring for flowchart
            
            :param prompt: the prompt to generate the code for graphviz flowchart
            """
            class ModelResponse(BaseModel):
                code:constr(min_length=1, strip_whitespace=True)
                className:str
            response_format = {
                "type":"json_schema",
                "json_schema":{
                    "name" : "Model_Response",
                    "schema":ModelResponse.model_json_schema(),
                    "strict":True
                }
            }
            try:
                res = Flowchart(input=prompt)
                base_dir = os.path.dirname(os.path.abspath(__file__))
                gen_dir = os.path.join(base_dir, "generated-flowcharts")
                os.makedirs(gen_dir, exist_ok=True)
                gen = os.path.join(gen_dir, f"{res.className}.py")
                if len(res.code) > 10:
                    print('writing to a file')
                    with open (gen, 'w') as f:
                        f.write(res.code)
                os.makedirs("generated-scripts", exist_ok=True)
                file = os.path.abspath(f"generated-flowcharts/{res.className}.py")
                subprocess.run(
                    [sys.executable, file],
                    check=True
                )
                return {
                    "ok" :True,
                    "result": "image successfully created. take this!",
                    "string":"aws hosted string for the svg"
                }
            except Exception as e:
                return {
                    "ok":False,
                    "error":str(e)
                }

    def run_with_retry(self, tool, prompt):
        last_error = None

        for _ in range(self.attempts):
            result = tool(prompt)

            if result.get("ok"):
                return result

            last_error = result.get("error")
            prompt += f"\n\nPrevious error:\n{last_error}\nFix it."

        return {
            "ok": False,
            "error": f"Failed after {self.attempts} attempts",
            "last_error": last_error
        }


    def run_agent(self,placeholder):
        """
        Docstring for Agent
        
        :param self: Description
        :param query: user's input to be processed and classified to perform any of the provided tools
        """
        # self.query= query
        prompt = """
            YOUR TASK IS TO VALIDATE THE USER'S QUERY AND CHOOSE WHICH TOOL TO CALL, ENSURE THAT THE TOOLS ARE CALLED UNTIL THEY ARE NOT ERROR FREE.
            YOU HAVE ACCESS TO THE FOLLOWING TOOLS:
                -- MANIM : GENERATES A MANIM ILLUSTRATION OF .mp4 UPON BEING CALLED WITH THE ILLUSTRATION QUERY FROM THE USER,
                    YOUR TASK IS TO ENSURE THAT THE OUTPUT CODE DOES NOT COME OUT AS AN ERROR.
                -- GRAPHVIZ : GENERATES A FLOWCHART FOR THE MANIM ILLUSTRATION, THE USER MUST BE SENT THIS FIRST, UPON BEING APPROVED BY THE USER, 
                    THE MANIM TOOL MUST BE CALLED WITH THE FEATURES ASKED BY THE USER BEING CONCATENATED INTO THE MANIM TOOL'S PROMPT
                -- TEXT : THIS TOOL GENERATES THE TEXT FOR THE USER'S QUERY, THIS TOOL MUST BE CALLED WHEN THE USER HAS NOT SPECIFICALLY ASKED FOR AN ILLUSTRATION.
            STRICT RULES:
        """
        
        # tools = [manim_tool,flowchart_tool,contextual_text_tool]
        prompt = f"""
        You must choose exactly ONE word. nothing else, no extra unnecessary tokens. you are a middle man who needs to decide which operation to perform 
            based on the user's input, if its an illustration or any visualisation

        Options:
        - manim_tool
        - flowchart_tool

        Rules:
        - Output ONLY one word
        - No punctuation
        - No explanation
        - No extra text

        User request:
        {self.query}
        """
        response = self.model.invoke(prompt)
        choice = response.content.strip().lower()

        if "manim_tool" in choice:
            return self.run_with_retry(self.manim_tool, self.query)

        elif "flowchart_tool" in choice:
            return self.run_with_retry(self.flowchart_tool, self.query)

        else:
            return self.run_with_retry(self.flowchart_tool, self.query)
# based on what the llm says, choose which one to run        

        

