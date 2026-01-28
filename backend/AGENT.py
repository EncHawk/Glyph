import os
import sys
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


    def Agent(self,):
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
        messages=[
            {
                "role":"system",
                "content": """ You are an agent whose prime function is to validate user input, 
                and call the tools that are provided to you which closely match the user's requirement 
                {prompt}
                """
            },
            {
                "role": "user",
                "content": self.query
            },
        ]
        
        @tool 
        def manim_tool(prompt):
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
            instance = Manim(response_format=response_format, prompt=prompt)
            response_code = instance.inferModel(model=ModelResponse)
            return response_code

        @tool 
        def contextual_text_tool(prompt):
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
        def flowchart_tool(prompt):
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
                instance = Flowchart(input=prompt)
                return {
                    "ok" :True,
                    "result":instance.create_flowchart()
                }
            except Exception as e:
                return {
                    "ok":False,
                    "error":str(e)
                }

            
            
        @wrap_model_call
        def dynamic_model_selection(self, request: ModelRequest, handler) -> ModelResponse:
            """Choose model based on conversation complexity. """
            message_count = len(request.state["messages"])

            if message_count > 10:
                # usse an advanced model for longer conversations, also to save on cost
                model = self.advanced_model
            else:
                model = self.basic_model

            return handler(request.override(model=model))
        
        tools = [manim_tool,flowchart_tool,contextual_text_tool]
        agent = create_agent(
            self.model, 
            tools = tools, 
            middleware =[dynamic_model_selection],
            system_prompt = """
                YOU ARE AN AGENT WHO HAS THE TOOLS TO CREATE CODE FOR A SPECIFIC USER BASED QUERY, 
                THE TOOLS INCLUDE:
                -- MANIM : GENERATES A MANIM ILLUSTRATION OF .mp4 UPON BEING CALLED WITH THE ILLUSTRATION QUERY FROM THE USER,
                    YOUR TASK IS TO ENSURE THAT THE OUTPUT CODE DOES NOT COME OUT AS AN ERROR.
                -- FLOWCHART : GENERATES A FLOWCHART FOR THE MANIM ILLUSTRATION, THE USER MUST BE SENT THIS FIRST, UPON BEING APPROVED BY THE USER, 
                    THE MANIM TOOL MUST BE CALLED WITH THE FEATURES ASKED BY THE USER BEING CONCATENATED INTO THE MANIM TOOL'S PROMPT. 
                    THIS IS USED TO PREPARE A PLAN FOR THE MANIM ILLUSTRATION.
                -- TEXT : THIS TOOL GENERATES THE TEXT FOR THE USER'S QUERY, THIS TOOL MUST BE CALLED WHEN THE USER HAS NOT SPECIFICALLY ASKED FOR AN ILLUSTRATION.
                THESE TOOLS RETURN CODE IN THE FORM OF TEXT, YOUR TASK IS TO SEND AND ERROR IF OCCURED, INCLUDING SUGGESTIONS TO ENHANCE THE CODE IF NEEDED (USE THIS JUDICIOUSLY)

            """
        )
        return agent.invoke({
            "messages":[{"role":"user","content":"create a "}]
        })


