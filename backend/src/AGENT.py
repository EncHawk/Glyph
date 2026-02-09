import os
import sys
import datetime
import subprocess
from manim_agent import Manim
from flowchart import Flowchart
import jsonify
from rag import RagAgent
import boto3
import botocore
from huggingface_hub import InferenceClient
from pydantic import BaseModel, ValidationError, constr
from langchain.tools import tool
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
)
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelResponse, ModelRequest
# this is essentially only gonna return the output of any of the tools RAG , manim or flowchart


class Agent:
    def __init__(self, session_id, attempts, use_LLM=True):
        self.session_id = session_id
        self.attempts = attempts
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        GEN_DIR = os.path.join(ROOT_DIR, "generated-scripts")
        GEN_FLOW = os.path.join(ROOT_DIR, "generated-flowcharts")
        MEDIA_DIR = os.path.join(ROOT_DIR, "media")
        FLOWCHART_MEDIA_DIR = os.path.join(ROOT_DIR, "flowchart_media")

        self.hf_llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            max_new_tokens=512,
            temperature=0.7,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        )
        self.basic_llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            max_new_tokens=512,
            temperature=0.7,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        )
        self.advanced_model = ChatHuggingFace(llm=self.hf_llm)
        self.basic_model = ChatHuggingFace(llm=self.basic_llm)
        self.model = self.basic_model
        # the paths for the directories
        self.ROOT_DIR = ROOT_DIR
        self.GEN_DIR = GEN_DIR
        self.GEN_FLOW = GEN_FLOW
        self.MEDIA_DIR = MEDIA_DIR
        self.FLOWCHART_MEDIA_DIR = FLOWCHART_MEDIA_DIR

    class ModelResponse(BaseModel):
        code: constr(min_length=1, strip_whitespace=True)
        className: str

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "Model_Response",
            "schema": ModelResponse.model_json_schema(),
            "strict": True,
        },
    }

    def manim_response(self, className: str, s3_bucket_name: str):
        os.makedirs(self.GEN_DIR, exist_ok=True)
        os.makedirs(self.MEDIA_DIR, exist_ok=True)
        assert os.path.join(self.MEDIA_DIR, "videos", "480p15", f"{className}.mp4")
        s3 = boto3.resource(
            "s3",
            aws_access_key=os.getenv("aws_access_key_id"),
            aws_secret_access_key=os.getenv("aws_secret_access_key"),
            region_name=os.getenv("aws_region"),
        )
        output_vid = os.path.abspath(self.GEN_DIR, f"{className}.py")
        try:
            subprocess.run(["manim", "-pql", output_vid, className], check=True)
        except Exception as e:
            print("Manim died:", e)
            raise Exception("Manim render failed")
        s3_key = f"manim/{className}_{datetime.datetime.now().isoformat()}.mp4"
        try:
            s3.Bucket(s3_bucket_name).upload_video(output_vid, s3_key)
        except botocore.exceptions.ClientError as e:
            print("S3 upload failed:", e)
            raise Exception("S3 upload failed")
        s3_url = f"https://{s3_bucket_name}.s3.amazonaws.com/{s3_key}"
        return s3_url

    def flowchart_response(self, className: str, s3_bucket_name: str):
        """
        Docstring for flowchart_response

        :param className: THE CLASSNAME IS THE NAME OF THE DIGRAPH CLASS, WHICH IS GONNA BE THE NAME OF THE FILE AS WELL.
        :type className: str
        :param s3_bucket_name: Description
        :type s3_bucket_name: str
        :return: Description
        :rtype: type[that]
        """
        os.makedirs(self.GEN_FLOW, exist_ok=True)
        os.makedirs(self.FLOWCHART_MEDIA_DIR, exist_ok=True)
        assert os.path.join(self.FLOWCHART_MEDIA_DIR, f"{className}.svg")

        s3 = boto3.resource(
            "s3",
            aws_access_key=os.getenv("aws_access_key_id"),
            aws_secret_access_key=os.getenv("aws_secret_access_key"),
            region_name=os.getenv("aws_region"),
        )
        output_img = os.path.abspath(self.FLOWCHART_MEDIA_DIR, f"{className}.svg")
        try:
            subprocess.run([sys.executable, f"{className}.py"], check=True)
        except Exception as e:
            print("Flowchart generation messed up:", e)
            raise Exception("Flowchart render failed")

        s3_key = f"flowchart/{className}_{datetime.datetime.now().isoformat()}.svg"
        try:
            extra_args = {"Content-Type": "image/xml/+svg"}
            s3.Bucket(s3_bucket_name).upload_file(
                output_img, s3_key, ExtraArgs=extra_args
            )

        except botocore.exceptions.ClientError as e:
            print("S3 upload failed:", e)
            raise Exception("S3 upload failed")
        s3_url = f"https://{s3_bucket_name}.s3.amazonaws.com/{s3_key}"
        return s3_url

    def manim_tool(self, prompt):
        """
        Docstring for Manim

        :param prompt: the input prompt to generate the illustration code
        """

        class ModelResponse(BaseModel):
            code: constr(min_length=1, strip_whitespace=True)
            className: str

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "Model_Response",
                "schema": ModelResponse.model_json_schema(),
                "strict": True,
            },
        }
        try:
            instance = Manim(response_format=response_format, prompt=prompt)
            response_text = instance.inferModel(model=ModelResponse)

            # Ensure the directory exists
            os.makedirs(self.GEN_DIR, exist_ok=True)

            # Write the generated code to file
            file = os.path.abspath(f"generated-scripts/{response_text.className}.py")
            with open(file, "w") as f:
                f.write(response_text.code)

            subprocess.run(["manim", "-pql", file, response_text.className], check=True)
            aws_string = self.manim_response(
                response_text.className, "glyph-data-storage"
            )
            return {
                "ok": True,
                "msg": "File Ran successfully, return mp4 string only for manim",
                "string": aws_string,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def contextual_text_tool(self, prompt):
        """
        Docstring for flowchart

        :param prompt: the prompt to generate the code for graphviz flowchart
        """

        class ModelResponse(BaseModel):
            code: constr(min_length=1, strip_whitespace=True)
            className: str

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "Model_Response",
                "schema": ModelResponse.model_json_schema(),
                "strict": True,
            },
        }
        llm = HuggingFaceEndpoint(  # add your huggingface token, this shit free and good heck yeah!
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            temperature=0.7,
        )
        model = ChatHuggingFace(llm=llm)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        instance = RagAgent(embeddings=embeddings, model=model)
        response_code = instance.inferModel(input=prompt)
        return response_code

    def flowchart_tool(self, prompt):
        """
        Docstring for flowchart

        :param prompt: the prompt to generate the code for graphviz flowchart
        """

        class ModelResponse(BaseModel):
            code: constr(min_length=1, strip_whitespace=True)
            className: str

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "Model_Response",
                "schema": ModelResponse.model_json_schema(),
                "strict": True,
            },
        }
        try:
            res = Flowchart(input=prompt)

            # Use predefined directory constants
            os.makedirs(self.GEN_FLOW, exist_ok=True)

            # Write the generated code to file
            gen_file = os.path.join(self.GEN_FLOW, f"{res.className}.py")
            if len(res.code) > 10:
                print("writing flowchart to file:", gen_file)
                with open(gen_file, "w") as f:
                    f.write(res.code)

            # Change to the generated-flowcharts directory and run the script
            # This ensures the SVG is created in the correct location
            original_cwd = os.getcwd()
            try:
                os.chdir(self.GEN_FLOW)
                subprocess.run(
                    [sys.executable, os.path.basename(gen_file)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            finally:
                os.chdir(original_cwd)

            s3_string = self.flowchart_response(
                f"{res.className}", "glyph-data-storage"
            )
            return {
                "ok": True,
                "result": "image successfully created. take this!",
                "string": s3_string,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def run_with_retry(self, tool, prompt):
        last_error = None
        context = prompt

        for attempt in range(self.attempts):
            result = tool(context)

            if result.get("ok"):
                return result

            last_error = result.get("error")

            context = f"""
            Original request: {prompt}

            Previous attempt {attempt + 1} failed with error:
            {last_error}

            Instructions to fix:
            - Review the error message carefully
            - Ensure the code is syntactically correct
            - For Manim: ensure all imports are correct and the Scene class is properly defined
            - For Flowchart: ensure Graphviz syntax is valid
            - Generate corrected code

            Generate the corrected code now:
            """

        return {
            "ok": False,
            "error": f"Failed after {self.attempts} attempts",
            "last_error": str(last_error)
            if last_error is not None
            else "Unknown error",
        }

    def run_agent(self, query):
        """
        Docstring for Agent

        :param self: Description
        :param query: user's input to be processed and classified to perform any of the provided tools
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
            - THE OUTPUT MUST BE FROM WITHIN WHAT EVER THE OPTIONS ARE, NOTHING ELSE MUST BE IN THE OUTPUT. 

            User request:
            {query}
        """
        response = self.model.invoke(prompt)
        choice = response.content.strip().lower()

        tools_map = {
            "manim_tool": ("manim", self.run_with_retry(self.manim_tool, query)),
            "flowchart_tool": (
                "flowchart",
                self.run_with_retry(self.flowchart_tool, query),
            ),
        }

        if "manim_tool" in choice:
            return jsonify(
                {
                    "tool": "manim",
                    "data": f"{self.run_with_retry(self.manim_tool, query)}",
                }
            ), 200

        elif "flowchart_tool" in choice:
            return jsonify(
                {
                    "tool": "flowchart",
                    "data": f"{self.run_with_retry(self.flowchart_tool, query)}",
                }
            ), 200

        else:
            return jsonify(
                {
                    "tool": "flowchart",
                    "data": f"{self.run_with_retry(self.flowchart_tool, query)}",
                }
            ), 200


# based on what the llm says, choose which one to run
