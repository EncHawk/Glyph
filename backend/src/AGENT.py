import os
import sys
import datetime
import subprocess
from manim_agent import Manim
from flowchart import Flowchart
from flask import jsonify
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
            repo_id="meta-llama/Llama-3.3-70B-Instruct",
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
            aws_access_key_id=os.getenv("aws_access_key_id"),
            aws_secret_access_key=os.getenv("aws_secret_access_key"),
            region_name=os.getenv("aws_region"),
        )
        output_vid = os.path.abspath(os.path.join(self.GEN_DIR, f"{className}.py"))
        generated_video = os.path.join(
            self.MEDIA_DIR, "videos", className, "480p15", f"{className}.mp4"
        )
        try:
            subprocess.run(["manim", "-pql", output_vid, className], check=True)
        except Exception as e:
            print("Manim died:", e)
            raise Exception("Manim render failed")
        s3_key = f"manim/{className}_{datetime.datetime.now().isoformat()}.mp4"
        try:
            s3.Bucket(s3_bucket_name).upload_file(
                generated_video, s3_key
            )  # ffs aws was getting a python file in the format of mp4 im retarded
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
            aws_access_key_id=os.getenv("aws_access_key_id"),
            aws_secret_access_key=os.getenv("aws_secret_access_key"),
            region_name=os.getenv("aws_region"),
        )
        output_img = os.path.abspath(
            os.path.join(self.FLOWCHART_MEDIA_DIR, f"{className}.svg")
        )
        try:
            subprocess.run([sys.executable, f"{className}.py"], check=True)
        except Exception as e:
            print("Flowchart generation messed up:", e)
            raise Exception("Flowchart render failed")

        s3_key = f"flowchart/{className}_{datetime.datetime.now().isoformat()}.svg"
        try:
            extra_args = {"ContentType": "image/xml/+svg"}
            s3.Bucket(s3_bucket_name).upload_file(
                output_img, s3_key, ExtraArgs=extra_args
            )

        except botocore.exceptions.ClientError as e:
            print("S3 upload failed:", e)
            raise Exception("S3 upload failed")
        s3_url = f"https://{s3_bucket_name}.s3.amazonaws.com/{s3_key}"
        return s3_url

    def manim_tool(self, prompt, correction_context=None):
        """
        :param prompt: the input prompt to generate the illustration code
        :param correction_context: Optional dict with previous error and code for retry
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
            # If we have correction context, modify the prompt
            if correction_context:
                enhanced_prompt = f"""
                    {prompt}

                    IMPORTANT: Previous attempt failed with this error:
                    {correction_context["error"]}

                    Previous code that failed:
                    ```python
                    {correction_context["code"]}
                    ```

                    Fix the error and generate corrected code. Common Manim fixes:
                    - Don't use 'opacity' parameter in Dot() - use set_opacity() method instead
                    - Use fill_opacity and stroke_opacity for VMobjects
                    - Ensure all imports are correct
                    - Check method signatures match Manim v0.19.2 API
                """
                instance = Manim(
                    response_format=response_format, prompt=enhanced_prompt
                )
            else:
                instance = Manim(response_format=response_format, prompt=prompt)

            response_text = instance.inferModel(model=ModelResponse)

            # Ensure the directory exists
            os.makedirs(self.GEN_DIR, exist_ok=True)

            # Write the generated code to file
            file = os.path.join(self.GEN_DIR, f"{response_text.className}.py")
            with open(file, "w") as f:
                f.write(response_text.code)

            subprocess.run(["manim", "-pql", file, response_text.className], check=True)
            aws_string = self.manim_response(
                response_text.className, "glyph-data-storage"
            )
            return {
                "ok": True,
                "msg": "File Ran successfully",
                "string": aws_string,
                "code": response_text.code,  # ← Return code on success too
                "className": response_text.className,
            }
        except Exception as e:
            # Try to return the code even on failure
            error_code = None
            class_name = None
            try:
                error_code = response_text.code if "response_text" in locals() else None
                class_name = (
                    response_text.className if "response_text" in locals() else None
                )
            except:
                pass

            return {
                "ok": False,
                "error": str(e),
                "code": error_code,  # ← Return code that failed
                "className": class_name,
            }

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

    def flowchart_tool(self, prompt, correction_context=None):
        """
        :param prompt: the prompt to generate the code for graphviz flowchart
        :param correction_context: Optional dict with previous error and code for retry
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
            # Apply correction if available
            if correction_context:
                enhanced_prompt = f"""
                {prompt}

                Previous attempt failed with error:
                {correction_context["error"]}

                Previous code:
                ```python
                {correction_context["code"]}
                ```

                {correction_context.get("correction_advice", "Fix the error and regenerate.")}
            """
                res = Flowchart(input=enhanced_prompt)
            else:
                res = Flowchart(input=prompt)

            os.makedirs(self.GEN_FLOW, exist_ok=True)
            gen_file = os.path.join(self.GEN_FLOW, f"{res.className}.py")

            if len(res.code) > 10:
                with open(gen_file, "w") as f:
                    f.write(res.code)

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

            s3_string = self.flowchart_response(res.className, "glyph-data-storage")
            return {
                "ok": True,
                "result": "image successfully created",
                "string": s3_string,
                "code": res.code,  # ← Return code
                "className": res.className,
            }
        except Exception as e:
            error_code = None
            class_name = None
            try:
                error_code = res.code if "res" in locals() else None
                class_name = res.className if "res" in locals() else None
            except:
                pass

            return {
                "ok": False,
                "error": str(e),
                "code": error_code,  # ← Return failed code
                "className": class_name,
            }

    def run_with_retry(self, tool, prompt):
        """
        Retry logic with LLM-powered error correction
        """
        last_error = None
        last_code = None
        correction_context = None

        previous_errors = []
        for attempt in range(self.attempts):
            print(f"\n Attempt {attempt + 1}/{self.attempts}")

            # Call the tool with correction context if available
            if tool == self.manim_tool:
                result = tool(prompt, correction_context=correction_context)
            elif tool == self.flowchart_tool:
                result = tool(prompt, correction_context=correction_context)
            else:
                result = tool(prompt)

            # Success!
            if result.get("ok"):
                print(f" Success on attempt {attempt + 1}")
                return result

            # Failed - prepare for retry
            last_error = result.get("error", "Unknown error")
            if any(code in str(last_error) for code in ['504', '402', '429', '503']):
                print(f" API Error: {last_error}")
                return {
                    "ok": False,
                    "error": "HuggingFace API error - check credits, quota, or try again later :( gotta pay up again",
                    "attempt": attempt
                }
            last_code = result.get("code", "")

            print(f" Attempt {attempt + 1} failed: {last_error[:200]}")

            # If this is the last attempt, give up
            if attempt == self.attempts - 1 or result.get('ok') == True:
                break

            # Use LLM to analyze error and suggest corrections
            if last_code:
                correction_prompt = f"""
                    You are a debugging and quality assurance expert. Analyze this error and provide specific fixes.
                    If the errors arent existing, analyse the code and take a look at the input prompt to ensure that the code is doing what is asked,
                    if it isnt enhance it to be as effective and illustrative as possible. the idea is to make the video accurate to the prompt's demand.

                    ERROR:
                    {last_error}

                    FAILED CODE:
                    ```python
                    {last_code}
                    ```

                    Common issues and fixes:
                    - Manim Dot: Don't use opacity= in constructor. Use .set_opacity() method after creation
                    - Manim colors: Use from manim import RED, BLUE, WHITE, etc.
                    - Graphviz: Check node/edge syntax

                    Provide SPECIFIC instructions to fix this error (not new code, just instructions):
                """

                try:
                    if last_error in previous_errors: #check if its the same error as before, redo if yes.
                        correction_context = {
                            "error": last_error,
                            "code": last_code,
                            "correction_advice": "Previous fix didn't work. Try a completely different approach to fix this error, This must not occur again."
                        }
                        continue
                    previous_errors.append(last_error)
                    correction_response = self.advanced_model.invoke(correction_prompt)
                    correction_instructions = correction_response.content.strip()

                    print(
                        f" LLM Correction Advice:\n{correction_instructions[:300]}..."
                    )

                    # Build correction context for next attempt
                    correction_context = {
                        "error": last_error,
                        "code": last_code,
                        "correction_advice": correction_instructions,
                    }
                except Exception as llm_error:
                    print(f" LLM correction failed: {llm_error}")
                    # Fallback: just pass error and code
                    correction_context = {
                        "error": last_error,
                        "code": last_code,
                        "correction_advice": "Fix the error shown above",
                    }
            else:
                print(" No code available to analyze")

        return {
            "ok": False,
            "error": f"Failed after {self.attempts} attempts",
            "last_error": last_error,
            "last_code": last_code,
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

        # tools_map = {
        #     "manim_tool": ("manim", self.run_with_retry(self.manim_tool, query)),
        #     "flowchart_tool": (
        #         "flowchart",
        #         self.run_with_retry(self.flowchart_tool, query),
        #     ),
        # }

        if "manim_tool" in choice:
            result = self.run_with_retry(self.manim_tool, query)
            return jsonify(
                {
                    "tool": "manim",
                    "data": result,
                }
            ), 200

        elif "flowchart_tool" in choice:
            result = self.run_with_retry(self.flowchart_tool, query)
            return jsonify(
                {
                    "tool": "flowchart",
                    "data": result,
                }
            ), 200

        else:
            result = self.run_with_retry(self.flowchart_tool, query)
            return jsonify(
                {
                    "tool": "flowchart",
                    "data": result,
                }
            ), 200


# based on what the llm says, choose which one to run
