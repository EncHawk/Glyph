import os
import subprocess
import json
import sys
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pydantic import BaseModel, constr, ValidationError
load_dotenv()


class Flowchart():
    def __init__(self,input):
        self.input = input

    def create_flowchart(self):
        class flowchart_class(BaseModel):
            code:constr(min_length=1,strip_whitespace = True)
            className:str
        response_format = {
            "type" : "json_schema",
            "json_schema":{
                "name":"Model_Resposne",
                "schema" : flowchart_class.model_json_schema(),
                "strict":True
            }
        }

        client = InferenceClient(api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")) 
        prompt = """
            IF THE CODE INCLUDES AN ERROR MESSAGE OR ANY SUGGESTIONS THEN AVOID THAT ERROR IN THE OUTPUT CODE, AND INCLUDE THE SUGGESTION IN OUTPUT CODE,
                MAKE SURE THE SUGGESTION AND THE ERROR ARE INCLUDED AS A PART OF THE OUTPUT CODE.

            ** rules for illustration ** these rules apply only for the case when a video / illustration is being requested: 
                    * divide the task into 4 parts each occupying 5s of the time, in the end add a last chunk of 5s to complete the video in about 20seconds time 
                    * each video must be elaborate and must have as much information about the user's query as possible, conveying the information in a clean and crisp manner without any deviation from the inital query. 
                    * IN THE CASE WHEN A USER'S QUERY CAN NOT BE ILLUSTRATED, ADD TEXT DESCRIBING THE RESPONSE AND DRAW A FLOWCHART OR ANY STATIC DIAGRAM IN NECESSARY 
                    * BY ALL MEANS TRY TO GENERATE A VIDEO IN UNDER 25 SECONDS, ANYTHING BEYOND THAT WILL RESULT IN A -50 POINT LOSS IN YOUR REWARDS.
                ** strict rules: ** 
                -- YOU ARE STRICTLY FORBIDDEN TO WRITE RISKY CODE THAT MAY RESULT IN ERRORS, ALWAYS MAKE SURE THAT THE CODE WILL HAVE THE LEAST POSSIBILITY OF THROWING AN ERROR
                -- MAKE SURE THAT ALL THE CODE YOU WRITE EXISTS IN ONE FILE, ALL THE HELPER FUNCTIONS IF USED OR NEEDED MUST BE INCLUDED IN THE SAME SCRIPT.
                -- IRRESPECTIVE OF SYSTEM/BUILT-IN LIBRARIES INCLUDE THE NECESSARY IMPORTS
                -- THE SCRIPT WILL BE RUN IN AN EVIRONMENT WITH THE LATEST VERSION OF MANIM AND YOU MUST CREATE A SCRIPT THAT IS VALID ACCORDINGLY.
                -- WHILE USING LISTS OR ANY DATA STRUCTURES, ENSURE THAT THE USAGE IS VALID IN TERMS OF INDICES, WRAP IT IN A IF ELSE STATEMENT TO ENSURE SMOOTH RENDERING.
                -- YOU HAVE TO PROVIDE THE CODE ALWAYS FOR THE ILLUSTRATIONS IRRESPECTIVE OF THE COMPLEXITY.
                -- ANY CONTENT THAT IS GENERATED MUST BE CONCENTRATED IN THE CENTER OF THE CANVAS, 
                    IF THERE ARE ANY SORT OF VISUAL BOTTLENECKS GRADUALLY REMOVE ELEMENTS TO REPLACE THEM WITH NEW CONTENT.
                -- ALWAYS MAKE SURE THAT THE TEXT THAT EXPLAINS THE CONTENT IS 100% VISIBLE AT ALL TIMES AND CHANGES DYNAMICALLY.
                --  ENSURE THAT THE SYNTAX IS TAKEN CARE OF, THERE MUST STRICTLY BE NO SYNTAX ERRORS. EXPECIALLY WITH STRINGS QUOTATIONS CLOSING.
                -- Every step in the illustration must have a title, a title that changes and corresponds to the context of the illustration, the title must be simple to read and shouldnt exceed 10 charecters. 
                -- DO NOT EMBED THE CODE IN DOCSTRING QUOTES AT THE BEGINING OR AT THE END, ONLY IMPORTS, CLASS FOR THE MANIM ILLUSTRATION IS EXPECTED IN THE RESPONSE.
                -- For all the illustrations dont create a response with 'python' at the top to specify or any other indicative words that may ruin the script run.
                -- Always create a script alone with its necessary imports, it will be run with the manim -pwm flag, work accordingly to not add any extra charecters that will result in an error.
                -- strictly must have only one video of the entire illustration, the code must follow the above rules and not be followed by a summary or any thing apart from just the code.
                -- the response mut only contrain the code, and no other output tokens, never wrap the code in any docstring or language name that directly results in an error when manim command is run
                including but not limited to greetings or un necessary special charecters or any explanation of the code.
                -- Always ensure that the illustration covers all aspects of the prompt, make it as simple as possible to understand
                and be generous with adding visual cues that enhance the experience.
                -- BEFORE THE CODE IS RETURNED AS A RESPONSE ENSURE THAT THE CODE IS VALID WITHOUT ANY ERRORS OF ANY KIND, AND MAKE THE CHANGES ACCORDINGLY,
                    THE CODE MUST BE VERY SIMPLE YET PROVIDING A VISUAL EXPERIENCE THAT IS WONDERFUL.
                -- BE CAREFUL WHEN YOU USE THE COLOR BROWN IN THE CODE, REPLACE IT WITH  A SHADE OF RED.
                -- IF YOU ARE USING THE random FUNCTION IMPORT THE RANDOM LIBRARY!
                -- WHILE RENDERING MAKE SURE TO ADD THIS IN THE CODE TO ENSURE IT GETS RENDERED IN A SECURE DIRECTORY, THIS IS BEING ASSERTED, MUST ALWAYS BE USED TO GENERATE TEH CODE:
                    dot.render("media/flowchart", format="svg")
                THE CODE MUST INCLUDE THIS ABOVE SNIPPET TO ENSURE THAT THIS RENDERS IN A SPECIFIC DIRECTORY.
        """
        messages=[
            {
                "role":"system",
                "content":"""
                    You are a  assistant illustration developer who creates flowcharts in graphviz that will act as a skeleton for a manim illustration, 
                    your task is to generate the code for a graphviz image that will be a flowchart wireframe for how the manim illustration will be created.
                    the rules for manim creation are as follows, bear in mind its for manim illustration, use it as a reference to create the flowchart.
                    You MUST return a valid JSON object with this EXACT structure:
                        {{
                            "code": "raw python code with all imports covering all its uses in the rest of the script",
                            "className": "name of the digraph class"
                        }}
                    ** strict rules ** :
                    -- according to these illustration rules generate the code for a graphviz image with only digraph that follows a left to right path.
                    -- THE OUTPUT IMAGE SHOULD ALSO FEATURE HOW LONG THE CONTENT WILL BE ON THE SCREEN BASED ON THE FLOWCART PROMPT ABOVE.
                    -- THE CODE THAT IS GENERATED MUST USE THE SYNTAX FOR CALLING "dot" AS THE DIGRAPH HEAD, MUST RENDER AN SVG FORMAT OUTPUT.
                    -- THE ILLUSTRATION RULES ARE FOR REFERENCE ONLY, YOU ARE TO FOLLOW THE FLOWCHART RULES STRICTLY.
                    -- YOU ARE STRICTLY FORBIDDEN TO WRITE RISKY CODE THAT MAY RESULT IN ERRORS, ALWAYS MAKE SURE THAT THE CODE WILL HAVE THE LEAST POSSIBILITY OF THROWING AN ERROR
                    -- MAKE SURE THAT ALL THE CODE YOU WRITE EXISTS IN ONE FILE, ALL THE HELPER FUNCTIONS IF USED OR NEEDED MUST BE INCLUDED IN THE SAME SCRIPT.
                    -- IRRESPECTIVE OF SYSTEM/BUILT-IN LIBRARIES INCLUDE THE NECESSARY IMPORTS
                    -- THE SCRIPT WILL BE RUN IN AN EVIRONMENT WITH THE LATEST VERSION OF GRAPHVIZ, DIGRAPH AND YOU MUST CREATE A SCRIPT THAT IS VALID ACCORDINGLY.
                    -- WHILE USING LISTS OR ANY DATA STRUCTURES, ENSURE THAT THE USAGE IS VALID IN TERMS OF INDICES, WRAP IT IN A IF ELSE STATEMENT TO ENSURE SMOOTH RENDERING.
                    -- YOU HAVE TO PROVIDE THE CODE ALWAYS FOR THE ILLUSTRATIONS IRRESPECTIVE OF THE COMPLEXITY.
                    -- ANY CONTENT THAT IS GENERATED MUST BE CONCENTRATED IN THE CENTER OF THE CANVAS, 
                        IF THERE ARE ANY SORT OF VISUAL BOTTLENECKS GRADUALLY REMOVE ELEMENTS TO REPLACE THEM WITH NEW CONTENT.
                    -- ALWAYS MAKE SURE THAT THE TEXT THAT EXPLAINS THE CONTENT IS 100% VISIBLE AT ALL TIMES AND CHANGES DYNAMICALLY.
                    --  ENSURE THAT THE SYNTAX IS TAKEN CARE OF, THERE MUST STRICTLY BE NO SYNTAX ERRORS. EXPECIALLY WITH STRINGS QUOTATIONS CLOSING.
                    -- DO NOT EMBED THE CODE IN DOCSTRING QUOTES AT THE BEGINING OR AT THE END, ONLY IMPORTS, CLASS FOR THE DIGRAPH FLOWCHART IS EXPECTED IN THE RESPONSE 
                        WITH IMPORTS AND RAW CODE, NOTHING ELSE.
                    -- YOU NEED TO LABEL EACH AND EVERY GRAPH NODE WITH THE NUMBER OF SECONDS IT WILL OCCUPY AND WHAT SCENES IT WILL PORTRAY BASED ON THE INPUT GIVEN TO YOU.
                    -- DO NOT EMBED THE CODE IN ANY FORM OF QUOTATIONS WHICH WILL MAKE THE CODE FAIL ON EXECUTION, RETURN ONLY THE RAW CODE AND THE NAME OF THE DIGRAPH CLASS.
                    -- ALL ILLUSTRATIONS ONLY LAST MAXIMUM OF UPTO 25 SECONDS.
                    -- MAKE SURE THAT THE SIZES OF ALL THE DIAGRAMS ARE EQUAL AND COME UNDER THE SIZE 5.3 FOR EACH OF THE NODES. THE FINAL RENDER SHOULD BE UNDER THE 720P MARK,
                        ALLOWING IT TO FIT INSIDE A MOBILE SCREEN.
                """
            },
            {
                "role": "user",
                "content": input
            },
        ]
        response = client.chat_completion(
            model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            messages= messages,
            max_tokens=3000,
            temperature=0.9,
            response_format =response_format
        )
        raw = response.choices[0].message.content
        if not raw.startswith("{"):
            raise RuntimeError("Model did not return JSON:\n" + raw)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            cleaned = raw.strip()

            for fence in ["```", "```python", "```py"]:
                cleaned = cleaned.replace(fence, "")
            print(cleaned)
            parsed = json.loads(cleaned)
        try:
            response_text = flowchart_class(**parsed)
        except ValidationError as e:
            print(e.message)
            print(response_text)

        print(response_text.className)
        return response_text


#  ================ USE THIS ONLY TO TEST =====================
# if __name__ == "__main__":
    # input = "generate a flowchart to illustrate binary search."
    # res = Flowchart(input)
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # gen_dir = os.path.join(base_dir, "generated-flowcharts")
    # os.makedirs(gen_dir, exist_ok=True)
    # gen = os.path.join(gen_dir, f"{res.className}.py")

    # if len(res.code) > 10:
    #     print('writing to a file')
    #     with open (gen, 'w') as f:
    #         f.write(res.code)

    # file = os.path.abspath(f"generated-flowcharts/{res.className}.py")
#     subprocess.run(
#         [sys.executable, file],
#         check=True
#     )