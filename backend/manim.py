from huggingface_hub import InferenceClient
from google import genai
import subprocess
import json
import os
import sys
import boto3
from pydantic import BaseModel, ValidationError, constr
from dotenv import load_dotenv
load_dotenv()

class Manim:
    def __init__ (self,response_format, prompt):
        self.response_format = response_format
        self.prompt = prompt
    # now this can be passes as a response format to the model, to ensure that it returns without any backticks or extra bullshit
    
    def inferModel(self,model)->str:
        #genai.client
        client = InferenceClient(api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")) 
        input = self.prompt
        print(input)
        print("manim logic")

        prompt= f"""{input}
            YOU ARE A DEVELOPER WHOSE TASK IS TO GENERATE MANIM CODE THAT WILL FURTHER BE RUN USING `manim -pql` COMMAND, 
            IF THE CODE INCLUDES AN ERROR MESSAGE OR ANY SUGGESTIONS THEN AVOID THAT ERROR IN THE OUTPUT CODE, AND INCLUDE THE SUGGESTION IN OUTPUT CODE,
                MAKE SURE THE SUGGESTION AND THE ERROR ARE INCLUDED AS A PART OF THE OUTPUT CODE.
            THE SYSTEM MESSAGE FOR THE CODE GENERATION IS AS FOLLOWS YOU ARE TO STRICTLY ADHERE TO EVERY SINGLE RULE IN THIS
            You MUST return a valid JSON object with this EXACT structure:
                {{
                    "code": "escaped python code here with \\n for newlines and \\\" for quotes",
                    "className": "YourClassName"
                }}
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

        """
        messages=[
            {
                "role":"system",
                "content":
                """
                    You are a manim developer, you are tasked to strictly return only code for manim illustrations
                    in the form of the given json response format.
                    If the input prompt does include an error code, try to avoid the error in the output.
                """
            },
            {
                "role": "user",
                "content": prompt
            },
        ]
        response = client.chat_completion(
            model="zai-org/GLM-4.7",
            messages= messages,
            max_tokens=3000,
            temperature=0.9,
            response_format=self.response_format
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

            # if cleaned.startswith('"') and cleaned.endswith('"'): # just incase its embedded in quotes, use when doomsday
            #     cleaned = cleaned[1:-1]

            parsed = json.loads(cleaned)


        try:
            response_text = model(**parsed)
        except ValidationError as e:
            print(e.message)
            print(response_text)

        print(response_text.className)
        return response_text
