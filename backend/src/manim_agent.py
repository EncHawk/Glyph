import json
import os
import re

from huggingface_hub import InferenceClient
from pydantic import ValidationError
from dotenv import load_dotenv

load_dotenv()


class Manim:
    def __init__(self, response_format, prompt):
        self.response_format = response_format
        self.prompt = prompt

    # now this can be passes as a response format to the model, to ensure that it returns without any backticks or extra bullshit

    def _chat_completion(self, client, messages, max_tokens=5000, temperature=0.6):
        response = client.chat_completion(
            model="zai-org/GLM-4.7",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=self.response_format,
        )
        return response.choices[0].message.content

    @staticmethod
    def _parse_json_payload(raw: str) -> dict:
        if raw is None:
            raise ValueError("Model returned None response")

        cleaned_raw = raw.strip()
        cleaned_raw = re.sub(r"^```(?:json|python|py)?\s*", "", cleaned_raw)
        cleaned_raw = re.sub(r"\s*```$", "", cleaned_raw)

        if not cleaned_raw.startswith("{"):
            json_start = cleaned_raw.find("{")
            json_end = cleaned_raw.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                cleaned_raw = cleaned_raw[json_start : json_end + 1].strip()

        parsed = json.loads(cleaned_raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Parsed model response is not a JSON object: {parsed}")
        return parsed

    def inferModel(self, model) -> str:
        client = InferenceClient(api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
        user_input = self.prompt
        print(user_input)
        print("manim logic")

        prompt = (
            user_input
            + """
            YOU ARE A DEVELOPER WHOSE TASK IS TO GENERATE MANIM CODE THAT WILL FURTHER BE RUN USING `manim -pql` COMMAND, 
            IF THE CODE INCLUDES AN ERROR MESSAGE OR ANY SUGGESTIONS THEN AVOID THAT ERROR IN THE OUTPUT CODE, AND INCLUDE THE SUGGESTION IN OUTPUT CODE,
            MAKE SURE THE SUGGESTION AND THE ERROR ARE INCLUDED AS A PART OF THE OUTPUT CODE. YOU ARE TO ALWAYS RETURN MANIM CODE, NEVER RETURN ANY THING BEYOND THAT AND STRICTLY FOLLOW THE PROMPT.
            THE SYSTEM MESSAGE FOR THE CODE GENERATION IS AS FOLLOWS YOU ARE TO STRICTLY ADHERE TO EVERY SINGLE RULE IN THIS
            You MUST return a valid JSON object with this EXACT structure:
               GOOD EXAMPLE (merely an example do not use this for fallback.):
                {
                "className": "CircleAnimation",
                "code": "from manim import *\\n\\nclass CircleAnimation(Scene):\\n    def construct(self):\\n        circle = Circle()\\n        self.play(Create(circle))"
                }

                BAD EXAMPLES (DON'T DO THIS):
                {
                "className": "Example",
                "code": "I will use random logic here. The prompt says to import random so I will do that. from manim import *"
                }
                {
                    "classname" : "User prompt explanation",
                    "code" : "the user asked about topic 'x' i will explain x in detail and talk about its indepth architecture and so on.."
                }
            ** rules for illustration ** these rules apply only for the case when a video / illustration is being requested: 
                * each video must be elaborate and must have as much information about the user's query as possible, conveying the information in a clean and crisp manner without any deviation from the inital query. 
                * IN THE CASE WHEN A USER'S QUERY CAN NOT BE ILLUSTRATED, ADD TEXT DESCRIBING THE RESPONSE AND DRAW A FLOWCHART OR ANY STATIC DIAGRAM IN NECESSARY 
                * BY ALL MEANS TRY TO GENERATE A VIDEO IN UNDER 15 SECONDS, ANYTHING BEYOND THAT WILL RESULT IN A -50 POINT LOSS IN YOUR REWARDS.
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
            -- IF YOU ARE USING THE random FUNCTION IMPORT THE RANDOM LIBRARY!
            -- NEVER MENTION THE NAME OF THE RESPONSE FORMAT IN THE TOP, ONLY RETURN THE JSON FORMAT THAT IS EXPECTED FROM YOU.
        """
        )
        messages = [
            {
                "role": "system",
                "content": """
                    You are a manim developer, you are tasked to strictly return only code for manim illustrations
                    in the form of the given json response format.
                    If the input prompt does include an error code, try to avoid the error in the output.
                """,
            },
            {"role": "user", "content": prompt},
        ]
        raw = self._chat_completion(client, messages)

        for attempt in range(2):
            try:
                parsed = self._parse_json_payload(raw)
                response_text = model(**parsed)
                print(response_text.className)
                return response_text
            except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
                if attempt == 1:
                    raise ValueError(
                        f"Failed to parse JSON response from model. Raw response: {raw}"
                    ) from exc

                repair_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You repair malformed or truncated Manim JSON outputs. "
                            "Return only one valid JSON object with exactly two keys: "
                            "\"className\" and \"code\". "
                            "Complete any cut-off code so it is syntactically valid."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "The previous model output was almost correct but malformed or truncated. "
                            "Repair it into valid JSON only.\n\n"
                            f"Original request:\n{user_input}\n\n"
                            f"Broken output:\n{raw}"
                        ),
                    },
                ]
                raw = self._chat_completion(
                    client,
                    repair_messages,
                    max_tokens=5000,
                    temperature=0.2,
                )
