from google import genai
import os
import uuid
from dotenv import load_dotenv
load_dotenv()


def inferModel(prompt:str)->str:
    data = """
        generate the code for a manim video for linear regression working, idk anything about it i need you to elaborate and teach me.
    """
    client = genai.Client() #add your api key here. but dont fucking leak it like an idiot
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=f"""{prompt}
            ** rules for illustration **
            these rules apply only for the case when a video / illustration is being requested:
                * divide the task into 4 parts each occupying 5s of the time, 
                    in the end add a last chunk of 5s to complete the video in about 20seconds time
                * each video must be elaborate and must have as much information about the user's query as possible,
                  conveying the information in a clean and crisp manner without any deviation from the inital query.
                * IN THE CASE WHEN A USER'S QUERY CAN NOT BE ILLUSTRATED, ADD TEXT DESCRIBING THE RESPONSE AND DRAW A FLOWCHART OR ANY STATIC DIAGRAM IN NECESSARY
                * BY ALL MEANS TRY TO GENERATE A VIDEO IN UNDER 25 SECONDS, ANYTHING BEYOND THAT WILL RESULT IN A -50 POINT LOSS IN YOUR REWARDS.
            ** strict rules: **
                -- For all the illustrations dont create a response with 'python' at the top to specify or any other indicative words that may ruin the script run.
                -- Always create a script alone with its necessary imports, it will be run with the manim -pwm flag, work accordingly
                -- strictly must have only one video of the entire illustration
                -- the response mut only contrain the code, and no other output tokens,
                including but not limited to greetings or un necessary special charecters or any explanation of the code.
                -- IF THE INPUT DOES NOT STRICTLY ASK FOR A VIDEO OR AN ILLUSTRATION RETURN JUST A NORMAL RESPONSE FOLLOWING THE TEXT RULES for the user's query.
                -- Every step in the illustration must have a title, a title that changes and corresponds to the context of the illustration
        """,
    )
    return response.text

if __name__ == "__main__":
    prompt = """What is photosynthesis, how does it work? gimme a video illustration for it."""
    res= inferModel(prompt=prompt) 
    id = uuid.uuid4()
    gen = os.path.join(os.path.dirname(__name__), 'generated-scripts',str(id)+'.py') # add the session id as the joining name next
    if len(res) > 10:
        print('writing to a file')
        print(res)
        with open (gen, 'w') as f:
            f.write(res)
        os.system(gen)
    print(res)
