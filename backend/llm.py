from google import genai



def inferModel(prompt:str)->str:
    
    client = genai.Client(api_key="AIzaSyCNA9DiMyeg7cERXKpElv_gQodKj9UAYr4")
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="""{prompt}
            
            ** strict rules: **
                -- strictly must have only one video of the entire illustration
                -- the response mut only contrain the code, and no other output tokens,
                including but not limited to greetings or un necessary special charecters or any explanation of the code.
                -- IF THE INPUT DOES NOT STRICTLY ASK FOR A VIDEO OR AN ILLUSTRATION RETURN JUST A NORMAL RESPONSE FOLLOWING THE TEXT RULES for the user's query.
                -- Every step in the illustration must have a title, a title that changes and corresponds to the context of the illustration
        """,
    )
    return response.text