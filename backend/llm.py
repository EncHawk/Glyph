from google import genai
from dotenv import load_dotenv
load_dotenv()


def inferModel(prompt:str)->str:
    data = """
        Dilip Kumar R
        linkedin.com/in/dilip
        +91 9606269992 | dilipkumar2000.r@gmail.com
        github.com/EncHawk
        ( A I  a n a l y s i s ,  G i t h u b  R e p o s i t o r y  A n a l y s i s ,  T e l e g r a m  i n t e g r a t i o n )
        Projects
        Education
        Experience
        Technical Skills
        Open Source Contributions
        B-Tech COMPUTER SCIENCE AND ENGINEERING 
        CMR University, Bangalore.
        September 2023 -  August 2027  *expected
        Pre University
        St. Joseph’s Pre University College, Bangalore.
        July 2021 - March 2023
        CGPA (current) : 8.86
        Grade : 77%
        SCIENTIST/ENGINEER INTERN
        Indian Space Research Organisation
        October 2025 - November 2025
        Bangalore,India
        Revamped flight software telemetry verification system using Python, slashing verification time by 50% through
        the elimination of manual, hard-coded checks, and validating 50+ telemetry parameters.
        Developed a production-grade Python library to parse Ada-based flight software and extract telemetry
        parameters, enabling automated data validation and visualization workflows.
        Wrote a comprehensive test suite with >65% code coverage using pytest and unittest, implementing unit tests that
        reduced debugging time by 40%
        AsyncApi - Enhanced the template quality by replacing direct calls to the sever by helper functions in the template.
        Resulting in a performance enhancement of over 20%. pr:  #1632
        Kubeflow - Redesigned the navigation bar with modern UI, full mobile responsiveness using Tailwind CSS utility
        classes and media queries, significantly incrementing overall user experience. pr:  #4270
        Freddy Bot
        T e c h  S t a c k :  T y p e s c r i p t ,  P y t h o n ,  H u g g i n g F a c e ,  G i t h u b  A P I ,  T e l e g r a f ,  R e a c t .
        Constructed an open-source developer productivity tool that delivers real-time Telegram notifications to repository
        contributors when new issues are created, streamlining response times.
        Architected an intelligent automation pipeline leveraging HuggingFace and open source models like Mistral-7B and Llama-3B
        to generate contextual and rapid summaries, including the solution for the issue in the repository. This results in a significant
        increase in the productivity of the contributor.
        G i t h u b - L i n k
        ( l i v e - l i n k )
        ( R A G ,  A I  c l a u s e  a n a l y s i s ,  C h a t  i n t e r f a c e  ,  R i s k  D e t e c t i o n )
        Clause Buddy
        T e c h  S t a c k :  J a v a s c r i p t ,  P y t h o n ,  F a s t A P I ,  L a n g C h a i n ,  G e m i n i  A P I ,   C h r o m a D B ,  R e a c t .  
        Engineered a full-stack legal tech application using Python and React that analyzes contracts to identify high-risk clauses,
        protecting clients from unfavorable terms before signing a contract.
        Implemented Retrieval-Augmented Generation pipeline with Google Gemini 1.5 Flash and LangChain, leveraging ChromaDB
        vector embeddings for semantic document search and context-aware question answering with sub-second response times. 
        G i t h u b - L i n k
        ( l i v e - l i n k )
        Languages : C, C++, Ada, Python, JavaScript,Typescript.
        Technologies : Git, Github, Docker, AWS.  
        Frameworks : React js, Node js, Express js, MongoDB, LangChain, LangGraph, PyTorch.  
        |Bangalore, India
        | leetcode.com/Enchantedhawk|
        3rd  Runner up - MarchMania hackathon 
        Winner - DevStorm Hackathon
        1400+  rating  in Leetcode DeepLearningAI : Langgraph: Building AI agents with
        OpenAI 
        Coursera : Machine Learning Specialisation - Andrew Ng
        1000+  rating  in Codeforces
        Achievements & Certifications
        | x.com/Dilip
    """
    client = genai.Client() #add your api key here. but dont fucking leak it like an idiot
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="""{data}
            gimme a 200 worded resume review behaving as an ATS agent, your task is to hire a ts dev for sde inter. return today's date too
            ** strict rules: **
                -- strictly must have only one video of the entire illustration
                -- the response mut only contrain the code, and no other output tokens,
                including but not limited to greetings or un necessary special charecters or any explanation of the code.
                -- IF THE INPUT DOES NOT STRICTLY ASK FOR A VIDEO OR AN ILLUSTRATION RETURN JUST A NORMAL RESPONSE FOLLOWING THE TEXT RULES for the user's query.
                -- Every step in the illustration must have a title, a title that changes and corresponds to the context of the illustration
        """,
    )
    return response.text

if __name__ == "__main__":
    res= inferModel(prompt="")
    print(res)
