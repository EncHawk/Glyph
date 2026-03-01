this is what's left : 

1) RAG -> 
        - [x] generation is left based on the input chunks
        - [x] PDF package that also allows OCR for image based pdfs
        - [x] Add logic to ensure this works out for texts too.
        - Include Milvus or any other vector db


2) Manim pipeline ->
        - [] FFMPEG in order to reduce the size or create a new filtype before being uploaded
        - ehtos is to create a loop that enhances the ouput by splitting it into multiple scenes rather than crushing it all into one mp4.

3) graphviz/image generation -> on hold for now, focussing on building the agentic layer and better features like diffusion models integration instead.

3.5) AGENT LAYER -> 
        - Add langgraph tools, for all the necessary tools. 

5) Flowchart: gonna exist as a standalone ednpoint to create and return aws string for a given prompt. the class will get called from flask.

## frontend / deployment

1) deployment ->
                        [] figure out a cheap deployment service
                        [] image and manim videos storage too.
                        [] maybe buy a domain.


IDEATION: 

    - Moving from Learners first to, feature rich platform enabling the following : 

letting the users provide api keys, 

context forwarding -> users choose a specific message and chain that to the model's input allowing the model to work on a specific piece of text 

text based interface is the go-to, maybe add an infinite canvas to allow context forwarding similar to infinite gpt.

SDLC update: moving from 20th to south of 25th for the beta. let's see where this takes me.