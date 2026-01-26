this is what's left : 

1) RAG -> 
                        - [x] generation is left based on the input chunks
                        - [x] PDF package that also allows OCR for image based pdfs
                        - [x] Add logic to ensure this works out for texts too.


2) Manim pipeline ->
        - [x] FIx the prompt to be more elaborate for ncihe cases that dont need a video, the model needs to come up with smtn that's makeshift
        - [x] running the created code, using subprocess
        - [x] Add logic to make sure the video's length does not cross 25seconds.
                |--> Use the input/4 method, saving the last frames for summary and each lasts for 5s.
                |--> this must aslo include a mini explanation.

3) graphviz/image generation -> 
        for showing the flowchart for the manim illustration, will be called by the agent class.
        - [] Figure out how to compress the image to save on cloud deployment
        - [] find models that are capable of drawing graphs on huggingface
        - [] very descriptive prompt for the generation irrespective of the generation model

3.5) AGENT LAYER -> 
        -[x] complete the agentic layer 
        -[] should call the tools on its own based on the user promtp
        -[] manim: for manim, shd somehow run the subprocess command to ensure that it runs perfectly, 
                without errors and somehow be smart enf to fix the error.
        -[] text generation /RAG : this is fully text so the agent to cross verify that prompt vs response deviation is as low as possible.

4) server side->
                        [] database connection with Neon
                        [] AWS , Cloudinary or other service for the files and images.
                        [] elaborate resposne validation for the same.

5) misc ->              
                []  RAG understadning, fixing the RAG endpoint
                []  FFMPEG compression and conversion to a diff file type for manim. maybe gif from mp4 (default)



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