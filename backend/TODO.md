this is what's left : 

1) RAG -> 
        - [x] generation is left based on the input chunks
        - [x] PDF package that also allows OCR for image based pdfs
        - [x] Add logic to ensure this works out for texts too.


2) Manim pipeline ->
        - [] FIx the prompt to be more elaborate for ncihe cases that dont need a video.
        - [x] running the created code, using subprocess
        - [x] Add logic to make sure the video's length does not cross 25seconds.
                |--> Use the input/4 method, saving the last frames for summary and each lasts for 5s.
                |--> this must aslo include a mini explanation.

3) graphviz/image generation ->
        - [] Figure out how to compress the image to save on cloud deployment
        - [] find models that are capable of drawing graphs on huggingface
        - [] very descriptive prompt for the generation irrespective of the generation model

4) server side->
                        [] database connection with mongo
                        [] AWS , Cloudinary or other service for the files and images.
                        [] elaborate resposne validation for the same.


## frontend / deployment

1) deployment ->
                        [] figure out a cheap deployment service
                        [] image and manim videos storage too.
                        [] maybe buy a domain.
