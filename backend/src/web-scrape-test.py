import os
from parallel import Parallel

client = Parallel(api_key="M0anFvTQZbxD35oGIAKvvl5sPt1y0lzMZIykC0Qa")

search = client.beta.search(
    objective="what is linear regression, use wikipedia",
    mode="fast",
    excerpts={"max_chars_per_result": 500},
)

for result in search.results:
    # print(f"{result.title}: {result.url}")
    for excerpt in result.excerpts:
        print(excerpt[:200])
