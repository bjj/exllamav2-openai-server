import openai
import asyncio
import random

# Set your OpenAI API key here
api_key = "YOUR_API_KEY"

# Define a conversation prompt
conversation_prompt = (
    "Write an essay on the subject of history being written by the victors."
)

client = openai.AsyncOpenAI(base_url="http://localhost:8000/v1", api_key=api_key)

async def request():
    global client, models
    
    start_time = asyncio.get_event_loop().time()
    try:
        response = await client.chat.completions.create(
            model=random.choice(models).id,
            #model=models[0].id, # we always return active model first
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": conversation_prompt},
            ],
            max_tokens=2000,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        tokens = response.usage.completion_tokens
    except Exception as e:
        content = repr(e)
        tokens = 0
    duration = asyncio.get_event_loop().time() - start_time
    print(f"duration {duration} {tokens / duration:.2f} t/s")
    return content

async def main():
    global models
    
    # wow this is an annoying interface
    models = []
    async for model_page in client.models.list():
        models.append(model_page)
    
    requests = []
    for i in range(20):
        requests.append(request())
    all = await asyncio.gather(*requests)
    print(all[int(len(all)/2)])
    
if __name__ == "__main__":
    asyncio.run(main())
