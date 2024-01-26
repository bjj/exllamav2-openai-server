import openai
import asyncio

# Set your OpenAI API key here
api_key = "YOUR_API_KEY"

# Define a conversation prompt
conversation_prompt = (
    "Write an essay on the subject of history being written by the victors."
)

client = openai.AsyncOpenAI(base_url="http://localhost:8000/v1", api_key=api_key)


async def request():
    global client
    
    start_time = asyncio.get_event_loop().time()
    response = await client.chat.completions.create(
        model="bob",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": conversation_prompt},
        ],
        max_tokens=2000,
        n=1,
        stop=None,
    )
    duration = asyncio.get_event_loop().time() - start_time
    print(f"duration {duration} {response.usage.completion_tokens / duration:.2f} t/s")
    return response.choices[0].message.content

async def main():
    requests = []
    for i in range(20):
        requests.append(request())
    all = await asyncio.gather(*requests)
    print(all[int(len(all)/2)])
    
if __name__ == "__main__":
    asyncio.run(main())
