# Fetch metadata about models known to ollama

import sys
import httpx
import json
import copy
from urllib.parse import urljoin
import asyncio

async def get_url_of_type(url, mimetype):
    async with httpx.AsyncClient() as client:
        headers = {
            'Accept': mimetype
        }

        try:
            response = await client.get(url, headers=headers, follow_redirects=True)

            # Check if the request was successful (HTTP status code 200)
            if response.status_code == 200:
                return response.text  # You can also use response.content for binary data
            else:
                print(f"Request failed with status code {response.status_code}")
                return None

        except httpx.RequestError as e:
            print(f"Request error: {e}")
            return None


async def get_ollama_model_descriptor(repository, debug=False):
    baseUrl = 'https://registry.ollama.ai/'
    namespace = 'library'
    tag = 'latest'
    
    if '/' in repository:
        namespace, repository = repository.split('/')
    if ':' in repository:
        repository, tag = repository.split(':')

    url = urljoin(baseUrl, f"v2/{namespace}/{repository}/manifests/{tag}")

    data = await get_url_of_type(url, 'application/vnd.docker.distribution.manifest.v2+json')
    
    if data:
        manifest = json.loads(data)
    else:
        print("Failed to retrieve data from the URL.")

    layerNames = {
        "application/vnd.ollama.image.model": "model",
        "application/vnd.ollama.image.license": "license",
        "application/vnd.ollama.image.template": "template",
        "application/vnd.ollama.image.params": "params",
        "application/vnd.ollama.image.system": "system",
        "application/vnd.docker.container.image.v1+json": "config",
    }

    blobs = copy.deepcopy(manifest['layers'])
    blobs.append(copy.deepcopy(manifest['config']))

    pending = []    
    for blob in blobs:
        if blob['size'] > 2048:
            continue
        url = urljoin(baseUrl, f"v2/{namespace}/{repository}/blobs/{blob['digest']}")
        if debug:
            print(url)
        async def fetch(b, u):
            b['body'] = await get_url_of_type(u, b['mediaType'])
        pending.append(fetch(blob, url))
    await asyncio.gather(*pending)

    descr = {}
    for blob in blobs:
        try:
            body = blob['body']
        except KeyError:
            continue
        name = layerNames[blob['mediaType']]
        if name in ['params', 'config']:
            body = json.loads(body)
        descr[name] = body
    return descr
    
async def main():
    repository = sys.argv[1]

    descr = await get_ollama_model_descriptor(repository, debug=True)
    print(json.dumps(descr, indent=4))
    

if __name__ == "__main__":
    asyncio.run(main())
