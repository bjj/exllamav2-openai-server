# Fetch metadata about models known to ollama

import sys
import httpx
import json
import copy
from urllib.parse import urljoin
import pprint

async def get_url_of_type(url, mimetype):
    print(url)
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

# Example usage
import asyncio

url = 'https://registry.ollama.ai/v2/library/mistral/manifests/latest'

async def main():
    pp = pprint.PrettyPrinter(indent=4)

    baseUrl = 'https://registry.ollama.ai/'
    namespace = 'library'
    repository = sys.argv[1]
    tag = 'latest'

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

    pp.pprint(manifest)

    blobs = copy.deepcopy(manifest['layers'])
    blobs.append(copy.deepcopy(manifest['config']))

    pending = []    
    for blob in blobs:
        if blob['size'] > 2048:
            continue
        url = urljoin(baseUrl, f"v2/{namespace}/{repository}/blobs/{blob['digest']}")
        async def fetch(b, u):
            b['body'] = await get_url_of_type(u, b['mediaType'])
            print(f"gotbody {b['mediaType']}")
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

    pp.pprint(descr)
    #blobPath = 'https://registry.ollama.ai/v2/library/mistral/blobs/sha256:ed11eda7790d05b49395598a42b155812b17e263214292f7b87d15e14003d337'
if __name__ == "__main__":
    asyncio.run(main())
