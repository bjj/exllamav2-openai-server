from huggingface_hub import HfApi
import asyncio, sys
import httpx
from tqdm import tqdm
from pathlib import Path


async def download_file(client, url, filepath: Path):
    async with client.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("Content-Length", 0))
        with filepath.open("wb") as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name) as progress_bar:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)
                    progress_bar.update(len(chunk))

async def download_hf_repo(repo_id, root_dir):
    root_path = Path(root_dir)
    repo_owner, repo_name = repo_id.split('/')
    path = root_path / repo_owner / repo_name
    path.mkdir(parents=True, exist_ok=True)
    
    api = HfApi()
    async with httpx.AsyncClient() as client:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        for file in files:
            url = f"https://huggingface.co/{repo_id}/resolve/main/{file}?download=true"
            await download_file(client, url, path / file)


async def main():
    await download_hf_repo(sys.argv[2], sys.argv[1])
    
if __name__ == "__main__":
    asyncio.run(main())