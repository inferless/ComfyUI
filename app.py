import json
from urllib import request, parse
import requests
from tqdm import tqdm
import os
import asyncio


class InferlessPythonModel:
    @staticmethod
    def download_file(url, file_name: str = None, folder_name: str = None):
        if file_name is None:
            file_name = url.split("/")[-1]

        full_path = os.path.join(folder_name, file_name)

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024

        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(full_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")

    def initialize(self):
        import subprocess
        self.process = subprocess.Popen(["python3.10", "main.py"])
        InferlessPythonModel.download_file(
            "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt",
            folder_name="models/checkpoints",
        )

    def infer(self, inputs):
        workflow = inputs["workflow"]
        workflow_file_name = f"{workflow}"

        prompt = json.loads(open(f"workflows/{workflow_file_name}").read())
        p = {"prompt": prompt}

        data = json.dumps(p).encode("utf-8")

        req = request.Request("http://127.0.0.1:8188/prompt", data=data)
        resp = request.urlopen(req)
        resp_data = resp.read()
        print("Response: ", resp_data)
        return None

    def finalize(self):
        self.process.terminate()


if __name__ == "__main__":
    model = InferlessPythonModel()
    model.initialize()
    model.infer({"workflow": "txt_2_img.json"})
    model.finalize()
