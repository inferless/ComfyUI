import json
from urllib import request, parse
import requests
import tqdm


class InferlessPythonModel:
    @staticmethod
    def download_file(url, file_name: str = None, folder_name: str = None):
        if file_name is None:
            file_name = url.split("/")[-1]

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024

        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(file_name, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")

    def initialize(self):
        InferlessPythonModel.download_file(
            "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt",
            "models/checkpoints",
        )

    def infer(self, inputs):
        workflow = inputs["workflow"]
        workflow_file_name = f"{workflow}.json"

        prompt = json.loads(open(f"worflows/{workflow_file_name}"))
        print("Prompt: ", prompt)
        p = {"prompt": prompt}

        data = json.dumps(p).encode("utf-8")

        req = request.Request("http://127.0.0.1:8188/prompt", data=data)
        resp = request.urlopen(req)
        resp_data = resp.read()
        print("Response: ", resp_data)
        return None

    def finalize(self, args):
        pass


if __name__ == "__main__":
    model = InferlessPythonModel()
    model.initialize()
    model.infer({"workflow": "txt_2_img.json"})
    model.finalize()
