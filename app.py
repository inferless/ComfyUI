import json
from urllib import request, parse


class InferlessPythonModel:
    def initialize(self):
        pass

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
