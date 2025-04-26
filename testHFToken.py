from transformers import AutoModel

access_token = "hf_your_api_key_here"
model = AutoModel.from_pretrained("private/model", token=access_token