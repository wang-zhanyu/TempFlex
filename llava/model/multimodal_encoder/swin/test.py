from transformers import AutoImageProcessor
from modeling_swin import SwinModel
import torch
from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("/mnt/bn/tns-algo-ue-my/zhanyuwang/Code/tns_ue_mllm_runner/checkpoints/backbone/swin-base-patch4-window7-224-in22k")
model = SwinModel.from_pretrained("/mnt/bn/tns-algo-ue-my/zhanyuwang/Code/tns_ue_mllm_runner/checkpoints/backbone/swin-base-patch4-window7-224-in22k").to('cuda:0')

inputs = image_processor(image, return_tensors="pt")
inputs = inputs.to('cuda:0')

for i in tqdm(range(256)):
    with torch.no_grad():
        outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)