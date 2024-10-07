## Preparation


### Requirements
We build `T-T+X` and `MosIT` via `ChatGPT`, you need first install the following package:
```
pip install openai
```

### Usage

We provide the sampled demos in [IT-demos.txt](data/IT_data/T-T+X_data/construction/IT-demos.txt) utilized to construct `T-T+X` dataset.
Then you need to set the parameter in [chatgpt.py](/home/haofei/mllm/NExT-GPT/data/IT_data/T-T+X_data/construction/chatgpt.py):
```
openai.api_key = ''
sample_number = 5000
modality = 'image'
```
Finally, run the code:
```
python chatgpt.py
python processing_log.py
```








