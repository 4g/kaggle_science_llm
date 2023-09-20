import torch
import numpy as np

LLM_GPU = 0

class GPTQLLM:
    def __init__(self, model_path, tokens=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                        revision="gptq-4bit-32g-actorder_True",
                                                        low_cpu_mem_usage=True,
                                                        torch_dtype=torch.float16,
                                                        device_map=LLM_GPU)

        self.model.to_bettertransformer()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.token_map = self.get_token_map(tokens)

        print(self.token_map)
        print("Model Up")

    def get_token_map(self,  tokens):
        if tokens is None:
            tokens = ['A', 'B', 'C', 'D', 'E']
        sent = "," + ",".join(tokens) + ","
        ids = self.tokenizer.encode(sent)
        indices = [i for i in tokens]

        for id in ids:
            token = self.tokenizer.decode(id)
            if token in tokens:
                indices[tokens.index(token)] = id

        return indices, tokens

    def get_logits(self, question):
        input_tensor = self.tokenizer.encode(question, return_tensors="pt")
        input_tensor = input_tensor.cuda()
        # print(input_tensor.shape)
        output = self.model.forward(input_tensor, use_cache=False)
        logits = output.logits[0, -1]
        logits = logits.detach().cpu().numpy()

        return logits

    def process(self, question):
        scores = self.get_logits(question)
        indices = self.token_map[0]
        tokens = self.token_map[1]
        # top_indices = np.argsort(scores)[::-1][:5]
        # top_tokens = self.tokenizer.decode(top_indices)
        # print(top_tokens, top_indices, scores[top_indices])
        probs = scores[indices]
        # probs = softmax(probs)
        choices = sorted(list(zip(list(probs), tokens)), key=lambda x: x[0], reverse=True)
        return choices

class GGMLLLM:
    def __init__(self, model_path, tokens):
        from llama_cpp import Llama
        self.model = Llama(model_path=model_path,
                           n_ctx=1280,
                           n_threads=5,
                           n_gpu_layers=20,
                           verbose=True,
                           n_gqa=None,
                           use_mmap=False,
                           use_mlock=True,
                           n_batch=512)
        print("Model Up")

        self.token_map = self.get_token_map(tokens=tokens)
        print(self.token_map)



    def tokenize(self, text):
        return self.model.tokenize(text)

    def detokenize(self, tokens):
        return self.model.detokenize(tokens)

    def get_token_map(self,  tokens):
        sent = "," + ",".join(tokens) + ","
        sent = b"" + sent.encode('utf-8')
        ids = self.tokenize(sent)

        indices = [i for i in tokens]

        for id in ids:
            token = self.detokenize([id])
            token = token.decode('utf-8')

            if token in tokens:
                indices[tokens.index(token)] = id

        return indices, tokens

    def process(self, prompt):
        output = self.model(prompt, max_tokens=1)
        # tokens = self.tokenize(question.encode('utf-8'))
        # self.model.reset()
        # self.model.eval(tokens)


        logits = self.model.eval_logits
        indices = self.token_map[0]
        tokens = self.token_map[1]
        scores = np.asarray(logits[0])
        choices = sorted(list(zip(list(scores[indices]), tokens)), key=lambda x: x[0], reverse=True)

        return choices

# llm = GGMLLLM(model_path="data/allmodels/model/vicuna-33b.ggmlv3.q4_K_M.bin", tokens=['A','B','C','D','E'])
# choices = llm.process("Which of these is the capital of France ? \nA. Paris\nB. Kolkata\nC. Chennai\nD. Delhi")
# print(choices)