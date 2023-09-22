# kaggle_science_llm

Kaggle competition to see if small llms can answer questions created by big llms. 
My conclusion
1. The questions provided in competition are simple and only require information present in a wikipedia paragraph to solve the question.
2.  Without any finetuning vicuna is able to answer 85% of questions with some prompt tuning
3.  deberta nli which is finetuned to answer mcq and other questions does better at 87%

If we finetune llama for this specific task, and dont maintain performance on others, the result LLM will be useless in real life as it wont be able to solve problems that require information beyond the paragraph. 

Solution:
1. Chunk wikipedia into sections. Embed sections and store them. 
2. Retrieve relevant section by embedding question, question+answers and use a task based encoder (e.g. bge) and matching with stored embeddings. Retrieve top k. 
3. Break the section into paragraphs and rerank them to get a small set of paragraphs.
4. If using LLM like vicuna, prompt with an expert prompt "you are an expert in wikipedia_title, ..". Vicuna has bias towards option positioning. So run the model 5 times changing position of options.
6. If tuning a model like deberta, use one that has been finetuned on tasksource, which has lot of contextual mcqa questions. 
