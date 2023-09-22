# kaggle_science_llm

Kaggle competition to see if small llms can answer questions created by big llms. 
My conclusion
1. The questions provided in competition are simple and only require information present in a wikipedia paragraph to solve the question.
2.  Without any finetuning vicuna is able to answer 85% of questions with some prompt tuning
3.  deberta nli which is finetuned to answer mcq and other questions does better at 87%

If we finetune llama for this specific task, and dont maintain performance on others, the result LLM will be useless in real life as it wont be able to solve problems that require information beyond the paragraph. 
