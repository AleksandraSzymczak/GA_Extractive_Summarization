import mosestokenizer as ms
import numpy as np
import pandas as pd
# import kagglehub

# Download latest version
# path = kagglehub.dataset_download("gowrishankarp/newspaper-text-summarization-cnn-dailymail")

# print("Path to dataset files:", path)
# path to data
# /home/ola/.cache/kagglehub/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail/versions/2/cnn_dailymail
test_data = pd.read_csv("/home/ola/.cache/kagglehub/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail/versions/2/cnn_dailymail/train.csv")
print(test_data.head())
tokenize = ms.MosesTokenizer('en')
array = np.empty((0,3))
unique_words = set()
for x in test_data["highlights"]:
    output = tokenize(str(x))
    unique_words.update([word.lower().strip('.,!?:;()') for word in output if word != ''])
random_grades = np.random.uniform(low=0.0, high=1.0, size=(len(output),))
np.append(array, random_grades, axis=0)
print(array)
tokenize.close()
