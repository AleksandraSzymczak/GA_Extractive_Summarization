import pandas as pd
from sklearn.preprocessing import LabelEncoder
from geneticalgorithm import geneticalgorithm as ga
from rouge_score import rouge_scorer
import numpy as np
# Sample DataFrame
data = {
    'id': [
        '0001d1afc246a7964130f43ae940af6bc6c57f01',
        '0002095e55fcbd3a2f366d9bf92a95433dc305ef',
        '00027e965c8264c35cc1bc55556db388da82b07f',
        '0002c17436637c4fe1837c935c04de47adb18e9a',
        '0003ad6ef0c37534f80b55b4235108024b407f0b',
    ],
    'highlights': [
        'Bishop John Folda, of North Dakota, is taking ...',
        'Criminal complaint: Cop used his role to help ...',
        'Craig Eccleston-Todd, 27, had drunk at least t...',
        'Nina dos Santos says Europe must be ready to a...',
        'Fleetwood top of League One after 2-0 win at S...'
    ]
}

df = pd.DataFrame(data)

# Step 1: Extract unique words from highlights
unique_words = set()
for highlight in df['highlights']:
    # Splitting by space and normalizing to lower case
    words = highlight.split()
    unique_words.update([word.lower().strip('.,!?:;()') for word in words])

# Step 2: Assign values using LabelEncoder
label_encoder = LabelEncoder()
word_values = label_encoder.fit_transform(list(unique_words))

min_val, max_val = word_values.min(), word_values.max()
scaled_word_values = (word_values - min_val) / (max_val - min_val)

# Step 3: Create vocabulary DataFrame
vocabulary_df = pd.DataFrame({
    'word': label_encoder.inverse_transform(word_values),
    'value': scaled_word_values
})

# Display the vocabulary DataFrame
print(vocabulary_df)

def sentence_to_score_array(sentence):
    # Tokenize the sentence, convert to lowercase, and strip punctuation
    words = sentence.lower().split()
    words = [word.strip('.,!?:;()') for word in words]
    
    # Map each word to its value using the dictionary, default to 0 if not found
    score_array = [vocabulary_df.loc[vocabulary_df['word'] == word, 'value'].values[0] for word in words]
    return score_array

# Step 3: Apply the function to each row in the 'highlights' column
df['score_array'] = df['highlights'].apply(sentence_to_score_array)

# Display the resulting DataFrame
print(df[['id', 'highlights', 'score_array']])
scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

# Fitness function using ROUGE-1 score
def fitness_function(x):
    total_score = 0
    for idx, row in df.iterrows():
        # Generate weighted "summary" for each highlight based on GA weights
        generated_summary = " ".join(
            [word for word, weight in zip(row['highlights'].split(), x) if weight > 0.5]
        )
        reference_summary = row['highlights']
        
        # Calculate ROUGE-1 score between generated summary and reference
        rouge_score = scorer.score(reference_summary, generated_summary)['rouge1']
        avg_rouge_score = (rouge_score.fmeasure + rouge_score.precision + rouge_score.recall) / 3
        total_score += avg_rouge_score

    # Return average ROUGE score across all samples (maximize this)
    return -total_score / len(df)  # Negate because `geneticalgorithm` minimizes

# Define the bounds for the weights (0 to 1)
varbound = np.array([[0, 1]] * len(unique_words))

# Genetic algorithm parameters
algorithm_param = {
    'max_num_iteration': 100,
    'population_size': 5,
    'mutation_probability': 0.01,
    'elit_ratio': 0.01,
    'crossover_probability': 0.8,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': None
}

# Run the Genetic Algorithm
model = ga(function=fitness_function, dimension=len(unique_words), variable_type='real', variable_boundaries=varbound, algorithm_parameters=algorithm_param)
model.run()
solution = model.output_dict['variable']

for s in solution:
    print(vocabulary_df.loc[vocabulary_df['value'] == s, 'word'].values[0])
    breakpoint()