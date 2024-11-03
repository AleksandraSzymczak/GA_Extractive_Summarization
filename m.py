import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

# Step 3: Create vocabulary DataFrame
vocabulary_df = pd.DataFrame({
    'word': label_encoder.inverse_transform(word_values),
    'value': word_values
})

# Display the vocabulary DataFrame
print(vocabulary_df)
