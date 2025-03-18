from faker import Faker
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import addcopyfighandler
from collections import defaultdict
import plotly.express as px
from collections import Counter


def visualize_word_frequency(word_list):
    # Count word frequencies
    word_counts = Counter(word_list)
    print(len(word_counts))
    # Convert to DataFrame
    df = pd.DataFrame(word_counts.items(), columns=['Word', 'Frequency'])
    df = df.sort_values(by='Frequency', ascending=False)

    # Create interactive bar chart
    fig = px.bar(df, x='Word', y='Frequency', title='Word Frequency Visualization',
                 labels={'Word': 'Words', 'Frequency': 'Count'},
                 color='Frequency', color_continuous_scale='blues')
    fig.update_layout(xaxis_tickangle=-45)
    fig.show()


fake = Faker(["en_US"], use_weighting=True)

words = []

for _ in range(88):
    content = fake.sentence(nb_words=10)
    words.extend(content.lower().replace(".", "").split(" "))

visualize_word_frequency(words)