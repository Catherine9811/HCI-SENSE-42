import numpy as np
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import addcopyfighandler
from collections import defaultdict
import plotly.express as px
from collections import Counter
from data_parser import DataParser
from data_definition import psydat_files


def visualize_word_frequency(word_list):
    # Count word frequencies
    word_counts = Counter(word_list)

    # Convert to DataFrame
    df = pd.DataFrame(word_counts.items(), columns=['Word', 'Frequency'])
    df = df.sort_values(by='Frequency', ascending=False)

    # Create interactive bar chart
    fig = px.bar(df, x='Word', y='Frequency', title='Word Frequency Visualization',
                 labels={'Word': 'Words', 'Frequency': 'Count'},
                 color='Frequency', color_continuous_scale='blues')
    fig.update_layout(xaxis_tickangle=-45)
    fig.show()


def to_title_case(s):
    return s.replace("_", " ").title()


class KeyboardTextMaterialsExtractor:
    name = "keyboard_text_materials"

    def process(self, parser):
        x_values = []
        for task_name, task_key, task_prefix, task_keyboard in [
            ('Shadow Typing', 'mail_content', 'single_note', 'mail.mail_content_user_key_release'),
            ('Side-by-side Typing', 'notes_repeat', 'notes', 'notes.notes_repeat_keyboard')
        ]:
            typing_task = parser[task_key]

            for entry in typing_task:
                content = entry[f"{task_prefix}_repeat_source"]
                x_values.extend(content.lower().replace(".", "").split(" "))
        return x_values


if __name__ == '__main__':

    outcome_definition = KeyboardTextMaterialsExtractor()

    processed = {}

    for psydat_file in tqdm(psydat_files):
        participant_id = int(psydat_file.split("_")[0])
        parser = DataParser(os.path.join("..", "..", "data", psydat_file))
        outcome_values = outcome_definition.process(parser)
        if outcome_definition.name not in processed:
            processed[outcome_definition.name] = []
        processed[outcome_definition.name].extend(outcome_values)

    visualize_word_frequency(processed[outcome_definition.name])

