import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import addcopyfighandler

# Load the data
df = pd.read_csv('./processed_data/questionnaire/42-questionnaires.csv')

# Display basic info about the data
print("Data shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Create a unique identifier for each (initiation, participant) pair
# Group by initiation, participant, and name (ignoring time)
grouped_data = df.groupby(['initiation', 'participant', 'name'])['value'].mean().reset_index()

# Pivot the data to have names as columns
pivot_df = grouped_data.pivot_table(
    index=['initiation', 'participant'],
    columns='name',
    values='value'
).reset_index()

# Display the pivoted data
print("\nPivoted data shape:", pivot_df.shape)
print("\nAvailable columns after pivot:", pivot_df.columns.tolist())
print("\nPivoted data head:")
print(pivot_df.head())

# Check if we have multiple questionnaire names for correlation analysis
questionnaire_names = df['name'].unique()
print(f"\nUnique questionnaire names: {questionnaire_names}")

if len(questionnaire_names) > 1:
    # Calculate correlation matrix
    correlation_cols = [col for col in pivot_df.columns if col not in ['initiation', 'participant']]

    if len(correlation_cols) > 1:
        correlation_matrix = pivot_df[correlation_cols].corr()

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            square=True,
            cbar_kws={"shrink": 0.8}
        )

        plt.title('Correlation Heatmap of Questionnaire Responses\n(Grouped by Initiation and Participant)', pad=20,
                  fontsize=14)
        plt.tight_layout()
        # plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\nCorrelation Matrix:")
        print(correlation_matrix)
    else:
        print(f"\nOnly one questionnaire type found: {correlation_cols}")
        print("Cannot create correlation matrix with only one variable.")

        # Create a simple visualization of the single variable
        plt.figure(figsize=(10, 6))
        plt.scatter(pivot_df['initiation'], pivot_df[correlation_cols[0]], alpha=0.6)
        plt.xlabel('Initiation')
        plt.ylabel(correlation_cols[0])
        plt.title(f'{correlation_cols[0]} by Initiation')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # plt.savefig('single_variable_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
else:
    print(f"\nOnly one type of questionnaire found: {questionnaire_names[0]}")
    print("Cannot create correlation matrix with only one questionnaire type.")
