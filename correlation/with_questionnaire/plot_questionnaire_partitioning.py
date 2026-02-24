import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# Load and prepare data
df = pd.read_csv('./processed_data/questionnaire/42-questionnaires.csv')

# Group by initiation and participant (ignoring time)
grouped = df.groupby(['initiation', 'participant', 'name'])['value'].mean().reset_index()
pivot_df = grouped.pivot_table(index=['initiation', 'participant'],
                               columns='name',
                               values='value').reset_index()

# Get questionnaire names (predictors)
predictors = [col for col in pivot_df.columns if col not in ['initiation', 'participant']]
print(f"Predictors found: {predictors}")

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pivot_df[predictors])
X_df = pd.DataFrame(X_scaled, columns=predictors)

# Method 3: Comprehensive variance partitioning plot
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Correlation heatmap
corr_matrix = X_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
            square=True, cbar_kws={"shrink": 0.8}, ax=axes[0, 0])
axes[0, 0].set_title('Correlation Matrix (Shared Variance)', fontsize=12)

# Plot 2: Variance decomposition using PCA
pca = PCA()
pca.fit(X_scaled)
explained_variance = pca.explained_variance_ratio_

axes[0, 1].bar(range(1, len(explained_variance) + 1), explained_variance,
               color='steelblue', alpha=0.7)
axes[0, 1].plot(range(1, len(explained_variance) + 1),
                np.cumsum(explained_variance), 'ro-')
axes[0, 1].set_xlabel('Principal Component')
axes[0, 1].set_ylabel('Explained Variance Ratio')
axes[0, 1].set_title('PCA - Variance Explained', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks(range(1, len(explained_variance) + 1))

# Plot 3: Variance partitioning (all predictors)
# Calculate unique and shared variances using R²
from sklearn.linear_model import LinearRegression

unique_variances = []
total_variances = X_df.var().values

for i, pred in enumerate(predictors):
    other_preds = [p for p in predictors if p != pred]
    if other_preds:
        X_others = X_df[other_preds]
        X_target = X_df[[pred]]

        model = LinearRegression()
        model.fit(X_others, X_target)
        y_pred = model.predict(X_others)
        residuals = X_target.values - y_pred
        unique_var = np.var(residuals)
    else:
        unique_var = total_variances[i]

    unique_variances.append(unique_var)

shared_variances = total_variances - np.array(unique_variances)

x = np.arange(len(predictors))
width = 0.35

axes[1, 0].bar(x - width / 2, unique_variances, width, label='Unique',
               color='#2E86AB', alpha=0.8)
axes[1, 0].bar(x + width / 2, shared_variances, width, label='Shared/Overlap',
               color='#A23B72', alpha=0.8)

axes[1, 0].set_xlabel('Predictors')
axes[1, 0].set_ylabel('Variance')
axes[1, 0].set_title('Unique vs Shared Variance', fontsize=12)
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(predictors, rotation=45 if len(predictors) > 3 else 0)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Visualize relationships with scatter matrix (if small number of predictors)
# Create a custom scatter matrix
from scipy.stats import gaussian_kde

n_predictors = len(predictors)
for i in range(n_predictors):
    for j in range(n_predictors):
        if i != j:
            axes[1, 1].scatter(X_df.iloc[:, j], X_df.iloc[:, i],
                               alpha=0.5, s=30, c='steelblue')
        else:
            # Diagonal: show histogram with density
            axes[1, 1].hist(X_df.iloc[:, i], bins=15, density=True,
                            alpha=0.7, color='steelblue')
            # Add kde
            kde = gaussian_kde(X_df.iloc[:, i])
            x_range = np.linspace(X_df.iloc[:, i].min(),
                                  X_df.iloc[:, i].max(), 100)
            axes[1, 1].plot(x_range, kde(x_range), 'r-', linewidth=2)

axes[1, 1].set_title('Predictor Relationships', fontsize=12)
axes[1, 1].set_xlabel('Predictors (standardized)')
axes[1, 1].set_ylabel('Predictors (standardized)')

plt.suptitle('Comprehensive Variance Analysis of Predictors', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('comprehensive_variance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "=" * 50)
print("Variance Partitioning Summary")
print("=" * 50)
for i, pred in enumerate(predictors):
    total_var = X_df[pred].var()
    unique_var = unique_variances[i] if i < len(unique_variances) else np.nan
    shared_var = shared_variances[i] if i < len(shared_variances) else np.nan

    print(f"\n{pred}:")
    print(f"  Total Variance: {total_var:.4f}")
    print(f"  Unique Variance: {unique_var:.4f} ({unique_var / total_var * 100:.1f}%)")
    print(f"  Shared Variance: {shared_var:.4f} ({shared_var / total_var * 100:.1f}%)")

# Calculate overall metrics
print("\n" + "=" * 50)
print("Overall Statistics")
print("=" * 50)
print(f"Total predictors: {len(predictors)}")
print(f"Average correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].mean():.3f}")
print(f"Total unique variance: {sum(unique_variances):.4f}")
print(f"Total shared variance: {sum(shared_variances):.4f}")