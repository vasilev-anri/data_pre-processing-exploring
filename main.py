import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    """
        Standardize numerical features (excluding target) using StandardScaler.

        Parameters:
            df (pd.DataFrame): Raw dataset including the target column 'DEATH_EVENT'.

        Returns:
            pd.DataFrame: Scaled features with the target column appended.
    """
    X = df.drop("DEATH_EVENT", axis=1)
    y = df["DEATH_EVENT"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    X_scaled_df["DEATH_EVENT"] = y.values
    return X_scaled_df


def remove_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)


def show_feature_roles(df):
    """
        Create a table indicating the role (Input/Target) and data type of each feature.

        Parameters:
            df (pd.DataFrame): Dataset including target column 'DEATH_EVENT'.

        Returns:
            pd.DataFrame: Table with columns 'Feature', 'Role', 'Data Type'.
    """
    roles = []
    for col in df.columns:
        role = "Target" if col == "DEATH_EVENT" else "Input"
        roles.append({
            "Feature": col,
            "Role": role,
            "Data Type": str(df[col].dtype)
        })
    roles_df = pd.DataFrame(roles)
    roles_df = roles_df.sort_values(by="Role")
    roles_df.reset_index(drop=True, inplace=True)
    return roles_df


def plot_scatter_2d(df):
    """
        Create a 2D scatter plot of Age vs Ejection Fraction, colored by DEATH_EVENT.

        Parameters:
            df (pd.DataFrame): Dataset (original or scaled) containing 'age',
                               'ejection_fraction', and 'DEATH_EVENT' columns.
    """
    plt.figure(figsize=(10, 7))
    ax = sns.scatterplot(
        data=df,
        x='age',
        y='ejection_fraction',
        hue='DEATH_EVENT',
        palette=['green', 'red'],
        alpha=0.85,
        s=75,
        edgecolor='black'
    )
    plt.title('Plot 1: Age vs Ejection Fraction', fontsize=14)
    plt.xlabel('Age')
    plt.ylabel('Ejection Fraction')
    plt.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles,
               labels=['Survived (0)', 'Died (1)'],
               title='Outcome',
               loc='upper right')


def plot_scatter_3d(df):
    """
        Create a 3D scatter plot: Age vs Ejection Fraction vs Serum Creatinine,
        colored by DEATH_EVENT.

        Parameters:
            df (pd.DataFrame): Dataset with columns 'age', 'ejection_fraction',
                               'serum_creatinine', and 'DEATH_EVENT'.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        df['age'],
        df['ejection_fraction'],
        df['serum_creatinine'],
        c=df['DEATH_EVENT'],
        cmap='RdYlGn_r',
        s=60,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    ax.set_xlabel('Age', fontsize=11)
    ax.set_ylabel('Ejection Fraction', fontsize=11)
    ax.set_zlabel('Serum Creatinine', fontsize=11)
    plt.title('3D Scatter Plot: Age vs Ejection Fraction vs Serum Creatinine', fontsize=13, pad=20)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor='green', markersize=10, label='Survived (0)'),
                       plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor='red', markersize=10, label='Died (1)')]
    ax.legend(handles=legend_elements, title='Outcome', loc='upper right')


def plot_histogram_1(df):
    """
        Plot stacked histogram with KDE for ejection fraction, split by outcome.

        Parameters:
            df (pd.DataFrame): Dataset containing 'ejection_fraction' and 'DEATH_EVENT'.
    """
    plt.figure(figsize=(8, 5))
    ax = sns.histplot(
        data=df,
        x="ejection_fraction",
        hue="DEATH_EVENT",
        hue_order=[0, 1],
        bins=20,
        kde=True,
        palette=['green', 'red'],
        alpha=0.7,
        multiple="stack"
    )
    custom_handles = [
        mpatches.Patch(color='green', alpha=0.7, label='Survived (0)'),
        mpatches.Patch(color='red', alpha=0.7, label='Died (1)')
    ]
    ax.legend(handles=custom_handles, title='Outcome', loc='upper right')
    plt.title("Ejection Fraction Distribution by Outcome")
    plt.xlabel("Ejection Fraction")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)


def plot_histogram_2(df):
    """
        Plot overlaid histogram with KDE for follow-up time, split by outcome.

        Parameters:
            df (pd.DataFrame): Dataset containing 'time' and 'DEATH_EVENT'.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x="time",
        hue="DEATH_EVENT",
        bins=30,
        kde=True,
        palette=['green', 'red'],
        alpha=0.6,
    )
    plt.title("Follow-up Time Distribution by Outcome")
    plt.xlabel("Time (days)")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)


def plot_boxplots(df):
    """
        Draw side-by-side boxplots for ejection fraction and serum creatinine by outcome.

        Parameters:
            df (pd.DataFrame): Dataset with 'DEATH_EVENT', 'ejection_fraction',
                               'serum_creatinine'.
    """
    plt.figure(figsize=(11, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(
        data=df,
        x="DEATH_EVENT",
        y="ejection_fraction",
        hue="DEATH_EVENT",
        palette=['green', 'red'],
        legend=False
    )
    plt.title("Ejection Fraction by Outcome")
    plt.xlabel("Outcome (0 = Survived, 1 = Died)")
    plt.ylabel("Ejection Fraction (%)")
    plt.subplot(1, 2, 2)
    sns.boxplot(
        data=df,
        x="DEATH_EVENT",
        y="serum_creatinine",
        hue="DEATH_EVENT",
        palette=['green', 'red'],
        legend=False
    )
    plt.title("Serum Creatinine by Outcome")
    plt.xlabel("Outcome (0 = Survived, 1 = Died)")
    plt.ylabel("Serum Creatinine (mg/dL)")
    plt.suptitle("Boxplots Showing Distribution by Outcome", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def plot_violinplots(df):
    """
        Draw side-by-side violin plots (with inner box) for ejection fraction and
        serum creatinine by outcome.

        Parameters:
            df (pd.DataFrame): Dataset with 'DEATH_EVENT', 'ejection_fraction',
                               'serum_creatinine'.
    """
    plt.figure(figsize=(11, 5))
    plt.subplot(1, 2, 1)
    sns.violinplot(
        data=df,
        x="DEATH_EVENT",
        y="ejection_fraction",
        hue="DEATH_EVENT",
        palette=['green', 'red'],
        legend=False,
        inner="box"
    )
    plt.title("Ejection Fraction Distribution by Outcome")
    plt.xlabel("Outcome (0 = Survived, 1 = Died)")
    plt.ylabel("Ejection Fraction (%)")
    plt.subplot(1, 2, 2)
    sns.violinplot(
        data=df,
        x="DEATH_EVENT",
        y="serum_creatinine",
        hue="DEATH_EVENT",
        palette=['green', 'red'],
        legend=False,
        inner="box"
    )
    plt.title("Serum Creatinine Distribution by Outcome")
    plt.xlabel("Outcome (0 = Survived, 1 = Died)")
    plt.ylabel("Serum Creatinine (mg/dL)")
    plt.suptitle("Violin Plots Showing Distribution by Outcome", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def plot_corr_heat_map(df):
    """
        Compute Pearson correlation matrix and display a heatmap (coolwarm colormap).

        Parameters:
            df (pd.DataFrame): Dataset (excluding duplicates recommended).
    """
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()


def calculate_statistics(df):
    """
        Compute central tendency (mean, median) and dispersion (std, min, max, range,
        variance, IQR) for continuous features. For binary features, show value counts.

        Parameters:
            df (pd.DataFrame): Dataset (raw, without normalization).

        Returns:
            pd.DataFrame: Statistics table for continuous features.
    """
    binary_cols = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT']
    numerical_cols = [col for col in df.columns if col not in binary_cols]
    stats_list = []
    for col in numerical_cols:
        data = df[col].dropna()
        stats = {
            'Feature': col,
            'Mean': data.mean(),
            'Median': data.median(),
            'Std Dev': data.std(),
            'Min': data.min(),
            'Max': data.max(),
            'Range': data.max() - data.min(),
            'Variance': data.var(),
            'IQR': data.quantile(0.75) - data.quantile(0.25)
        }
        stats_list.append(stats)
    stats_df = pd.DataFrame(stats_list)
    stats_df = stats_df.sort_values(by="Std Dev", ascending=False)
    print("\n=== Continuous Features Statistics (sorted by Std Dev) ===")
    print(stats_df.round(2).to_string(index=False))
    print("\n=== Categorical / Binary Features ===")
    for col in binary_cols:
        if col in df.columns:
            print("\n")
            print(df[col].value_counts().to_string())
    return stats_df


#  PART 2

def hierarchical_clustering(df):
    """
        Perform hierarchical clustering (Ward linkage) on scaled features (no target).
        Plot dendrogram with three cut-off lines (20, 30, 45). For each cut-off,
        print number of clusters, cluster sizes, and composition w.r.t true labels.
        Also show subplots of Age vs Ejection Fraction colored by cluster.

        Parameters:
            df (pd.DataFrame): Preprocessed dataset (scaled, with target column).
    """
    X = df.drop("DEATH_EVENT", axis=1)
    Z = linkage(X, method='ward')

    plt.figure(figsize=(12, 8))
    dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=10.)
    plt.title('Hierarchical Clustering Dendrogram (Ward linkage)')
    plt.xlabel('Data points (or cluster size)')
    plt.ylabel('Distance')
    plt.axhline(y=20, color='r', linestyle='--', label='Cut at 20 ( Experiment 1 )')
    plt.axhline(y=30, color='g', linestyle='--', label='Cut at 30 ( Experiment 2 )')
    plt.axhline(y=45, color='b', linestyle='--', label='Cut at 45 ( Experiment 3 )')
    plt.ylim(0, 60)
    plt.legend()

    cutoffs = [20, 30, 45]
    exp_labels = {}
    for i, cutoff in enumerate(cutoffs, 1):
        labels = fcluster(Z, cutoff, criterion='distance')
        n_clusters = len(set(labels))
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n--- Experiment {i}: cut-off = {cutoff} ---")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Cluster sizes:      {dict(zip(unique, counts))}")
        true_labels = df["DEATH_EVENT"].values
        for cl in unique:
            mask = labels == cl
            survived = (true_labels[mask] == 0).sum()
            died = (true_labels[mask] == 1).sum()
            print(f"    Cluster {cl}: survived={survived}, died={died}")
        exp_labels[i] = labels

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, cutoff in enumerate(cutoffs, 1):
        labels = fcluster(Z, cutoff, criterion='distance')
        n_clusters = len(set(labels))
        ax = axes[i - 1]
        scatter = ax.scatter(
            df['age'], df['ejection_fraction'],
            c=labels, cmap='tab10', alpha=0.6, s=20, edgecolors='none'
        )
        ax.set_title(f'Exp {i}: cut={cutoff} - {n_clusters} clusters')
        ax.set_xlabel('Age')
        ax.set_ylabel('Ejection Fraction')
        plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.suptitle('Hierarchical Clustering — Age vs Ejection Fraction', fontsize=14)
    plt.tight_layout()


def k_means(df, k_range=range(2, 8)):
    """
        Apply K-means clustering for k in k_range (default 2..7). Compute silhouette
        scores, find best k, plot scores (with best k marked) and visualise clusters
        (Age vs Ejection Fraction) alongside true labels.

        Parameters:
            df (pd.DataFrame): Preprocessed dataset (scaled, with target column).
            k_range (range): Range of k values to evaluate.

        Returns:
            tuple: (list of silhouette scores, best k value)
    """
    X = df.drop("DEATH_EVENT", axis=1)
    scores = []

    print("\n K-Means: Silhouette Scores ")
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels, metric='euclidean')
        scores.append(score)
        print(f"  k = {k} : silhouette score = {score:.4f}")

    k_values = list(k_range)
    best_k = k_values[np.argmax(scores)]
    print(f"\n  Best k = {best_k}  (silhouette = {max(scores):.4f})")

    plt.figure(figsize=(9, 5))
    plt.plot(k_values, scores, marker='o', linestyle='-', markersize=7, color='steelblue')
    plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best k = {best_k}')
    plt.title('K-Means: Silhouette Score vs Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    best_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = best_model.fit_predict(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(
        df['age'], df['ejection_fraction'],
        c=cluster_labels, cmap='tab10', alpha=0.6, s=20, edgecolors='none'
    )
    axes[0].set_title(f'K-Means clusters (k={best_k})')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Ejection Fraction')

    colors_true = ['green' if v == 0 else 'red' for v in df['DEATH_EVENT']]
    axes[1].scatter(
        df['age'], df['ejection_fraction'],
        c=colors_true, alpha=0.5, s=20, edgecolors='none'
    )
    axes[1].set_title('True Labels (green=survived, red=died)')
    axes[1].set_xlabel('Age')
    axes[1].set_ylabel('Ejection Fraction')
    plt.suptitle(f'K-Means (k={best_k}) vs True Labels — Age vs Ejection Fraction', fontsize=13)
    plt.tight_layout()

    return scores, best_k


#  PART 3


def split_data(df):
    """
        Split dataset into training (80%) and test (20%) sets with stratification
        on the target variable 'DEATH_EVENT'.

        Parameters:
            df (pd.DataFrame): Cleaned, non‑scaled dataset (or scaled – but scaling
                               should be applied after split).

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop("DEATH_EVENT", axis=1)
    y = df["DEATH_EVENT"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def print_split_details(X_train, X_test, y_train, y_test):
    """
        Print total number of samples, train/test sizes (counts and percentages),
        and class distribution for the training and test sets.

        Parameters:
            X_train, X_test, y_train, y_test: Output from train_test_split.
    """
    total = len(X_train) + len(X_test)
    print(f"\n Dataset Split ")
    print(f"Total data objects: {total}")
    print(f"Training set: {len(X_train)} ({len(X_train) / total * 100:.1f}%)")
    print(f"Test set:     {len(X_test)} ({len(X_test) / total * 100:.1f}%)")
    print("\nClass distribution (0 = survived, 1 = died):")
    print(f"  Train - 0: {(y_train == 0).sum()} ({(y_train == 0).mean() * 100:.1f}%)")
    print(f"         1: {(y_train == 1).sum()} ({(y_train == 1).mean() * 100:.1f}%)")
    print(f"  Test  - 0: {(y_test == 0).sum()} ({(y_test == 0).mean() * 100:.1f}%)")
    print(f"         1: {(y_test == 1).sum()} ({(y_test == 1).mean() * 100:.1f}%)")


def _cv_evaluate(model, X_train, y_train, cv=5):
    """
        Helper function: evaluate a model using 5‑fold cross‑validation on the
        training set only. Returns mean accuracy and mean F1‑score.

        Parameters:
            model: Scikit‑learn estimator (not yet fitted).
            X_train (array-like): Training features.
            y_train (array-like): Training labels.
            cv (int): Number of folds.

        Returns:
            tuple: (mean_accuracy, mean_f1)
    """
    acc = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
    f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1').mean()
    return acc, f1


def _final_test(model, X_train, X_test, y_train, y_test):
    """
    Helper function: fit the model on the full training set and evaluate
    once on the test set. Returns a dictionary with accuracy, F1‑score,
    classification report, and the fitted model.

    Parameters:
        model: Scikit‑learn estimator.
        X_train, X_test, y_train, y_test: Data splits.

    Returns:
        dict: {'accuracy': float, 'f1': float, 'report': str, 'model': fitted model}
    """
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "report": classification_report(y_test, pred,
                                        target_names=['Survived (0)', 'Died (1)']),
        "model": model
    }


def mlp_experiments(X_train, y_train):
    """
    Perform three experiments with MLP classifiers using different architectures
    and activation functions. Evaluate each via 5‑fold CV on training data.
    Print hyperparameters and CV results, then return list of (name, acc, f1, model).

    Parameters:
        X_train (array-like): Training features (scaled).
        y_train (array-like): Training labels.

    Returns:
        list: List of tuples (model_name, accuracy, f1_score, estimator).
    """
    configs = [
        MLPClassifier(hidden_layer_sizes=(50,), activation='relu', max_iter=300, random_state=42),
        MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=400, random_state=42),
        MLPClassifier(hidden_layer_sizes=(100, 100), activation='tanh', max_iter=500, random_state=42),
    ]

    results = []
    print("\n-- MLP (Neural Network) Training Experiments --")
    for i, model in enumerate(configs, 1):
        print(f"\nMLP Experiment {i} hyperparameters:")
        print(f"  hidden_layer_sizes = {model.hidden_layer_sizes}")
        print(f"  activation         = {model.activation}")
        print(f"  max_iter           = {model.max_iter}")
        acc, f1 = _cv_evaluate(model, X_train, y_train)
        print(f"MLP Experiment {i} CV results:")
        print(f"  Accuracy = {acc:.4f}  |  F1 = {f1:.4f}")
        results.append((f"MLP_{i}", acc, f1, model))

    best = max(results, key=lambda t: t[2])
    print(f"\n  - Best MLP config: {best[0]} (CV F1 = {best[2]:.4f})")
    return results


def mlp_final_test(results, X_train, X_test, y_train, y_test):
    """
        Select the best MLP model from the CV experiments, train it on the full
        training set, evaluate on the test set, and print the results.

        Parameters:
            results: Output from mlp_experiments.
            X_train, X_test, y_train, y_test: Data splits.

        Returns:
            dict: Final test metrics (accuracy, f1, report, fitted model).
    """
    best_model = max(results, key=lambda t: t[2])[3]
    res = _final_test(best_model, X_train, X_test, y_train, y_test)
    print("\n-- MLP Final Test Results --")
    print(f"  Accuracy : {res['accuracy']:.4f}")
    print(f"  F1-score : {res['f1']:.4f}")
    print("\nClassification Report:")
    print(res['report'])
    return res


def rf_experiments(X_train, y_train):
    """
        Perform three experiments with Random Forest classifiers (varying n_estimators
        and max_depth). Evaluate via 5‑fold CV on training set.
        Returns list of (name, acc, f1, model).
    """
    configs = [
        RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
        RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
        RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42),
    ]

    results = []
    print("\n-- Random Forest Training Experiments --")
    for i, model in enumerate(configs, 1):
        print(f"\nRF Experiment {i} hyperparameters:")
        print(f"  n_estimators = {model.n_estimators}")
        print(f"  max_depth    = {model.max_depth}")
        print(f"  random_state = {model.random_state}")
        acc, f1 = _cv_evaluate(model, X_train, y_train)
        print(f"RF Experiment {i} CV results:")
        print(f"  Accuracy = {acc:.4f}  |  F1 = {f1:.4f}")
        results.append((f"RF_{i}", acc, f1, model))

    best = max(results, key=lambda t: t[2])
    print(f"\n  - Best RF config: {best[0]} (CV F1 = {best[2]:.4f})")
    return results


def rf_final_test(results, X_train, X_test, y_train, y_test):
    """
       Select best Random Forest model from CV, retrain on full training set,
       evaluate on test set, print metrics.
   """
    best_model = max(results, key=lambda t: t[2])[3]
    res = _final_test(best_model, X_train, X_test, y_train, y_test)
    print("\n-- Random Forest Final Test Results --")
    print(f"  Accuracy : {res['accuracy']:.4f}")
    print(f"  F1-score : {res['f1']:.4f}")
    print("\nClassification Report:")
    print(res['report'])
    return res


def knn_experiments(X_train, y_train):
    """
        Perform three experiments with k‑NN classifiers (varying n_neighbors,
        weights, metric). Evaluate via 5‑fold CV.
        Returns list of (name, acc, f1, model).
    """
    configs = [
        KNeighborsClassifier(n_neighbors=3),
        KNeighborsClassifier(n_neighbors=5, weights="distance"),
        KNeighborsClassifier(n_neighbors=11, metric="manhattan"),
    ]

    results = []
    print("\n-- KNN Training Experiments --")
    for i, model in enumerate(configs, 1):
        print(f"\nKNN Experiment {i} hyperparameters:")
        print(f"  n_neighbors = {model.n_neighbors}")
        print(f"  weights     = {model.weights}")
        print(f"  metric      = {model.metric}")
        acc, f1 = _cv_evaluate(model, X_train, y_train)
        print(f"KNN Experiment {i} CV results:")
        print(f"  Accuracy = {acc:.4f}  |  F1 = {f1:.4f}")
        results.append((f"KNN_{i}", acc, f1, model))

    best = max(results, key=lambda t: t[2])
    print(f"\n  - Best KNN config: {best[0]} (CV F1 = {best[2]:.4f})")
    return results


def knn_final_test(results, X_train, X_test, y_train, y_test):
    """
        Select best k‑NN model from CV, retrain on full training set,
        evaluate on test set, print metrics.
    """
    best_model = max(results, key=lambda t: t[2])[3]
    res = _final_test(best_model, X_train, X_test, y_train, y_test)
    print("\n-- KNN Final Test Results --")
    print(f"  Accuracy : {res['accuracy']:.4f}")
    print(f"  F1-score : {res['f1']:.4f}")
    print("\nClassification Report:")
    print(res['report'])
    return res


def build_results_table(mlp_res, rf_res, knn_res):
    """
        Combine training experiment results from all three algorithms into a single
        DataFrame, sort by F1‑score, and display the summary.

        Parameters:
            mlp_res, rf_res, knn_res: Lists from the respective experiment functions.

        Returns:
            pd.DataFrame: Sorted table with columns Model, Accuracy, F1-score.
    """
    all_results = mlp_res + rf_res + knn_res
    df = pd.DataFrame(all_results, columns=["Model", "Accuracy", "F1-score", "Estimator"])
    df = df.sort_values(by="F1-score", ascending=False)
    print("\n--- TRAINING EXPERIMENTS SUMMARY ---")
    print(df[["Model", "Accuracy", "F1-score"]].to_string(index=False))
    return df


def build_final_comparison(mlp_test_res, rf_test_res, knn_test_res):
    """
        Create a comparison table for the best model of each algorithm evaluated
        on the test set. Sorted by F1‑score.

        Parameters:
            mlp_test_res, rf_test_res, knn_test_res: Dictionaries returned by
            the *_final_test functions.

        Returns:
            pd.DataFrame: Test‑set comparison table.
    """
    rows = [
        ("MLP (best)", mlp_test_res["accuracy"], mlp_test_res["f1"]),
        ("Random Forest (best)", rf_test_res["accuracy"], rf_test_res["f1"]),
        ("KNN (best)", knn_test_res["accuracy"], knn_test_res["f1"]),
    ]
    df = pd.DataFrame(rows, columns=["Model", "Accuracy", "F1-score"])
    df = df.sort_values(by="F1-score", ascending=False).reset_index(drop=True)
    print("\n=== FINAL MODEL COMPARISON — TEST SET RESULTS ===")
    print(df.to_string(index=False))
    return df


def plot_model_comparison(results_df):
    """
        Create a bar chart comparing Accuracy and F1‑score of the best models from
        MLP, Random Forest, and KNN (test set results). Adds value labels, legend,
        and a dashed line marking the best score.

        Parameters:
            results_df (pd.DataFrame): Output from build_final_comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    models = results_df["Model"]
    palette = {'MLP': 'steelblue', 'Random': 'darkorange', 'KNN': 'mediumseagreen'}
    colors = [palette.get(m.split(' ')[0].split('_')[0], 'gray') for m in models]

    for ax, metric in zip(axes, ["Accuracy", "F1-score"]):
        bars = ax.bar(models, results_df[metric], color=colors, edgecolor='black', alpha=0.85)
        ax.set_title(f'Model Comparison — {metric}')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=results_df[metric].max(), color='red', linestyle='--', alpha=0.6, label='Best')
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in palette.items()]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=10)
    plt.suptitle('Final Model Comparison — Test Set', fontsize=14)
    plt.tight_layout(rect=[0, 0.07, 1, 1])


if __name__ == '__main__':
    df = load_data("data/heart_failure_clinical_records.csv")
    clean_df = remove_duplicates(df)

    # Part 1
    plot_scatter_2d(df)
    plot_scatter_3d(df)
    plot_histogram_1(df)
    plot_histogram_2(df)
    plot_boxplots(df)
    plot_violinplots(df)
    plot_corr_heat_map(clean_df)
    calculate_statistics(df)

    # Part 2
    scaled_df = preprocess_data(clean_df)
    hierarchical_clustering(scaled_df)
    k_means(scaled_df)

    # Part 3
    X_train, X_test, y_train, y_test = split_data(clean_df)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print_split_details(X_train_sc, X_test_sc, y_train, y_test)

    mlp_res = mlp_experiments(X_train_sc, y_train)
    rf_res = rf_experiments(X_train_sc, y_train)
    knn_res = knn_experiments(X_train_sc, y_train)

    build_results_table(mlp_res, rf_res, knn_res)

    print("--FINAL TESTING — held-out test set (run once per model)--")
    mlp_test_res = mlp_final_test(mlp_res, X_train_sc, X_test_sc, y_train, y_test)
    rf_test_res = rf_final_test(rf_res, X_train_sc, X_test_sc, y_train, y_test)
    knn_test_res = knn_final_test(knn_res, X_train_sc, X_test_sc, y_train, y_test)

    final_df = build_final_comparison(mlp_test_res, rf_test_res, knn_test_res)
    plot_model_comparison(final_df)

    plt.show()