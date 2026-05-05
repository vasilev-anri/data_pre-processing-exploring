import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np


def load_data(path):
    df = pd.read_csv(path)
    # print(df.describe())
    # print(df.describe().T[['min', 'max', 'mean', 'std']])
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # print(df.describe().round(2))
    # print(df.info())
    return df

def preprocess_data(df):
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
    plt.figure(figsize=(8,5))

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
    plt.figure(figsize=(11, 5))

    # jection Fraction
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

    #  Serum Creatinine
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
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()


def calculate_statistics(df):

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
            # print(f"\n{col}:")
            print("\n")
            print(df[col].value_counts().to_string())

    return stats_df


def hierarchical_clustering(df):
    X = df.drop("DEATH_EVENT", axis=1)
    Z = linkage(X, method='ward')

    plt.figure(figsize=(12, 8))
    dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=10.)
    plt.title('Dendrogram (Ward linkage)')
    plt.xlabel('Data points')
    plt.ylabel('Distance')
    plt.axhline(y=70, color='r', linestyle='--', label='Cut at distance 70')
    plt.axhline(y=37, color='g', linestyle='--', label='Cut at distance 37')
    plt.axhline(y=20, color='b', linestyle='--', label='Cut at distance 20')
    plt.legend()
    plt.show()

    cutoffs = [20, 37, 70]
    for cutoff in cutoffs:
        labels = fcluster(Z, cutoff, criterion='distance')
        n_clusters = len(set(labels))
        print(f"Cut-off distance = {cutoff} -> Number of clusters: {n_clusters}")
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Cluster sizes: {dict(zip(unique, counts))}")



def k_means(df, k_range=range(2, 7)):
    X = df.drop("DEATH_EVENT", axis=1)

    scores = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels, metric='euclidean')
        scores.append(score)
        print(f"k = {k} : silhouette score = {score:.4f}")

    k_values = list(k_range)
    best_k = k_values[np.argmax(scores)]
    print(f"Best k = {best_k}")

    plt.figure(figsize=(10, 8))
    plt.plot(list(k_range), scores, marker='o', linestyle='-', markersize=5, label='silhouette')
    plt.title("silhouette score vs number of clusters")
    plt.xlabel("number of clusters (k)")
    plt.ylabel("silhouette score")
    plt.grid(True)

    return scores, best_k


def split_data(df):
    X = df.drop("DEATH_EVENT", axis=1)
    y = df["DEATH_EVENT"]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "model": model
    }

def print_split_details(X_train, X_test, y_train, y_test):
    total = len(X_train) + len(X_test)
    print(f"Total data objects: {total}")
    print(f"Training set: {len(X_train)} ({len(X_train)/total*100:.1f}%)")
    print(f"Test set:     {len(X_test)} ({len(X_test)/total*100:.1f}%)")
    print("\nClass distribution (0 = survived, 1 = died):")
    print(f"  Train - 0: {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
    print(f"         1: {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
    print(f"  Test  - 0: {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%)")
    print(f"         1: {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")



def mlp_experiments(X_train, X_test, y_train, y_test):

    configs = [
        MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=42),
        MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=400, random_state=42),
        MLPClassifier(hidden_layer_sizes=(100, 100), activation="tanh", max_iter=500, random_state=42),
    ]

    results = []

    for i, model in enumerate(configs, 1):
        res = evaluate_model(model, X_train, X_test, y_train, y_test)
        print(f"\nMLP Experiment {i}: F1={res['f1']:.4f}, Acc={res['accuracy']:.4f}")
        results.append((f"MLP_{i}", res["accuracy"], res["f1"], model))

    return results







def rf_experiments(X_train, X_test, y_train, y_test):

    configs = [
        RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
        RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
        RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42),
    ]

    results = []

    for i, model in enumerate(configs, 1):
        res = evaluate_model(model, X_train, X_test, y_train, y_test)
        print(f"\nRF Experiment {i}: F1={res['f1']:.4f}, Acc={res['accuracy']:.4f}")
        results.append((f"RF_{i}", res["accuracy"], res["f1"], model))

    return results





def knn_experiments(X_train, X_test, y_train, y_test):

    configs = [
        KNeighborsClassifier(n_neighbors=3),
        KNeighborsClassifier(n_neighbors=5, weights="distance"),
        KNeighborsClassifier(n_neighbors=11, metric="manhattan"),
    ]

    results = []

    for i, model in enumerate(configs, 1):
        res = evaluate_model(model, X_train, X_test, y_train, y_test)
        print(f"\nKNN Experiment {i}: F1={res['f1']:.4f}, Acc={res['accuracy']:.4f}")
        results.append((f"KNN_{i}", res["accuracy"], res["f1"], model))

    return results


def build_results_table(mlp_res, rf_res, knn_res):
    all_results = mlp_res + rf_res + knn_res

    df = pd.DataFrame(all_results, columns=[
        "Model",
        "Accuracy",
        "F1-score",
        "Estimator"
    ])

    df = df.sort_values(by="F1-score", ascending=False)

    print("\n===== FINAL MODEL COMPARISON =====")
    print(df[["Model", "Accuracy", "F1-score"]])

    return df






if __name__ == '__main__':
    df = load_data("data/heart_failure_clinical_records.csv")

    clean_df = remove_duplicates(df)

    # split first
    X_train, X_test, y_train, y_test = split_data(clean_df)

    # scaling (NO leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print_split_details(X_train, X_test, y_train, y_test)

    # models
    mlp_res = mlp_experiments(X_train, X_test, y_train, y_test)
    rf_res = rf_experiments(X_train, X_test, y_train, y_test)
    knn_res = knn_experiments(X_train, X_test, y_train, y_test)

    final_table = build_results_table(mlp_res, rf_res, knn_res)


    # hierarchical_clustering(x_scaled_df)

    # k_means(x_scaled_df)

    # mlp_experiments(x_scaled_df)

    # rf_experiments(x_scaled_df)

    # knn_experiments(x_scaled_df)


    # print_split_details(X_train, X_test, y_train, y_test)
    #
    #
    # mlp_res = mlp_experiments(X_train, X_test, y_train, y_test)
    # rf_res = rf_experiments(X_train, X_test, y_train, y_test)
    # knn_res = knn_experiments(X_train, X_test, y_train, y_test)
    #
    # final_table = build_results_table(mlp_res, rf_res, knn_res)


    # labels = hierarchical_clustering(x_scaled_df)

    # plot_scatter_2d(df)

    # clean_df = remove_duplicates(df)
    # plot_scatter_2d(clean_df)
    # plot_scatter_3d(clean_df)
    # plot_histogram_1(clean_df)
    # plot_histogram_2(clean_df)
    # plot_boxplots(clean_df)
    # plot_violinplots(clean_df)
    # calculate_statistics(clean_df)

    # print("\nBEFORE NORMALIZATION")
    # print(df.describe().round(3).to_string())
    #
    # print("\nAFTER NORMALIZATION")
    # print(x_scaled_df.describe().round(3).to_string())

    # roles_df = show_feature_roles(df)
    # print(roles_df.to_string(index=False))

    # plot_scatter_2d(x_scaled_df)
    # plot_scatter_3d(x_scaled_df)
    # plot_histogram_1(df)
    # plot_histogram_2(df)
    # plot_boxplots(df)
    # plot_violinplots(df)
    # calculate_statistics(df)

    # print(df.head(10).to_string())      # show data file structure


    #
    # duplicated_rows = df[df.duplicated()]
    # print(duplicated_rows)

    # print(df[df.duplicated(keep=False)].shape)
    # print(df.drop_duplicates().shape)

    # duplicates = df[df.duplicated(keep=False)]
    # print(duplicates.sample(10).to_string(index=False))
    #
    # dups = df[df.duplicated(keep=False)].sort_values(df.columns.tolist())
    # print(dups.to_string(index=False))

    # print("Total rows:", len(df))
    # print("Duplicate rows:", df.duplicated().sum())

    # duplicates = df[df.duplicated(keep=False)]
    # print(duplicates.head(10))

    # dups = df[df.duplicated(keep=False)].sort_values(df.columns.tolist())
    # print(dups.to_string(index=False))

    # print("Unique rows:", len(df.drop_duplicates()))
    # print("Duplicated rows removed:", len(df) - len(df.drop_duplicates()))

    # print("Unique rows:", len(df.drop_duplicates()))
    # print("Dataset shape:", df.shape)

    # df_clean = remove_duplicates(df)
    #
    #
    # plot_corr_heat_map(df_clean)




    # remove self-correlation
    # corr_pairs = corr_pairs[corr_pairs < 1]
    #
    # print(corr_pairs.head(10))


    # print("Original:", df.shape)
    # print("Cleaned:", df_clean.shape)
    #
    # print("Duplicates in cleaned dataset:", df_clean.duplicated().sum())
    #
    # print(df_clean[df_clean.duplicated(keep=False)])
    #
    # assert df_clean.duplicated().sum() == 0

    # df_clean = df.drop_duplicates().reset_index(drop=True)
    #
    # print("Original size:", df.shape)
    # print("After deduplication:", df_clean.shape)
    # print("Removed:", df.shape[0] - df_clean.shape[0])
    #
    # print(df.duplicated(keep=False).sum())
    #
    # print(df[df.duplicated(keep=False)].head(10))

    plt.show()