import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def load_data(path):
    df = pd.read_csv(path)
    print(df.describe())
    print(df.info())
    return df

def preprocess_data(df):
    x = df.drop("DEATH_EVENT", axis=1)
    y = df["DEATH_EVENT"]

    scaler = StandardScaler()

    x_scaled = scaler.fit_transform(x)
    x_scaled_df = pd.DataFrame(x_scaled, columns=x.columns)
    x_scaled_df["DEATH_EVENT"] = y.values

    return df, x_scaled_df


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

    plt.title('Plot 1: Age vs Ejection Fraction (Standardized)', fontsize=14)
    plt.xlabel('Age (standardized)')
    plt.ylabel('Ejection Fraction (standardized)')
    plt.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles,
               labels=['Survived (0)', 'Died (1)'],
               title='Outcome',
               loc='upper right')



def plot_scatter_3d(df):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
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

    ax.set_xlabel('Age (standardized)', fontsize=11)
    ax.set_ylabel('Ejection Fraction (standardized)', fontsize=11)
    ax.set_zlabel('Serum Creatinine (standardized)', fontsize=11)

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
            print(f"\n{col}:")
            print(df[col].value_counts().to_string())

    return stats_df



if __name__ == '__main__':
    df = load_data("data/heart_failure_clinical_records.csv")
    df, x_scaled_df = preprocess_data(df)

    roles_df = show_feature_roles(df)
    print(roles_df.to_string(index=False))

    plot_scatter_2d(x_scaled_df)
    plot_scatter_3d(x_scaled_df)
    plot_histogram_1(df)
    plot_histogram_2(df)
    plot_boxplots(df)
    plot_violinplots(df)
    calculate_statistics(df)

    plt.show()