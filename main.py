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






if __name__ == '__main__':
    df = load_data("data/heart_failure_clinical_records.csv")
    df, x_scaled_df = preprocess_data(df)

    plot_scatter_2d(x_scaled_df)
    plot_scatter_3d(x_scaled_df)
    plot_histogram_1(df)
    plot_histogram_2(df)

    plt.show()