import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def make_bar_graph(data_dict, color, title):
    plt.style.use("fivethirtyeight")
    x = np.arange(len(data_dict))
    f, ax = plt.subplots(figsize=(15, 5))
    ax.set_title(title, size=20)
    ax.bar(x, data_dict.values(), width=0.8, align="center", color=color)
    plt.ylabel("Number of comments")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(label).replace(" product or service", "") for label in data_dict.keys()]
    )
    plt.savefig(f"static/images/{title}.png")


def main():
    data = pd.read_csv("product_sentiment.csv").to_numpy()
    negative_emotions = defaultdict(int)
    positive_emotions = defaultdict(int)
    neutral_emotions = defaultdict(int)

    for row in data:
        if row[2] is np.nan:
            continue
        if row[3] == "Negative emotion":
            negative_emotions[row[2]] += 1
        elif row[3] == "Positive emotion":
            positive_emotions[row[2]] += 1
        elif row[3] == "No emotion toward brand or product":
            neutral_emotions[row[2]] += 1

    make_bar_graph(negative_emotions, "firebrick", "Negative comments")
    make_bar_graph(positive_emotions, "forestgreen", "Positive comments")
    make_bar_graph(neutral_emotions, "royalblue", "Neutral comments")


if __name__ == "__main__":
    main()
