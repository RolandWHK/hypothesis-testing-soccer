# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import pingouin


def load_and_filter(path: str, group: str) -> pd.DataFrame:
    """Load dataset and filter to World Cup matches since 2002."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= "2002-01-01") & (df["tournament"] == "FIFA World Cup")]
    df["group"] = group
    df["goals_scored"] = df["home_score"] + df["away_score"]
    return df


def run_test(men_path: str, women_path: str, alpha: float = 0.10) -> dict:
    """Run Mann-Whitney U test comparing women's vs men's WC goals."""
    men = load_and_filter(men_path, "men")
    women = load_and_filter(women_path, "women")

    # Quick histograms
    men["goals_scored"].hist(alpha=0.5, label="Men")
    women["goals_scored"].hist(alpha=0.5, label="Women")
    plt.legend()
    plt.title("Distribution of Goals per Match (since 2002)")
    plt.xlabel("Goals")
    plt.ylabel("Frequency")
    plt.savefig("reports/figures/goals_hist.png")
    plt.close()

    # Mann-Whitney U (right-tailed: women > men)
    results_pg = pingouin.mwu(
        x=women["goals_scored"],
        y=men["goals_scored"],
        alternative="greater",
    )
    p_val = results_pg["p-val"].values[0]

    result = "reject" if p_val < alpha else "fail to reject"
    return {"p_val": float(p_val), "result": result}


if __name__ == "__main__":
    result_dict = run_test("data/men_results.csv", "data/women_results.csv")
    print(result_dict)
