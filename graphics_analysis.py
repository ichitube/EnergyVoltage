import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], utc=False, errors="coerce")
    df = df[["timestamp_local", "total_load_mw"]].dropna()
    df = df.sort_values("timestamp_local").set_index("timestamp_local")
    df = df.groupby(level=0).mean()
    df = df.asfreq("H")

    if df["total_load_mw"].isna().any():
        df["total_load_mw"] = df["total_load_mw"].interpolate(method="time").ffill().bfill()

    return df


def plot_window(
    series: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    out_path: str,
    title: str,
    y_label: str = "Load (MW)",
):
    window = series.loc[start:end]

    fig, ax = plt.subplots(figsize=(16, 6), dpi=180)
    ax.plot(window.index, window.values, linewidth=2.0)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.grid(True, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    Path("figs").mkdir(exist_ok=True)
    Path("out_data").mkdir(exist_ok=True)

    csv_path = "data/pjm_2024_clean.csv"
    df = load_df(csv_path)

    df.to_csv("out_data/pjm_hourly_clean_for_plot.csv", index=True)

    start = df.index.min()
    end = start + pd.Timedelta(days=3)

    plot_window(
        series=df["total_load_mw"],
        start=start,
        end=end,
        out_path="figs/pjm_first_3_days_hourly.png",
        title="PJM — Hourly load (first 3 days)",
        y_label="Load (MW)",
    )

    print("Saved: out_data/pjm_hourly_clean_for_plot.csv")
    print("Saved: figs/pjm_first_3_days_hourly.png")


if __name__ == "__main__":
    main()
