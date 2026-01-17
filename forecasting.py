import pandas as pd
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_hourly_series(csv_path: str, ts_col: str = "timestamp_local", y_col: str = "total_load_mw") -> pd.Series:
    df = pd.read_csv(csv_path)
    df[ts_col] = pd.to_datetime(df[ts_col], utc=False, errors="coerce")
    df = df[[ts_col, y_col]].dropna()
    df = df.set_index(ts_col).sort_index()
    df = df.groupby(level=0).mean()
    df = df.asfreq("h")

    y = df[y_col].astype(float)
    if y.isna().any():
        y = y.interpolate(method="time").ffill().bfill()

    return y


def make_forecast_index(last_time: pd.Timestamp, steps: int, freq: str = "h") -> pd.DatetimeIndex:
    start = last_time + pd.Timedelta(hours=1)
    return pd.date_range(start=start, periods=steps, freq=freq)


def plot_timeline_actual_then_forecast(
    actual_hist: pd.Series,
    forecast: pd.Series,
    split_time: pd.Timestamp,
    out_path: str,
    title: str = "PJM — Actual then Forecast",
    y_label: str = "Load (MW)",
):
    fig, ax = plt.subplots(figsize=(16, 6), dpi=180)

    ax.plot(actual_hist.index, actual_hist.values, linewidth=2.0, label="Actual")
    ax.plot(forecast.index, forecast.values, linewidth=2.2, label="Forecast")

    ax.axvline(split_time, linewidth=1.2)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)
    ax.grid(True, linewidth=0.6)
    ax.legend()

    ax.set_xlim(actual_hist.index.min(), forecast.index.max())

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    Path("figs").mkdir(exist_ok=True)
    Path("out_data").mkdir(exist_ok=True)

    csv_path = "data/pjm_2024_clean.csv"

    train_days = 180
    history_days = 7
    horizon_hours = 72

    train_end = pd.Timestamp("2024-10-31 23:00")

    print("Reading data...")
    y_all = load_hourly_series(csv_path)
    print("Total points:", len(y_all), "| Range:", y_all.index.min(), "->", y_all.index.max())

    if train_end > y_all.index.max():
        train_end = y_all.index.max()

    train_start = train_end - pd.Timedelta(days=train_days)
    y_train = y_all.loc[train_start:train_end].dropna()

    if len(y_train) < 24 * 30:
        raise ValueError(f"Not enough training data in the selected window: {len(y_train)} points")

    print("Training window:", y_train.index.min(), "->", y_train.index.max(), "| points:", len(y_train))

    print("Building model...")
    model = SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    print("Fitting model...")
    res = model.fit(disp=False, maxiter=80)
    print("Model trained.")

    print(f"Forecasting next {horizon_hours} hours...")
    forecast_values = res.get_forecast(steps=horizon_hours).predicted_mean
    forecast_index = make_forecast_index(last_time=y_train.index.max(), steps=horizon_hours, freq="h")
    forecast = pd.Series(forecast_values.values, index=forecast_index, name="forecast_mw")

    forecast.to_csv("out_data/forecast_sarimax_timeline.csv", header=True)
    print("Saved: out_data/forecast_sarimax_timeline.csv")

    split_time = y_train.index.max()

    hist_start = max(y_train.index.min(), split_time - pd.Timedelta(days=history_days))
    actual_hist = y_train.loc[hist_start:split_time]

    plot_timeline_actual_then_forecast(
        actual_hist=actual_hist,
        forecast=forecast,
        split_time=split_time,
        out_path="figs/actual_then_forecast_timeline.png",
        title="PJM — Actual (last 7 days) then Forecast (next 72 hours)",
        y_label="Load (MW)",
    )
    print("Saved: figs/actual_then_forecast_timeline.png")


if __name__ == "__main__":
    main()
