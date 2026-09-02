# %% Imports

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %% Configuration

DATA_PATH = Path("../data/Water_WHW.csv")

# Unix timestamp from the AMPds2 documentation for the meter replacement.
V100_UNIX_TS = 1342287780

# V100 meter installation / transition timestamp.
# Observations before this point use the old, coarser-resolution meter.
V100_START = pd.to_datetime(V100_UNIX_TS, unit="s")


# %% Load WHW data

df = pd.read_csv(DATA_PATH)

print(f"Shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst rows:")
display(df.head())

print("\nMissing values:")
print(df.isna().sum())


# %% Convert timestamp

df["datetime"] = pd.to_datetime(df["unix_ts"], unit="s")

# Keep chronological order.
df = df.sort_values("datetime").reset_index(drop=True)

# print(df[["unix_ts", "datetime"]].head())
# print()

print("\nFirst rows:")
display(df.head())

print()
print(f"Start: {df['datetime'].min()}")
print(f"End:   {df['datetime'].max()}")


# %% Basic timestamp checks

time_diff = df["datetime"].diff().dropna()
print("Most common sampling intervals:")
print(time_diff.value_counts().head())

print("\nDuplicate timestamps:", df["datetime"].duplicated().sum())
print("Rows:", len(df))

expected_rows = 730 * 24 * 60
print("Expected rows for 730 days:", expected_rows)


# %% Basic WHW target statistics

target = df["avg_rate"]

print(target.describe())

print("\nZero-consumption observations:")
print(f"{(target == 0).sum():,} / {len(target):,}")
print(f"{(target == 0).mean() * 100:.2f}%")

print("\nNon-zero observations:")
print(f"{(target > 0).sum():,}")


# %% Distribution of WHW avg_rate

plt.figure(figsize=(10, 5))

plt.hist(target, bins=100)

plt.xlabel("WHW avg_rate (L/min)")
plt.ylabel("Frequency")
plt.title("Distribution of Whole-House Water Consumption")

plt.tight_layout()
plt.show()


# %% Distribution excluding zero consumption

non_zero = target[target > 0]

plt.figure(figsize=(10, 5))

plt.hist(non_zero, bins=100)

plt.xlabel("WHW avg_rate (L/min)")
plt.ylabel("Frequency")
plt.title("Distribution of Non-Zero Whole-House Water Consumption")

plt.tight_layout()
plt.show()


# %% Time-series overview

plt.figure(figsize=(14, 5))

plt.plot(df["datetime"], df["avg_rate"], linewidth=0.5)

plt.xlabel("Date")
plt.ylabel("WHW avg_rate (L/min)")
plt.title("Whole-House Water Consumption Over Time")

plt.tight_layout()
plt.show()


# %% Example: one week of consumption

week_start = df["datetime"].min()
week_end = week_start + pd.Timedelta(days=7)

week = df[
    (df["datetime"] >= week_start)
    & (df["datetime"] < week_end)
]

plt.figure(figsize=(14, 5))

plt.plot(week["datetime"], week["avg_rate"], linewidth=0.8)

plt.xlabel("Date")
plt.ylabel("WHW avg_rate (L/min)")
plt.title("Example Week of Whole-House Water Consumption")

plt.tight_layout()
plt.show()


# %% Basic daily pattern

df["hour"] = df["datetime"].dt.hour

hourly_mean = df.groupby("hour")["avg_rate"].mean()

plt.figure(figsize=(10, 5))

plt.plot(hourly_mean.index, hourly_mean.values, marker="o")

plt.xlabel("Hour of day")
plt.ylabel("Mean WHW avg_rate (L/min)")
plt.title("Average Water Consumption by Hour of Day")
plt.xticks(range(24))

plt.tight_layout()
plt.show()


# %% Day-of-week pattern

df["day_of_week"] = df["datetime"].dt.dayofweek

dow_mean = df.groupby("day_of_week")["avg_rate"].mean()

plt.figure(figsize=(10, 5))

plt.plot(dow_mean.index, dow_mean.values, marker="o")

plt.xlabel("Day of week")
plt.ylabel("Mean WHW avg_rate (L/min)")
plt.title("Average Water Consumption by Day of Week")
plt.xticks(
    range(7),
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
)

plt.tight_layout()
plt.show()


# %% Inspect meter transition

print("V100 transition timestamp:")
print(V100_START)

print("\nRows before transition:")
print((df["datetime"] < V100_START).sum())

print("\nRows from V100 transition onward:")
print((df["datetime"] >= V100_START).sum())

print("\nFirst observations around transition:")

transition_window = df[
    (df["datetime"] >= V100_START - pd.Timedelta(minutes=5))
    & (df["datetime"] <= V100_START + pd.Timedelta(minutes=5))
]

display(
    transition_window[
        ["unix_ts", "datetime", "counter", "avg_rate", "inst_rate"]
    ]
)


# %% Compare old-meter and V100 periods

old_period = df[df["datetime"] < V100_START]
v100_period = df[df["datetime"] >= V100_START]

print("Old-meter period")
print("----------------")
print(f"Rows: {len(old_period):,}")
print(f"Mean avg_rate: {old_period['avg_rate'].mean():.4f}")
print(f"Zero proportion: {(old_period['avg_rate'] == 0).mean() * 100:.2f}%")

print("\nV100 period")
print("-----------")
print(f"Rows: {len(v100_period):,}")
print(f"Mean avg_rate: {v100_period['avg_rate'].mean():.4f}")
print(f"Zero proportion: {(v100_period['avg_rate'] == 0).mean() * 100:.2f}%")


# %% Keep only V100-meter data

df = df[df["datetime"] >= V100_START].copy()

df = df.reset_index(drop=True)

print(f"Post-transition shape: {df.shape}")
print(f"Start: {df['datetime'].min()}")
print(f"End:   {df['datetime'].max()}")


# %% Final post-transition target summary

print(df["avg_rate"].describe())

print("\nZero proportion:")
print(f"{(df['avg_rate'] == 0).mean() * 100:.2f}%")

print("\nMissing values:")
print(df[["datetime", "counter", "avg_rate", "inst_rate"]].isna().sum())


# %% Final post-transition overview

plt.figure(figsize=(14, 5))

plt.plot(df["datetime"], df["avg_rate"], linewidth=0.5)

plt.xlabel("Date")
plt.ylabel("WHW avg_rate (L/min)")
plt.title("Whole-House Water Consumption — V100 Period")

plt.tight_layout()
plt.show()
# %%
