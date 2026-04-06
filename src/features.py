import pandas as pd
import numpy as np

def cumulative_energy(backbone, arc):

    arc = arc.copy()
    arc["duration_s"] = (arc["Heating end"] - arc["Heating start"]).dt.total_seconds()
    arc["energy_MJ"] = arc["Active power"] * arc["duration_s"]

    results = []
    for key, group in backbone.groupby("key"):
        melt_arc = arc[arc["key"] == key]
        for _, row in group.iterrows():
            t = row["time"]

            prior = melt_arc[melt_arc["Heating end"] <= t]
            results.append(prior["energy_MJ"].sum())

    return pd.Series(results, index=backbone.index, name="cumulative_energy_MJ")

def time_since_last_heating(backbone, arc):
   
    results = []
    for key, group in backbone.groupby("key"):
        melt_arc = arc[arc["key"] == key]
        for _, row in group.iterrows():
            t = row["time"]
            prior = melt_arc[melt_arc["Heating end"] <= t]
            if len(prior) == 0:
                results.append(np.nan)
            else:
                last_end = prior["Heating end"].max()
                delta = (t - last_end).total_seconds()
                results.append(delta)

    return pd.Series(results, index=backbone.index, name="time_since_heating_s")

def cumulative_bulk(backbone, bulk, bulk_time):
    """
    For each temperature measurement, compute the cumulative kg of each
    bulk material added before the measurement timestamp.

    Returns a DataFrame with columns Bulk_1_cum through Bulk_15_cum,
    plus total_bulk_cum (sum of all materials).
    """
    bulk_cols = [f"Bulk {i}" for i in range(1, 16)]
    result_cols = [f"Bulk_{i}_cum" for i in range(1, 16)]

    results = pd.DataFrame(0.0, index=backbone.index, columns=result_cols)

    for key, group in backbone.groupby("key"):
        melt_bulk = bulk[bulk["key"] == key]
        melt_bulk_time_row = bulk_time[bulk_time["key"] == key]

        if len(melt_bulk) == 0 or len(melt_bulk_time_row) == 0:
            continue

        masses = melt_bulk[bulk_cols].values[0]
        timestamps = melt_bulk_time_row[bulk_cols].values[0]

        for idx, row in group.iterrows():
            t = row["time"]
            cumulative = np.zeros(15)
            for i in range(15):
                if pd.notna(timestamps[i]) and pd.Timestamp(timestamps[i]) <= t:
                    cumulative[i] = masses[i] if pd.notna(masses[i]) else 0.0
            results.loc[idx] = cumulative

    results["total_bulk_cum"] = results[result_cols].sum(axis=1)
    return results

def cumulative_wire(backbone, wire, wire_time):
    """
    For each temperature measurement, compute the cumulative kg of each
    wire material added before the measurement timestamp.

    Returns a dataframe with columns Wire_1_cum through Wire_9_cum,
    plus total_wire_cum (sum of all materials).
    """
    wire_cols = [f"Wire {i}" for i in range(1, 10)]
    result_cols = [f"Wire_{i}_cum" for i in range(1, 10)]

    results = pd.DataFrame(0.0, index=backbone.index, columns=result_cols)

    for key, group in backbone.groupby("key"):
        melt_wire = wire[wire["key"] == key]
        melt_wire_time_row = wire_time[wire_time["key"] == key]

        if len(melt_wire) == 0 or len(melt_wire_time_row) == 0:
            continue

        masses = melt_wire[wire_cols].values[0]
        timestamps = melt_wire_time_row[wire_cols].values[0]

        for idx, row in group.iterrows():
            t = row["time"]
            cumulative = np.zeros(9)
            for i in range(9):
                if pd.notna(timestamps[i]) and pd.Timestamp(timestamps[i]) <= t:
                    cumulative[i] = masses[i] if pd.notna(masses[i]) else 0.0
            results.loc[idx] = cumulative

    results["total_wire_cum"] = results[result_cols].sum(axis=1)
    return results

def previous_temperature(backbone):
    """
    For each measurement within a heat, return the most recent prior
    temperature reading. NaN for the first measurement in each heat.
    """
    result = backbone.groupby("key")["Temperature"].shift(1)
    result.name = "prev_temp"
    return result


def measurement_index(backbone):
    """
    Ordinal position of each measurement within its heat (1-indexed).
    Captures process stage: early measurements occur before significant
    heating, later ones after multiple heating/addition cycles.
    """
    result = backbone.groupby("key").cumcount() + 1
    result.name = "measurement_index"
    return result