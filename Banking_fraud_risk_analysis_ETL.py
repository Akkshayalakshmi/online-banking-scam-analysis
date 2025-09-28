# banking_fraud_risk_analysis.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import timedelta, datetime
import os

# --------- CONFIG / THRESHOLDS (tweak these) ----------
INPUT_CSV = "Online Banking Scam .csv"  # adjust if needed
OUTPUT_DIR = "."  # output folder
SMALL_AMOUNT_THRESHOLD = 200.0  # small transfer amount for smurfing
SMURF_COUNT_THRESHOLD = 5  # >= this many small txns in window -> smurf
SMURF_SUM_THRESHOLD = 500.0  # and total sum threshold
SMURF_WINDOW_HOURS = 24  # time window for smurfing (hours)

MULE_WINDOW_DAYS = 30  # window for money mule detection
MULE_UNIQUE_COUNTERPARTY = 10  # unique merchant/counterparty threshold

LARGE_TRANSFER_QUANTILE = 0.999  # quantile for extreme transfers
LARGE_TRANSFER_MULTIPLIER = 10.0  # or > multiplier of account avg

ODD_HOUR_START = 0  # midnight
ODD_HOUR_END = 5  # 5 AM inclusive

LOGIN_ATTEMPT_THRESHOLD = 3  # > this considered suspicious

IF_CONTAMINATION = 0.02  # isolation forest contamination
IF_N_ESTIMATORS = 200

# Feature list used for IsolationForest (add/remove as needed)
IF_FEATURES = [
    "TransactionAmount",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
]

# Rule weights for final aggregation (tweak for business priorities)
WEIGHTS = {
    "SmurfFlag": 2.0,
    "MuleFlag": 2.5,
    "AccountTakeoverFlag": 3.0,
    "LargeTransferFlag": 3.0,
    "OddHourFlag": 1.0,
    "LoginFraudFlag": 2.0,
    "IForestFlag": 2.0,
}
# -------------------------------------------------------


def load_data(path):
    df = pd.read_csv(
        path,
        parse_dates=["TransactionDate", "PreviousTransactionDate"],
        infer_datetime_format=True,
    )
    return df


def add_basic_fields(df):
    # normalize column names if needed (strip)
    df.columns = [c.strip() for c in df.columns]
    # ensure TransactionDate exists
    if "TransactionDate" not in df.columns:
        raise KeyError("TransactionDate column not found")
    df = df.sort_values(["AccountID", "TransactionDate"]).reset_index(drop=True)
    df["TxnHour"] = df["TransactionDate"].dt.hour
    return df


def rule_smurfing(df):
    # For each account, find within rolling SMURF_WINDOW_HOURS windows counts of small transactions (Debit)
    df["IsSmallTxn"] = (df["TransactionAmount"] <= SMALL_AMOUNT_THRESHOLD).astype(int)
    # We'll compute for each transaction: number and total of small txns for the account in past window
    df["SmurfCount24h"] = 0
    df["SmurfSum24h"] = 0.0

    # Using efficient groupby + rolling via timestamps:
    for acc, g in df.groupby("AccountID"):
        times = g["TransactionDate"].values
        amounts = g["TransactionAmount"].values
        is_small = g["IsSmallTxn"].values
        idxs = g.index.values

        start = 0
        n = len(times)
        for i in range(n):
            # advance start until within window
            while start < i and (times[i] - times[start]) > np.timedelta64(
                int(SMURF_WINDOW_HOURS * 3600), "s"
            ):
                start += 1
            window_idxs = range(start, i + 1)
            count_small = int(is_small[window_idxs].sum())
            sum_small = float((amounts[window_idxs] * is_small[window_idxs]).sum())
            df.at[idxs[i], "SmurfCount24h"] = count_small
            df.at[idxs[i], "SmurfSum24h"] = sum_small

    df["SmurfFlag"] = (
        (df["SmurfCount24h"] >= SMURF_COUNT_THRESHOLD)
        & (df["SmurfSum24h"] >= SMURF_SUM_THRESHOLD)
    ).astype(int)
    return df


def rule_money_mule(df):
    # count unique counterparties (MerchantID) per account in last MULE_WINDOW_DAYS days
    df["MuleUniqueCounterparty30d"] = 0
    window = pd.Timedelta(days=MULE_WINDOW_DAYS)
    for acc, g in df.groupby("AccountID"):
        times = g["TransactionDate"].values
        merchants = g["MerchantID"].values
        idxs = g.index.values
        start = 0
        n = len(times)
        for i in range(n):
            while start < i and (times[i] - times[start]) > np.timedelta64(
                int(window.total_seconds()), "s"
            ):
                start += 1
            unique_cnt = len(set(merchants[start : i + 1]))
            df.at[idxs[i], "MuleUniqueCounterparty30d"] = unique_cnt
    df["MuleFlag"] = (
        df["MuleUniqueCounterparty30d"] >= MULE_UNIQUE_COUNTERPARTY
    ).astype(int)
    return df


def rule_account_takeover(df):
    # new device/IP for account = first time seeing DeviceID or IP for that account
    df["IsNewDeviceFlag"] = (
        ~df.duplicated(subset=["AccountID", "DeviceID"], keep="first")
    ).astype(int)
    df["IsNewIPFlag"] = (
        ~df.duplicated(subset=["AccountID", "IP Address"], keep="first")
    ).astype(int)
    # combine with login anomalies if available
    df["AccountTakeoverFlag"] = (
        (df["IsNewDeviceFlag"] == 1) | (df["IsNewIPFlag"] == 1)
    ).astype(int)
    # if AuthResult exists, we can check failed attempts immediately before a transaction (advanced)
    if "AuthResult" in df.columns:
        # lower-case normalize
        df["AuthResult"] = df["AuthResult"].astype(str).str.lower()
        df["FailedBeforeSuccessFlag"] = 0
        for acc, g in df.groupby("AccountID"):
            auths = g[["TransactionDate", "AuthResult"]].sort_values("TransactionDate")
            idxs = auths.index.tolist()
            # sliding check: if a 'success' is preceded by >=3 'fail' within 10 minutes
            for i in range(len(idxs)):
                if auths.at[idxs[i], "AuthResult"] == "success":
                    # look back up to 10 minutes
                    t0 = auths.at[idxs[i], "TransactionDate"]
                    window_mask = (
                        auths["TransactionDate"] >= (t0 - pd.Timedelta(minutes=10))
                    ) & (auths["TransactionDate"] < t0)
                    fails = (auths.loc[window_mask, "AuthResult"] == "fail").sum()
                    if fails >= LOGIN_ATTEMPT_THRESHOLD:
                        df.at[idxs[i], "FailedBeforeSuccessFlag"] = 1
        df["AccountTakeoverFlag"] = (
            (df["AccountTakeoverFlag"] == 1)
            | (df.get("FailedBeforeSuccessFlag", 0) == 1)
        ).astype(int)
    return df


def rule_large_transfer(df):
    # large by distribution quantile OR by multiple of account average
    q = df["TransactionAmount"].quantile(LARGE_TRANSFER_QUANTILE)
    df["LargeByQuantile"] = (df["TransactionAmount"] >= q).astype(int)
    # account average
    acct_avg = df.groupby("AccountID")["TransactionAmount"].transform("mean").fillna(0)
    df["LargeByAvgMult"] = (
        df["TransactionAmount"] >= (acct_avg * LARGE_TRANSFER_MULTIPLIER)
    ).astype(int)
    df["LargeTransferFlag"] = (
        (df["LargeByQuantile"] == 1) | (df["LargeByAvgMult"] == 1)
    ).astype(int)
    return df


def rule_odd_hours(df):
    df["OddHourFlag"] = df["TxnHour"].apply(
        lambda h: 1 if (h >= ODD_HOUR_START and h <= ODD_HOUR_END) else 0
    )
    return df


def rule_login_fraud(df):
    # basic: LoginAttempts > threshold
    df["LoginFraudFlag"] = (df["LoginAttempts"] > LOGIN_ATTEMPT_THRESHOLD).astype(int)
    # If AuthResult exists, detect many failed logins followed by successful login immediately before txn
    if "AuthResult" in df.columns:
        df["AuthResult"] = df["AuthResult"].astype(str).str.lower()
        df["FailedThenSuccessFlag"] = 0
        # We attempt to detect failed attempts (as separate auth events) -- only helpful if dataset has auth events
        # If dataset only has LoginAttempts per transaction, we keep simple flag above.
        # (Implementation left minimal because dataset likely doesn't have separate auth logs.)
    return df


def compute_isolation_forest(df):
    # ensure features exist
    X = df.copy()
    for f in IF_FEATURES:
        if f not in X.columns:
            X[f] = 0.0
    Xf = X[IF_FEATURES].fillna(0)
    iso = IsolationForest(
        n_estimators=IF_N_ESTIMATORS, contamination=IF_CONTAMINATION, random_state=42
    )
    iso_pred = iso.fit_predict(Xf)  # -1 anomaly, 1 normal
    df["IForestFlag"] = (iso_pred == -1).astype(int)
    # optionally store anomaly score
    if hasattr(iso, "decision_function"):
        df["IForestScore"] = -iso.decision_function(Xf)  # higher means more anomalous
    else:
        df["IForestScore"] = 0.0
    return df


def combine_scores(df):
    # Weighted sum using WEIGHTS
    df["FinalRiskScore"] = 0.0
    for flag, w in WEIGHTS.items():
        if flag in df.columns:
            df["FinalRiskScore"] += df[flag] * w
    # Also include RuleRiskScore (unweighted total of core flags) for transparency
    core_flags = [
        "SmurfFlag",
        "MuleFlag",
        "AccountTakeoverFlag",
        "LargeTransferFlag",
        "OddHourFlag",
        "LoginFraudFlag",
    ]
    df["RuleRiskScore"] = df[[c for c in core_flags if c in df.columns]].sum(axis=1)
    return df


def save_output(df):
    outpath = "OUTPUT_FILE_PATH"
    df.to_csv(outpath, index=False)
    print(f"✅ Risk scoring completed. File saved as: {outpath}")
    return outpath


def main():
    print("Loading data...")
    df = load_data(INPUT_CSV)
    df = add_basic_fields(df)

    print("Applying smurfing rule...")
    df = rule_smurfing(df)

    print("Applying money mule rule...")
    df = rule_money_mule(df)

    print("Applying account takeover rule...")
    df = rule_account_takeover(df)

    print("Applying large transfer rule...")
    df = rule_large_transfer(df)

    print("Applying odd hours rule...")
    df = rule_odd_hours(df)

    print("Applying login fraud rule...")
    df = rule_login_fraud(df)

    print("Running Isolation Forest...")
    df = compute_isolation_forest(df)

    print("Combining scores...")
    df = combine_scores(df)

    print("Saving results...")
    save_output(df)
    # Print a sample of suspicious transactions
    suspicious = df[df["FinalRiskScore"] >= 4].sort_values(
        "FinalRiskScore", ascending=False
    )
    print("\nTop suspicious transactions (FinalRiskScore >= 4):")
    print(
        suspicious[
            [
                "TransactionID",
                "AccountID",
                "TransactionAmount",
                "TxnHour",
                "RuleRiskScore",
                "IForestFlag",
                "FinalRiskScore",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
