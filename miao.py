import math
import yaml
import os
from typing import List, Tuple, Dict, Union, Any

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from fracdiff import fdiff
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.svar_model import SVAR

def parse_miao_db_yml(path: str):
    with open(path, 'r') as f:
        miao_db = yaml.safe_load(f)
    return miao_db

def unit_root_test(series: pd.Series):
    results = adfuller(series, regression='c')
    adf = results[0]
    pvalue = results[1]
    
    return adf, pvalue

def to_stationary_process_recursive(s: pd.Series, retry_count = 0, init_state = None):
    significance_level = 0.05
    n = 0.1 * retry_count

    adf, p = unit_root_test(s)

    if init_state is None:
        init_state = (adf, p)

    if p >= significance_level:
        s_ = fdiff(s, n)

        adf, p = unit_root_test(s_)
        if p >= significance_level:
            return to_stationary_process_recursive(s, retry_count + 1, init_state)
        else:
            return init_state[0], init_state[1], adf, p, n, s_
    
    return init_state[0], init_state[1], None , None, n, s

def split_dates(start, end, years):
    dates = []
    while start.year < end.year:
        next_year = start.year + years
        next_year = min(next_year, end.year)
        next_year = pd.Timestamp(next_year, start.month, start.day)
        dates.append({
            "start": start,
            "end": next_year
        })
        start = next_year
    
    return dates

def get_unique_idx(db):
    name = db["name"]
    osss = [name] + [c["name"] for c in db["competitors"]]
    return gen_idx_(osss)

def gen_idx_(oss_names):
    return ",".join([oss.split("/")[1] for oss in oss_names])

class AutoDetectTmsError(Exception):
    """
    AutoDetectTmsError is an exception class for errors that occur during the auto detection of Tms.
    E0001: No overlap between start and end dates
    E0002: No data after start date
    """
    code: str 

    def __init__(self, code):
        self.code = code

    def __str__(self):
        return f"AutoDetectTmsError: {self.code}"

def auto_detect_tms_(db, N_MONTHS = 12, limit_y = 4, early_stopping_month = 0, daily_dataset_path = "./datasets/original/daily", monthly_dataset_path = "./datasets/original/monthly"):
    anual_days = 365.25

    start_dates = []
    end_dates = []
    target_df = None

    target = {"name": db["name"]}
    if "appearance_time" in db:
        target["appearance_time"] = db["appearance_time"]

    data = [target] + db["competitors"]

    for row in data:
        repo = row["name"]
        df_monthly = pd.read_csv(f"{monthly_dataset_path}/{repo}.csv", index_col=0)
        df_monthly.index = [pd.Timestamp(idx) for idx in df_monthly.index]

        df_daily = pd.read_csv(f"{daily_dataset_path}/{repo}.csv", index_col=0)
        df_daily.index = [pd.Timestamp(idx) for idx in df_daily.index]

        if target_df is None:
            target_df = df_monthly

        start_date = pd.Timestamp(row["appearance_time"]) if "appearance_time" in row else df_daily.index[0]
        start_dates.append(start_date)
        end_dates.append(df_daily.index[-1])

    start_dates = list(sorted(start_dates, key=lambda d: d.timestamp(), reverse=True))
    start_date: pd.Timestamp = start_dates[0]

    end_dates_min = np.min(end_dates)
    
    # 重複している時間がない
    if end_dates_min < start_date:
        raise AutoDetectTmsError("E0001")

    # if db["rev"] == True:
    df_after_start = target_df[target_df.index > pd.Timestamp(start_date.year, start_date.month, start_date.day)]
    if len(df_after_start.n_commits) == 0:
        raise AutoDetectTmsError("E0002")

    DORMANT_THRESHOLD = 1.5
    end_date = None

    target_df_from_start = target_df[start_date:]

    if "end_date" in db:
        tm_end = pd.Timestamp(db["end_date"])
    else:
        if db["rev"] == 1:
            rolling_commits = target_df_from_start['n_commits'].rolling(window='365D', min_periods=1).mean()
            decline_dates = rolling_commits[rolling_commits <= DORMANT_THRESHOLD].index
            end_date = decline_dates.min()
            
            if pd.isnull(end_date):
                end_date = end_dates_min

            if end_date > end_dates_min:
                end_date = end_dates_min

            if (end_date - start_date).days < anual_days:
                raise AutoDetectTmsError("E0003")

            tm_end = pd.Timestamp(end_date.year, end_date.month, end_date.day)
        else:
            tm_end = end_dates_min

    tm_start = pd.Timestamp(start_date.year, start_date.month, start_date.day)

    num_y = math.floor((tm_end - tm_start).days/anual_days)
    remaining_days = (tm_end - tm_start).days - (limit_y*anual_days)
    diff = remaining_days/anual_days
    should_split = diff >= 0.5 or num_y > limit_y

    tms = []
    if should_split:
        tms = split_dates(tm_start, tm_end, limit_y)
        if (tms[-1]["end"] - tms[-1]["start"]).days < anual_days:
            del tms[-1]
    else:
        y = (tm_end - tm_start).days/anual_days
        if y <= limit_y: 
            tms = [{
                "start": tm_start,
                "end": tm_end
            }]
        else:
            tm_start = tm_end - pd.DateOffset(years=limit_y)
            tms = [{
                "start": tm_start,
                "end": tm_end
            }]

    return tms

def auto_detect_tms(db,
                         N_MONTHS: int = 12,
                         limit_y: int = 4,
                         min_days_in_a_tm: int = 365,
                         early_stopping_months: Union[int, None] = None,
                         daily_dataset_path: str = "./datasets/original/daily",
                         monthly_dataset_path: str = "./datasets/original/monthly") -> List[Dict[str, pd.Timestamp]]:
    """Detect analysis periods while excluding long dormant spans in the target series."""

    target_repo = db["name"]
    target_df = pd.read_csv(f"{daily_dataset_path}/{target_repo}.csv", index_col=0)
    
    if target_df.n_commits.sum() < 500:
        raise AutoDetectTmsError("E0005")

    base_tms = auto_detect_tms_(db,
                               N_MONTHS=N_MONTHS,
                               limit_y=limit_y,
                               daily_dataset_path=daily_dataset_path,
                               monthly_dataset_path=monthly_dataset_path)

    target_df.index = [pd.Timestamp(idx) for idx in target_df.index]
    target_series = target_df.n_commits.astype(float)

    refined_tms: List[Dict[str, pd.Timestamp]] = []

    for tm in base_tms:
        start = tm["start"]
        end = tm["end"]
        window_series = target_series[start:end]
        zero_ratio = len(window_series[window_series == 0])/len(window_series)
        if zero_ratio > 0.99:
            continue
        
        refined_tms.append({"start": start, "end": end})

    if len(refined_tms) > 0 and early_stopping_months is not None:
        last_start = refined_tms[-1]["start"]
        last_end_before_early_stopping = refined_tms[-1]["end"] - pd.DateOffset(months=early_stopping_months)
        if (last_end_before_early_stopping - last_start).days >= min_days_in_a_tm:
            refined_tms[-1]["end"] = last_end_before_early_stopping
        else:
            del refined_tms[-1]

    if len(refined_tms) == 0:
        raise AutoDetectTmsError("E0004")

    return refined_tms

def create_Tms(start, end, interval, split_interval = False):
    current = start
    result = []
    
    while current < end:
        next_year = current + pd.DateOffset(months=interval)
        next_year_end = min(next_year, end)
        if split_interval:
            result.append([current, next_year_end - pd.DateOffset(days=1)])
        else:
            end_ = next_year_end - pd.DateOffset(days=1)

            if len(result) >= 1:
                if (end_ - result[-1][1]).days == 1:
                    break

            result.append([start, end_])
        current = next_year_end

    return result

def subframe(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp, date_col: str = 'ym', expand_end_date: bool = False):
    start_idx = df[df[date_col] == start_date].index[0]

    try:
        end_idx = df[df[date_col] == end_date].index[0]
    except Exception as e:
        if expand_end_date:
            last_date = df[date_col].iloc[-1]
            diff = (end_date - last_date).days

            other_cols = df.columns.difference([date_col])

            additional_data = {}
            for col in other_cols:
                additional_data[col] = pd.NA

            additional_data[date_col] = []
            for i in range(diff):
                last_date += pd.DateOffset(days=1)
                additional_data[date_col].append(last_date)

            additional_df = pd.DataFrame(additional_data)
            df = pd.concat([df, additional_df], ignore_index=True)
            end_idx = df[df[date_col] == end_date].index[0]
        else:
            raise e

    sub_df = df.iloc[start_idx:end_idx+1].reset_index(drop=True)
    sub_df.attrs['idx'] = {
        'start': start_idx,
        'end': end_idx
    }
    sub_df.attrs['date'] = {
        'start': start_date,
        'end': end_date        
    }
    return sub_df

def find_optimal_lag(X: pd.DataFrame, maxlag: int, ic: str):
    model = VAR(X)

    order_selection = model.select_order(maxlag)
    ic_table = {name: values for name, values in order_selection.ics.items()}
    if ic not in ic_table:
        raise ValueError(f"Unknown information criterion: {ic}")

    significance_level = 0.1

    candidates = {}
    for lag in range(1, maxlag + 1):
        try:
            results = model.fit(lag)
            results.irf(10)
        except:
            continue

        ic_value = ic_table[ic][lag]
        if np.isnan(ic_value):
            continue

        r, whiteness_test_lag = whiteness_test(model, lag)

        ic_map = {}
        for name, values in ic_table.items():
            ic_map[name] = values[lag]

        candidates[lag] = (lag, ic_map, r, whiteness_test_lag)

    if not candidates:
        best_lag = np.argsort(ic_table[ic])[0]
        ic_map = {}
        for name, values in ic_table.items():
            ic_map[name] = values[best_lag]

        return best_lag, ic_map, candidates[best_lag][2], candidates[best_lag][3]

    # Pareto Optimal Lag searching
    candidate_items = list(candidates.values())

    def is_dominated(a, b):
        a_ic = a[1][ic]
        b_ic = b[1][ic]
        a_p = a[2].pvalue
        b_p = b[2].pvalue
        better_or_equal = (b_ic <= a_ic) and (b_p >= a_p)
        strictly_better = (b_ic < a_ic) or (b_p > a_p)
        return better_or_equal and strictly_better

    pareto_items = []
    for item in candidate_items:
        dominated = False
        for other in candidate_items:
            if item is other:
                continue
            if is_dominated(item, other):
                dominated = True
                break
        if not dominated:
            pareto_items.append(item)

    if not pareto_items:
        pareto_items = candidate_items

    preferred = [item for item in pareto_items if item[2].pvalue >= significance_level]
    if not preferred:
        preferred = pareto_items

    best_item = max(preferred, key=lambda item: (item[2].pvalue, -item[1][ic]))
    return best_item

def prepare_var_(data: List[Tuple[str, pd.Series]], maxlag, ic="aic", verbose = True):
    var_data = {}
    adf_test_results = {}
    for name, X, dates in data:
        X = X.fillna(0)
        adf, p, fracdif_adf, fracdif_p, n, X = to_stationary_process_recursive(X)
        var_data[name] = X
        adf_test_results[name] = (adf, p, fracdif_adf, fracdif_p, n)
        
    VAR_X = pd.DataFrame(var_data).fillna(0)
    VAR_X.index = dates
    VAR_X = VAR_X.asfreq('D')
    
    lag, ics, var_whiteness_test_result, var_whiteness_test_lag = find_optimal_lag(VAR_X, maxlag, ic)

    max_rp = var_whiteness_test_result.pvalue

    adf_ps = [val[3] if val[3] is not None else val[1] for val in adf_test_results.values()]
    adf_p = ", ".join(['{:.3f}'.format(p) for p in adf_ps])

    if verbose:
        if max_rp > 0.1:
            print(f"Failed to reject the H0 of whiteness test: pvalue={max_rp}, lag={lag}, adf_p={adf_p}")
        else:
            print(f"Rejected the H0 of whiteness test: pvalue={max_rp}, lag={lag}, adf_p={adf_p}")

    VAR_X.attrs["whiteness_test_result"] = var_whiteness_test_result
    VAR_X.attrs["whiteness_test_lag"] =  var_whiteness_test_lag
    VAR_X.attrs["ics"] = ics
    VAR_X.attrs["adf_test_results"] = adf_test_results

    return lag, VAR_X

def prepare_var(tms, As, maxlag, verbose = True):
    VAR_Xs = []

    for tm in tms:
        data = []
        for name, A in As:
            start, end = tm

            A_ = pd.DataFrame({
                "date": [pd.Timestamp(idx) for idx in A.n_commits.index],
                "n_commits": A.n_commits.values
            })
            A_ = subframe(A_, start, end, date_col="date", expand_end_date=True)
            # A_.n_commits = weak_seasonal_adjusment(A_.n_commits.fillna(0))
            data.append((name, A_.n_commits.fillna(0), A_.date.values))

        try:            
            lag, VAR_X = prepare_var_(data, maxlag=maxlag, ic="aic", verbose=verbose)
            VAR_X.index = A_.date.values
            VAR_Xs.append((lag, VAR_X))
        except Exception as e:
            names = [d[0] for d in data]
            print(f"[Error] {e}", names)
            VAR_X_empty = pd.DataFrame({r:[] for r in names})
            VAR_X_empty.attrs["error"] = e
            VAR_X_empty.attrs["data"] = data
            VAR_Xs.append((None, VAR_X_empty))

    return VAR_Xs

def whiteness_test(model, lag):
    if type(model) == SVAR:
        results = model.fit(maxlags=lag)
    else:
        results = model.fit(lag)
    whiteness_lags = 10
    if lag >= whiteness_lags:
        whiteness_lags = lag*2

    r = results.test_whiteness(nlags=whiteness_lags, signif=0.1)

    return r, whiteness_lags

def miao_score(sces: List[float]) -> float:
    """
    Calculate MS_{ij} from list of SCE
    """
    return (-1)*np.sum(sces)

def miao_phase1_with_period_shifts(
    A_mat: np.ndarray,
    shifts: List[str],
    Tms: List[str],
    group_with_sep: str,
    permutation_id: int,
    orth: bool = True,
    dataset_path: str = "./",
    irf_k: int = 200,
    select_best_shift: bool = True,
) -> List[Dict[str, List[float]]]:
    """
    This function performs a MIAO Phase 1 and outputs the SCE (Shock Cumulative Effect) for $A_i -> A_j$.
    Note that PREPARE_VAR has already been completed, and the data preprocessing and optimal lag orders use pre-calculated results.

    When ``select_best_shift`` is True (default), diagnostics per period shift are
    evaluated and only the most stable shift is retained to suppress noisy
    windows. Set it to False to recover the previous behaviour of returning all
    shifts for simple averaging.
    """

    def generate_irf_pair(labels: List[str]) -> Dict[str, List[str]]:
        result = {}

        for i in range(len(labels)):
            for j in range(len(labels)):
                key = "{}x{}".format(i, j)
                value = [labels[j], labels[i]]
                result[key] = value

        return result

    shift_summaries = []
    var_result = pd.read_csv(f"{dataset_path}/var_estimation_results/aic/permute_{permutation_id}.csv")

    for shift_idx, shift in enumerate(shifts):
        sces = {}
        normalized_sces = {}
        mad_normalized_sces = {}
        uncertainty_matrix = {}
        granger_causality = {}
        shift_structural_covs = []
        shift_is_stables = []
        spectral_radii = []
        resid_traces = []
        
        for Tm in Tms:
            csv_path = f"{dataset_path}/datasets/preprocessed/permute_{permutation_id}/{shift}/{Tm}/{group_with_sep}.csv"
            
            if not os.path.exists(csv_path):
                # print(f"[Error not found] permutation_id: {permutation_id}, shift: {shift}, Tm: {Tm}, group_with_sep: {group_with_sep}")
                continue
            X = pd.read_csv(csv_path, index_col=0)
            X.index = [pd.Timestamp(idx) for idx in X.index]
            pair = generate_irf_pair(X.columns)

            cond = (var_result["group"] == group_with_sep) & (var_result["Tm"] == Tm) & (var_result["period_shift"] == shift)
            if not cond.any():
                # print(f"No valid var_result found for {group_with_sep}, {Tm}, {shift}")
                continue

            lag = var_result[cond].iloc[0].lag

            model = SVAR(X, svar_type='A', A=A_mat)
            # model = VAR(X)
            if type(model) == SVAR:
                results = model.fit(maxlags=lag)
            else:
                results = model.fit(lag)

            irf = results.irf(irf_k)

            #Checks if det(I - Az) = 0 for any mod(z) <= 1, so all the eigenvalues of the companion matrix must lie outside the unit circle
            is_stable = results.is_stable()
            shift_is_stables.append(bool(is_stable))

            roots = getattr(results, "roots", [])
            if isinstance(roots, (list, tuple)):
                roots = np.array(roots)
            if np.size(roots) > 0:
                spectral_radii.append(float(np.max(np.abs(roots))))
            else:
                spectral_radii.append(float("inf"))

            sigma_u = getattr(results, "sigma_u", None)
            if sigma_u is not None:
                resid_traces.append(float(np.trace(sigma_u)))
            else:
                resid_traces.append(float("inf"))

            n = len(X.columns)

            for from_ in range(n):
                for to_ in range(n):
                    if from_ == to_:
                        continue

                    left, right = pair[f"{from_}x{to_}"]
                    label = f"{left} -> {right}"
                    if label not in sces:
                        sces[label] = []

                    if label not in uncertainty_matrix:
                        uncertainty_matrix[label] = []

                    cum_effects = None

                    if type(model) == SVAR:
                        cum_effects = irf.svar_cum_effects
                    else:
                        if orth:
                            cum_effects = irf.orth_cum_effects
                        else:
                            cum_effects = irf.cum_effects

                    sce = cum_effects[-1, from_, to_]
                    sces[label].append(sce)

                    _, right = label.split(" -> ")
                    s = X[right.strip()]

                    sigma = s.std()
                    normalized_sces[label] = sce / sigma

                    mad = np.median(np.abs(s - s.median()))
                    mad = 1e-3 if mad == 0 else mad
                    mad_normalized_sces[label] = sce / mad
    
        shift_summary = {
            "name": shift,
            "order": shift_idx,
            "sces": sces,
            "normalized_sces": normalized_sces,
            "mad_normalized_sces": mad_normalized_sces,
            "uncertainty_matrix": uncertainty_matrix,
            "granger_causality": granger_causality,
            "is_stables": shift_is_stables,
            "spectral_radius": float(np.max(spectral_radii)) if spectral_radii else float("inf"),
            "stable_ratio": float(np.mean(shift_is_stables)) if shift_is_stables else 0.0,
            "resid_trace": float(np.mean(resid_traces)) if resid_traces else float("inf"),
            "structural_covs": shift_structural_covs,
        }
        shift_summaries.append(shift_summary)

    return shift_summaries

def miao_phase2_with_period_shifts(shift_summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Calculate AMS_{ij} and construct the MIAO score table with optional uncertainty-based weighting.

    Parameters:
    -----------
    shift_summaries: List[Dict[str, Any]]
        Shift summaries for each shift

    Returns:
    --------
    pd.DataFrame
        MIAO score table with AMS_ij values
    """

    ams = {}
    normalized_ams = {}

    sces_with_shifts = [summary["sces"] for summary in shift_summaries]
    normalized_sces_with_shifts = [summary["normalized_sces"] for summary in shift_summaries]
    # mad_normalized_sces_with_shifts = [summary["mad_normalized_sces"] for summary in shift_summaries]

    ziped = zip(sces_with_shifts, normalized_sces_with_shifts)

    for sces, normalized_sces in ziped:
        for key in sces:
            if key not in ams:
                ams[key] = 0
                normalized_ams[key] = 0
                # mad_normalized_ams[key] = 0
            
            ams[key] += miao_score(sces[key])
            normalized_ams[key] += miao_score(normalized_sces[key])
            # mad_normalized_ams[key] += miao_score(mad_normalized_sces[key])

    table = pd.DataFrame({
        "AMS_ij": [v/len(sces_with_shifts) for v in ams.values()],
        "normalized_AMS_ij": [v/len(sces_with_shifts)*100 for v in normalized_ams.values()],
        # "mad_normalized_AMS_ij": [v/len(sces_with_shifts) for v in mad_normalized_ams.values()],
    }, index=ams.keys())

    return table

def to_symbol(db):
    t = db["name"]
    c1 = db["competitors"][0]["name"]
    c2 = db["competitors"][1]["name"]
    
    return t, c1, c2

def create_decision_tree_df(dataset_yml, score_dfs, normalize = True, column = "AMS_ij"):
    data = {
        "label": [],
        "group_id": [],
        "split_id": [],
        "group": [],
        "target": []
    }

    for df in score_dfs:
        t = df.attrs["target"]
        db = list(filter(lambda db: get_unique_idx(db) == t, dataset_yml))[0]
        t, c1, c2 = to_symbol(db)
        df_ = df.copy()

        if normalize:
            df_[column] = df_[column]/df.attrs["n_Tm"]

        df_ = df_.rename(index={
            f"{c1} -> {t}":  "c1 -> t",
            f"{c2} -> {t}":  "c2 -> t",
            f"{c2} -> {c1}":  "c2 -> c1",
            f"{c1} -> {c2}":  "c1 -> c2",
            f"{t} -> {c1}":  "t -> c1",
            f"{t} -> {c2}":  "t -> c2",
        })

        for idx, score in zip(df_.index, df_[column]):
            if idx not in data:
                data[idx] = []

            data[idx].append(score)
        
        data["label"].append(1 if db["rev"] else 0)

        try:
            group_id, split_id = df.attrs["group"].split("_")
            data["group_id"].append(int(group_id))
            data["split_id"].append(int(split_id))
        except:
            data["group_id"].append(int(df.attrs["group"]))
            data["split_id"].append(0)

        data["group"].append(df.attrs["group"])
        data["target"].append(get_unique_idx(db))

    return pd.DataFrame(data)

def calculate_overlap(period_a, period_b):
    """2つの期間の重複時間を計算する関数"""
    # 重複部分の開始時刻は、それぞれの開始時刻の遅い方
    overlap_start = max(period_a['__start__'], period_b['__start__'])
    
    # 重複部分の終了時刻は、それぞれの終了時刻の早い方
    overlap_end = min(period_a['__end__'], period_b['__end__'])
    
    # 重複期間の長さを計算 (重複がない場合は0)
    overlap_duration = max(0, overlap_end - overlap_start)
    
    return overlap_duration

def sort_periods_by_overlap(base_period, periods_list):
    """
    基準期間との重複期間が長い順にリストをソートする関数
    
    :param base_period: 基準となる期間 {'start': start0, 'end': end0}
    :param periods_list: ソート対象の期間のリスト [{'start':...}, {'end':...}]
    :return: ソートされた新しいリスト
    """
    # sorted関数のkeyに、各要素とbase_periodとの重複期間を計算するラムダ式を指定
    # reverse=Trueで降順（長い順）にソート
    sorted_list = sorted(
        periods_list, 
        key=lambda p: calculate_overlap(base_period, p), 
        reverse=True
    )
    return sorted_list

def competitors_sorted_by_lifetime_duplication(db, dataset):
    target = dataset[db["name"]]

    for c in db["competitors"]:
        comp = dataset[c["name"]]
        c["__start__"] = comp.index[0].timestamp()
        c["__end__"] = comp.index[-1].timestamp()

    period0 = {"__start__": target.index[0].timestamp(), "__end__": target.index[-1].timestamp()}
    sorted_competitors = sort_periods_by_overlap(period0, db["competitors"])

    return sorted_competitors    