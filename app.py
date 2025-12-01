import os
from pathlib import Path
import io

import joblib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from tensorflow import keras

# App settings
BASE_DIR = Path('.')
MODEL_PATH = BASE_DIR / 'best_tft_enhanced.keras'
SCALER_PATH = BASE_DIR / 'scaler.joblib'
EVENT_MAP_PATH = BASE_DIR / 'event_type_mapping.joblib'

# default hyperparams (tune in UI)
FREQ = '1min'
HISTORY_MINUTES = 60
HORIZONS_MIN = [1, 5, 15]
MAX_MACHINES = 200

st.set_page_config(page_title='TFT-enhanced inference', layout='wide')

# Ensure stdout/stderr use UTF-8 on Windows to avoid 'charmap' encode errors
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def _safe_text(obj):
    try:
        s = str(obj)
    except Exception:
        try:
            s = repr(obj)
        except Exception:
            s = '<unprintable object>'
    # ensure valid utf-8
    return s.encode('utf-8', errors='replace').decode('utf-8')

st.title('TFT-enhanced — Resource prediction (Streamlit)')

st.markdown('Upload the three CSV files (`server_usage.csv`, `server_event.csv`, `batch_task.csv`) or leave blank to use workspace files if present.')

uploaded_usage = st.file_uploader('server_usage.csv', type=['csv'])
uploaded_event = st.file_uploader('server_event.csv', type=['csv'])
uploaded_batch = st.file_uploader('batch_task.csv', type=['csv'])

use_workspace_files = False
if not uploaded_usage and not uploaded_event and not uploaded_batch:
    if (BASE_DIR / 'server_usage.csv').exists() and (BASE_DIR / 'server_event.csv').exists() and (BASE_DIR / 'batch_task.csv').exists():
        use_workspace_files = st.checkbox('Use workspace CSV files', value=True)

def read_csv_maybe(uploaded, path, names):
    if uploaded is not None:
        try:
            return pd.read_csv(uploaded, header=None, names=names, low_memory=False)
        except Exception as e:
            st.error(_safe_text(f'Error reading uploaded {path.name}: {e}'))
            raise
    else:
        return pd.read_csv(path, header=None, names=names, low_memory=False)

def safe_int_series(s):
    return pd.to_numeric(s, errors='coerce').fillna(0).astype(int)

def resample_machine(mid, usage_df, plan_df, server_event_static, freq=FREQ):
    g = usage_df[usage_df['machineID'] == mid].copy()
    if g.empty:
        return None
    g = g.set_index('ts').sort_index()
    res = g[['util_cpu','util_mem','util_disk','load1','load5','load15']].resample(freq).mean()

    try:
        cap = server_event_static[server_event_static['machineID']==mid].iloc[0]
        for c in ['capacity_cpu','capacity_mem','capacity_disk','event_type']:
            res[c] = cap[c]
    except Exception:
        res[['capacity_cpu','capacity_mem','capacity_disk']] = np.nan
        res['event_type'] = 'add'

    res = res.reset_index().merge(plan_df[['ts','plan_cpu','plan_mem']], on='ts', how='left')
    res['plan_cpu'] = res['plan_cpu'].fillna(0.0)
    res['plan_mem'] = res['plan_mem'].fillna(0.0)
    res['machineID'] = mid
    return res

def build_sequences_panel(df, features, H, horizons):
    X, Y = [], []
    max_h = max(horizons)
    for mid, g in df.groupby('machineID', sort=False):
        g = g.sort_values('ts').reset_index(drop=True)
        vals = g[features].values
        target = g['util_cpu'].values
        L = len(g)
        for i in range(0, L - H - max_h + 1):
            idxs = [i + H + (h-1) for h in horizons]
            if any(idx >= L for idx in idxs):
                continue
            X.append(vals[i:i+H])
            Y.append(target[idxs])
    if len(X) == 0:
        return np.empty((0, H, len(features))), np.empty((0, len(horizons)))
    return np.stack(X), np.stack(Y)

def per_horizon_metrics(y_true, y_pred, horizons):
    metrics = {}
    for i, h in enumerate(horizons):
        abs_err = np.abs(y_true[:, i] - y_pred[:, i])
        mae = abs_err.mean()
        denom = np.maximum(np.abs(y_true[:, i]), 1e-3)
        mape = (abs_err / denom).mean() * 100.0
        metrics[f'h_{h}_mae'] = mae
        metrics[f'h_{h}_mape'] = mape
    return metrics


if st.button('Run pipeline'):
    with st.spinner('Loading CSVs and preparing data...'):
        try:
            # Read CSVs
            usage = read_csv_maybe(uploaded_usage, BASE_DIR / 'server_usage.csv',
                                   ["timestamp","machineID","util_cpu","util_mem","util_disk","load1","load5","load15"])
            event = read_csv_maybe(uploaded_event, BASE_DIR / 'server_event.csv',
                                   ["timestamp","machineID","event_type","event_detail","capacity_cpu","capacity_mem","capacity_disk"])
            batch = read_csv_maybe(uploaded_batch, BASE_DIR / 'batch_task.csv',
                                   ["create_ts","modify_ts","job_id","task_id","instance_num","status","plan_cpu","plan_mem"])

            # Safe types and timestamps
            usage['machineID'] = usage['machineID'].astype(int)
            event['machineID'] = event['machineID'].astype(int)

            EPOCH = pd.Timestamp('2017-01-01')
            usage['ts'] = EPOCH + pd.to_timedelta(usage['timestamp'].astype(int), unit='s')
            event['ts'] = EPOCH + pd.to_timedelta(event['timestamp'].astype(int), unit='s')
            batch['ts'] = EPOCH + pd.to_timedelta(batch['create_ts'].fillna(0).astype(int), unit='s')

            server_event_static = event.sort_values('timestamp').drop_duplicates('machineID')[["machineID","capacity_cpu","capacity_mem","capacity_disk","event_type"]]

            machines = usage['machineID'].unique()[:MAX_MACHINES]
            usage = usage[usage['machineID'].isin(machines)].copy()
            server_event_static = server_event_static[server_event_static['machineID'].isin(machines)].copy()

            plan = batch.groupby('create_ts', as_index=False).agg({'plan_cpu':'sum','plan_mem':'sum'})
            plan['ts'] = EPOCH + pd.to_timedelta(plan['create_ts'].astype(int), unit='s')

            out_frames = []
            for m in machines:
                r = resample_machine(m, usage, plan, server_event_static, freq=FREQ)
                if r is not None:
                    out_frames.append(r)
            resampled = pd.concat(out_frames, ignore_index=True)

            numeric_cols = ['util_cpu','util_mem','util_disk','load1','load5','load15',
                            'plan_cpu','plan_mem','capacity_cpu','capacity_mem','capacity_disk']

            resampled[numeric_cols] = resampled.groupby('machineID')[numeric_cols].apply(lambda x: x.ffill().bfill()).reset_index(level=0, drop=True)

            resampled['event_type'] = resampled['event_type'].fillna('add')

            # load event mapping & scaler if available
            event_map = None
            scaler = None
            if EVENT_MAP_PATH.exists():
                try:
                    event_map = joblib.load(EVENT_MAP_PATH)
                except Exception as e:
                    st.warning(f'Could not load event mapping: {e}')
            if SCALER_PATH.exists():
                try:
                    scaler = joblib.load(SCALER_PATH)
                except Exception as e:
                    st.warning(f'Could not load scaler: {e}')

            # Build event_type_cat based on mapping if available
            if event_map is not None:
                # mapping is int -> category, invert
                cat_to_code = {str(v): int(k) for k, v in event_map.items()}
                resampled['event_type_cat'] = resampled['event_type'].astype(str).map(lambda x: cat_to_code.get(x, 0))
            else:
                resampled['event_type_cat'] = resampled['event_type'].astype('category').cat.codes

            features = numeric_cols + ['event_type_cat']

            unique_times = np.sort(resampled['ts'].unique())
            t1, t2 = unique_times[int(0.7*len(unique_times))], unique_times[int(0.85*len(unique_times))]
            train_df = resampled[resampled['ts'] <= t1].copy()
            val_df   = resampled[(resampled['ts'] > t1) & (resampled['ts'] <= t2)].copy()
            test_df  = resampled[resampled['ts'] > t2].copy()

            if scaler is None:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler().fit(train_df[features])
                st.warning('No `scaler.joblib` found — fitted a new scaler on uploaded data (not recommended).')

            def scale_df(df):
                df2 = df.copy()
                df2[features] = scaler.transform(df[features])
                return df2

            train_s, val_s, test_s = scale_df(train_df), scale_df(val_df), scale_df(test_df)

            X_train, Y_train = build_sequences_panel(train_s, features, HISTORY_MINUTES, HORIZONS_MIN)
            X_val, Y_val = build_sequences_panel(val_s, features, HISTORY_MINUTES, HORIZONS_MIN)
            X_test, Y_test = build_sequences_panel(test_s, features, HISTORY_MINUTES, HORIZONS_MIN)

            st.success(f'Resampled shape: {resampled.shape} — sequences: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}')

        except Exception as e:
            st.error(f'Error preparing data: {e}')
            raise

    # Load model
    with st.spinner('Loading model...'):
        model = None
        if MODEL_PATH.exists():
            try:
                # compile=False to avoid optimizer issues; recompile after loading
                model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
                model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.Huber(), metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')])
                st.success('Model loaded from workspace.')
            except Exception as e:
                msg = str(e)
                # Common error when Lambda layers contain Python lambdas — offer unsafe deserialization
                if 'Lambda' in msg and ('Python lambda' in msg or 'unsafe' in msg or 'enable_unsafe_deserialization' in msg):
                    st.warning('Model contains a Lambda layer using a Python lambda. Enabling unsafe deserialization to attempt load (this may execute arbitrary code).')
                    try:
                        keras.config.enable_unsafe_deserialization()
                        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
                        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.Huber(), metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')])
                        st.success('Model loaded from workspace (unsafe deserialization enabled).')
                    except Exception as e2:
                        st.error(f'Could not load model even after enabling unsafe deserialization: {e2}')
                else:
                    st.error(f'Could not load model: {e}')
        else:
            st.error('Model file `best_tft_enhanced.keras` not found in workspace.')

    if model is not None:
        with st.spinner('Evaluating on test dataset...'):
            try:
                if X_test.shape[0] == 0:
                    st.warning('No test sequences built — cannot evaluate or predict.')
                else:
                    ds_test = tf.data.Dataset.from_tensor_slices((X_test.astype(np.float32), Y_test.astype(np.float32))).batch(128)
                    res = model.evaluate(ds_test, verbose=0)
                    st.write('Test evaluation (loss, mae):', res)

                    y_pred = model.predict(ds_test)
                    y_true_all = np.vstack([y for x, y in ds_test])
                    metrics = per_horizon_metrics(y_true_all, y_pred, HORIZONS_MIN)
                    st.subheader('Per-horizon metrics')
                    st.json(metrics)

                    # Plot sample predictions
                    num_points = min(400, len(y_true_all))
                    fig, axes = plt.subplots(len(HORIZONS_MIN), 1, figsize=(12, 3 * len(HORIZONS_MIN)), sharex=True)
                    if len(HORIZONS_MIN) == 1:
                        axes = [axes]
                    for i, h in enumerate(HORIZONS_MIN):
                        axes[i].plot(y_true_all[:num_points, i], label=f'Actual +{h}m')
                        axes[i].plot(y_pred[:num_points, i], '--', label=f'Pred +{h}m')
                        axes[i].set_ylabel('CPU %')
                        axes[i].set_title(f'+{h} min — MAE: {metrics[f"h_{h}_mae"]:.3f}')
                        axes[i].grid(True, linestyle='--', alpha=0.5)
                        axes[i].legend()
                    axes[-1].set_xlabel('Sequence index')
                    st.pyplot(fig)

                    # Inference example: pick a machine and show last sequence
                    st.subheader('Single-sequence inference')
                    machine_list = sorted(resampled['machineID'].unique())
                    sel_mid = st.selectbox('Select machine', machine_list)
                    last_seq = resampled[resampled.machineID==sel_mid].sort_values('ts').tail(HISTORY_MINUTES)
                    if len(last_seq) < HISTORY_MINUTES:
                        st.warning('Not enough history for selected machine to run inference.')
                    else:
                        seq = last_seq.copy()
                        seq[features] = scaler.transform(seq[features])
                        arr = seq[features].values.astype(np.float32)[None, ...]
                        pred = model.predict(arr)[0]
                        st.write('Predictions for horizons (minutes):')
                        for i, h in enumerate(HORIZONS_MIN):
                            st.metric(label=f'+{h} min', value=f'{pred[i]:.3f}%')

            except Exception as e:
                st.error(f'Error during model evaluation/prediction: {e}')

    st.info('Done. You can re-run with different CSV uploads or adjust code as needed.')
