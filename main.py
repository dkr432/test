import os
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import mannwhitneyu
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="결함 분류 모델 비교", layout="wide")

LABEL = "defect_label"
SENSORS = {
    "temperature_c": "온도",
    "pressure_torr": "압력",
    "gas_flow_sccm": "가스 유량",
    "etch_rate_nm_min": "식각률",
    "voltage_v": "전압",
    "current_ma": "전류",
}
DEFAULT_FILE = os.path.join(os.path.dirname(__file__), "data",
                            "semiconductor_wafer_defect_dataset_small.csv")

M_DUMMY = "기준선: 항상 정상(더미)"
M_BASE = "베이스라인: 로지스틱 회귀"
M_L1 = "① L1 규제 로지스틱"
M_RF = "② 랜덤포레스트"
M_MONO = "③ 단조 제약 부스팅"
MODELS = [M_DUMMY, M_BASE, M_L1, M_RF, M_MONO]
COLORS = {M_DUMMY: "#9aa0a6", M_BASE: "#4C78A8", M_L1: "#72B7B2",
          M_RF: "#F58518", M_MONO: "#E45756"}
MONO_MODES = {
    "높을수록 위험(+) 가설을 모든 센서에 적용": "plus",
    "데이터에서 센서별 방향 추정(로지스틱 부호)": "auto",
    "제약 없음": "none",
}


# ------------------------------------------------------------------ 모델 정의
class MonotoneBoost(BaseEstimator, ClassifierMixin):
    """단조 제약을 거는 Histogram Gradient Boosting.
    mode: plus=모든 센서 +1(높을수록 위험), auto=학습 데이터의 로지스틱 계수 부호, none=제약 없음"""

    def __init__(self, mode="plus", max_iter=100, learning_rate=0.07, max_depth=3,
                 min_samples_leaf=5, random_state=0):
        self.mode = mode
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        k = X.shape[1]
        if self.mode == "plus":
            cst = np.ones(k, dtype=int)
        elif self.mode == "auto":
            sc = StandardScaler().fit(X)
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
            lr.fit(sc.transform(X), y)
            cst = np.sign(lr.coef_[0]).astype(int)
        else:
            cst = np.zeros(k, dtype=int)
        self.cst_ = cst
        self.model_ = HistGradientBoostingClassifier(
            max_iter=self.max_iter, learning_rate=self.learning_rate,
            max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
            l2_regularization=1.0, monotonic_cst=[int(c) for c in cst],
            class_weight="balanced", early_stopping=False,
            random_state=self.random_state)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))


def _sk_version():
    try:
        major, minor = sklearn.__version__.split(".")[:2]
        return int(major), int(minor)
    except Exception:
        return (1, 0)


def l1_logreg(C):
    """sklearn 버전에 맞춰 L1 로지스틱 회귀를 만든다."""
    if _sk_version() >= (1, 8):
        return LogisticRegression(l1_ratio=1, solver="liblinear", C=C,
                                  class_weight="balanced", max_iter=2000)
    return LogisticRegression(penalty="l1", solver="liblinear", C=C,
                              class_weight="balanced", max_iter=2000)


def build_models(c_l1, rf_leaf, mono_mode):
    return {
        M_DUMMY: DummyClassifier(strategy="prior"),
        M_BASE: Pipeline([("sc", StandardScaler()),
                          ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                                     max_iter=2000))]),
        M_L1: Pipeline([("sc", StandardScaler()), ("clf", l1_logreg(c_l1))]),
        M_RF: RandomForestClassifier(n_estimators=120, min_samples_leaf=rf_leaf,
                                     class_weight="balanced_subsample",
                                     random_state=0, n_jobs=1),
        M_MONO: MonotoneBoost(mode=mono_mode),
    }


@st.cache_data(show_spinner="모델 5개를 교차검증하는 중... (처음에는 시간이 걸립니다)")
def run_cv(X, y, c_l1, rf_leaf, mono_mode, folds, repeats):
    oof = {m: np.zeros((repeats, len(y))) for m in MODELS}
    for r in range(repeats):
        skf = StratifiedKFold(folds, shuffle=True, random_state=r)
        for tr, te in skf.split(X, y):
            for name, est in build_models(c_l1, rf_leaf, mono_mode).items():
                if name == M_DUMMY:
                    continue
                est.fit(X.iloc[tr], y[tr])
                oof[name][r, te] = est.predict_proba(X.iloc[te])[:, 1]
    oof[M_DUMMY][:] = y.mean()  # 항상 같은 확률 = 아무 정보 없는 기준선
    return oof


@st.cache_data(show_spinner="단조 제약 가설을 검증하는 중...")
def run_mono_cv(X, y, folds, repeats):
    out = {}
    for mode in ["plus", "auto", "none"]:
        oof = np.zeros((repeats, len(y)))
        for r in range(repeats):
            skf = StratifiedKFold(folds, shuffle=True, random_state=r)
            for tr, te in skf.split(X, y):
                est = MonotoneBoost(mode=mode).fit(X.iloc[tr], y[tr])
                oof[r, te] = est.predict_proba(X.iloc[te])[:, 1]
        out[mode] = oof
    return out


@st.cache_resource(show_spinner="전체 데이터로 학습하는 중...")
def fit_full(X, y, c_l1, rf_leaf, mono_mode):
    models = build_models(c_l1, rf_leaf, mono_mode)
    for est in models.values():
        est.fit(X, y)
    return models


def recall_at_top(y, p, frac):
    k = max(1, int(np.ceil(frac * len(y))))
    idx = np.argsort(-p, kind="stable")[:k]
    return y[idx].sum() / y.sum()


def all_metrics(y, p, thr, topk):
    pred = (p >= thr).astype(int)
    return {
        "PR-AUC": average_precision_score(y, p),
        "ROC-AUC": roc_auc_score(y, p),
        f"상위 {topk:g}% 재현율": recall_at_top(y, p, topk / 100),
        "재현율(임계값)": recall_score(y, pred, zero_division=0),
        "정밀도(임계값)": precision_score(y, pred, zero_division=0),
        "F1(임계값)": f1_score(y, pred, zero_division=0),
        "MCC(임계값)": matthews_corrcoef(y, pred),
    }


# ------------------------------------------------------------------ 데이터
@st.cache_data
def load(src):
    return pd.read_csv(src)


st.title("🔬 결함 분류 모델 비교 (베이스라인 vs 추천 모델 3개)")
up = st.sidebar.file_uploader("CSV 업로드 (없으면 샘플 사용)", type="csv")
if up is not None:
    df = load(up)
elif os.path.exists(DEFAULT_FILE):
    df = load(DEFAULT_FILE)
else:
    st.info("왼쪽에서 CSV 파일을 업로드해 주세요.")
    st.stop()

need = [c for c in [*SENSORS, LABEL] if c not in df.columns]
if need:
    st.error(f"필요한 열이 없습니다: {need}")
    st.stop()

y = df[LABEL].astype(int).values
n, n_pos = len(df), int(y.sum())
if n_pos < 3 or n_pos == n:
    st.error("결함(1)이 최소 3건, 정상(0)도 있어야 교차검증을 할 수 있습니다.")
    st.stop()

# ------------------------------------------------------------------ 사이드바
st.sidebar.header("설정")
feats = st.sidebar.multiselect("입력 센서", list(SENSORS), default=list(SENSORS),
                               format_func=SENSORS.get)
if len(feats) < 1:
    st.warning("입력 센서를 1개 이상 선택하세요.")
    st.stop()
max_folds = max(2, min(7, n_pos))
folds = st.sidebar.slider("교차검증 겹 수", 2, max_folds, min(5, max_folds))
repeats = st.sidebar.slider("반복 횟수", 1, 10, 3)
c_l1 = st.sidebar.select_slider("② L1 규제 강도 C (작을수록 강함)",
                                options=[0.01, 0.03, 0.1, 0.3, 1.0], value=0.1)
rf_leaf = st.sidebar.slider("③ 랜덤포레스트 leaf 최소 샘플", 1, 20, 3)
mono_label = st.sidebar.selectbox("③ 단조 제약 방식", list(MONO_MODES), index=1)
mono_mode = MONO_MODES[mono_label]
topk = st.sidebar.slider("상위 몇 %를 '위험'으로 볼지", 1.0, 20.0, 5.0, 1.0)
thr = st.sidebar.slider("확률 임계값(보조 지표용)", 0.05, 0.95, 0.5, 0.05)

X = df[feats].copy()
st.markdown(f"**웨이퍼 {n:,}개** · **결함 {n_pos}건** ({n_pos / n:.2%}) · "
            f"교차검증 {folds}겹 × {repeats}회")
st.warning(f"결함이 {n_pos}건뿐이라 모델 순위가 분할에 따라 바뀔 수 있습니다. "
           "표준편차를 함께 보고, 경향 확인용으로만 해석하세요. 정확도(accuracy)는 쓰지 않습니다.")

oof = run_cv(X, y, c_l1, rf_leaf, mono_mode, folds, repeats)
mean_p = {m: oof[m].mean(axis=0) for m in MODELS}

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["성능 비교", "PR · ROC 곡선", "센서 영향 (방향)", "가설 검증: 높을수록 위험?", "예측 확률"])

# ------------------------------------------------------------------ 1. 성능 비교
with tab1:
    per_rep = {m: pd.DataFrame([all_metrics(y, oof[m][r], thr, topk) for r in range(repeats)])
               for m in MODELS}
    cols = list(per_rep[M_BASE].columns)
    summ = pd.DataFrame({m: [f"{per_rep[m][c].mean():.3f} ± {per_rep[m][c].std(ddof=0):.3f}"
                             for c in cols] for m in MODELS}, index=cols).T
    st.subheader("교차검증 성능 (평균 ± 표준편차)")
    st.dataframe(summ)
    st.caption("앞의 3개(PR-AUC, ROC-AUC, 상위 k% 재현율)는 임계값과 무관한 순위 지표라 모델끼리 비교하기 좋고, "
               "뒤의 '(임계값)' 지표는 확률 임계값에 따라 달라지므로 보조로만 보세요.")

    ap_mean = [per_rep[m]["PR-AUC"].mean() for m in MODELS]
    ap_std = [per_rep[m]["PR-AUC"].std(ddof=0) for m in MODELS]
    fig = go.Figure(go.Bar(x=MODELS, y=ap_mean, marker_color=[COLORS[m] for m in MODELS],
                           error_y=dict(type="data", array=ap_std)))
    fig.add_hline(y=n_pos / n, line_dash="dot", line_color="black",
                  annotation_text=f"무작위 기준 {n_pos / n:.3f}")
    fig.update_layout(title="PR-AUC 비교 (막대 = 평균, 선 = 표준편차)", yaxis_title="PR-AUC",
                      yaxis_range=[0, 1])
    st.plotly_chart(fig)

    fig = go.Figure()
    for m in MODELS:
        fig.add_trace(go.Box(y=per_rep[m]["PR-AUC"], name=m, marker_color=COLORS[m],
                             boxpoints="all", jitter=0.4, showlegend=False))
    fig.update_layout(title="반복마다의 PR-AUC (점이 퍼질수록 불안정)", yaxis_title="PR-AUC")
    st.plotly_chart(fig)

    best = max(MODELS[1:], key=lambda m: per_rep[m]["PR-AUC"].mean())
    gain = per_rep[best]["PR-AUC"].mean() - per_rep[M_BASE]["PR-AUC"].mean()
    if best == M_BASE:
        st.info("PR-AUC 기준으로 베이스라인이 가장 높습니다. 복잡한 모델의 이점이 이 데이터에서는 확인되지 않았습니다.")
    else:
        sd = max(per_rep[best]["PR-AUC"].std(ddof=0), per_rep[M_BASE]["PR-AUC"].std(ddof=0))
        verdict = "표준편차보다 커서 의미가 있을 수 있습니다" if gain > sd else "표준편차 안쪽이라 우연일 수 있습니다"
        st.info(f"PR-AUC 최고: **{best}** (베이스라인 대비 {gain:+.3f}) — 이 차이는 {verdict}. "
                "다만 ROC-AUC나 상위 k% 재현율에서는 순위가 다를 수 있으니 표 전체를 함께 보세요.")

# ------------------------------------------------------------------ 2. 곡선
with tab2:
    c1, c2 = st.columns(2)
    fig = go.Figure()
    for m in MODELS:
        pr, rc, _ = precision_recall_curve(y, mean_p[m])
        fig.add_trace(go.Scatter(x=rc, y=pr, mode="lines", name=m,
                                 line=dict(color=COLORS[m], width=3 if m != M_DUMMY else 1.5)))
    fig.update_layout(title="Precision-Recall 곡선", xaxis_title="재현율", yaxis_title="정밀도",
                      yaxis_range=[0, 1.02], legend=dict(orientation="h", y=-0.25))
    c1.plotly_chart(fig)
    fig = go.Figure()
    for m in MODELS:
        fpr, tpr, _ = roc_curve(y, mean_p[m])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=m,
                                 line=dict(color=COLORS[m], width=3 if m != M_DUMMY else 1.5)))
    fig.update_layout(title="ROC 곡선", xaxis_title="False positive rate",
                      yaxis_title="True positive rate", legend=dict(orientation="h", y=-0.25))
    c2.plotly_chart(fig)
    st.caption("반복 평균 확률로 그린 곡선입니다. 위쪽·오른쪽(PR) 또는 왼쪽 위(ROC)에 가까울수록 좋은 모델입니다.")

# ------------------------------------------------------------------ 3. 센서 영향
with tab3:
    full = fit_full(X, y, c_l1, rf_leaf, mono_mode)
    names = [SENSORS[c] for c in feats]

    st.subheader("센서별 방향과 크기")
    coef = pd.DataFrame({
        "센서": names,
        "베이스라인 계수": full[M_BASE].named_steps["clf"].coef_[0],
        "L1 계수": full[M_L1].named_steps["clf"].coef_[0],
        "랜덤포레스트 중요도": full[M_RF].feature_importances_,
        "단조 제약 방향": full[M_MONO].cst_,
    }).round(3)
    coef["단조 제약 방향"] = coef["단조 제약 방향"].map({1: "+1 (높을수록 위험)", -1: "-1 (낮을수록 위험)", 0: "0 (제약 없음)"})
    st.dataframe(coef)
    st.caption("표준화된 센서 기준 계수라 크기를 센서끼리 비교할 수 있습니다(+는 높을수록 결함 확률 증가, -는 감소). "
               "L1에서 0이 된 센서는 모델이 쓰지 않은 센서입니다. 결함이 적어 계수의 절대 크기는 부풀려지므로 부호와 순위만 참고하세요.")

    fig = go.Figure()
    for m in [M_BASE, M_L1]:
        fig.add_trace(go.Bar(x=names, y=full[m].named_steps["clf"].coef_[0], name=m,
                             marker_color=COLORS[m]))
    fig.update_layout(barmode="group", title="로지스틱 계수 비교 (표준화 센서)", yaxis_title="계수")
    st.plotly_chart(fig)

    st.subheader("센서 값에 따른 결함 확률 (나머지 센서는 중앙값 고정)")
    base = X.median()
    k = len(feats)
    ncol = 3
    nrow = int(np.ceil(k / ncol))
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=names)
    for i, s in enumerate(feats):
        grid = np.linspace(X[s].quantile(0.01), X[s].quantile(0.99), 50)
        g = pd.DataFrame({c: np.full(len(grid), base[c]) for c in feats})
        g[s] = grid
        for m in MODELS[1:]:
            p = full[m].predict_proba(g)[:, 1]
            fig.add_trace(go.Scatter(x=grid, y=p, mode="lines", name=m,
                                     line=dict(color=COLORS[m]), legendgroup=m,
                                     showlegend=(i == 0)),
                          row=i // ncol + 1, col=i % ncol + 1)
    fig.update_layout(height=300 * nrow + 80, legend=dict(orientation="h", y=-0.1))
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig)
    st.caption("곡선이 오른쪽으로 갈수록 올라가면 '값이 높을수록 결함 확률 증가', 내려가면 반대입니다. "
               "단조 제약 모델(빨강)은 제약 방향으로만 움직입니다. 모델마다 곡선이 크게 다르면 그 관계는 불확실하다는 뜻입니다.")

# ------------------------------------------------------------------ 4. 가설 검증
with tab4:
    st.subheader("센서별로 '높을수록 결함'이 맞는지 (단일 센서)")
    rows = []
    for s in feats:
        auc = roc_auc_score(y, df[s])
        pval = mannwhitneyu(df.loc[y == 1, s], df.loc[y == 0, s]).pvalue
        z = (df.loc[y == 1, s].mean() - df[s].mean()) / df[s].std()
        rows.append({"센서": SENSORS[s], "단일 AUC": round(auc, 3),
                     "결함 평균의 z-score": round(z, 2), "p-value": round(pval, 4),
                     "방향": "높을수록 결함" if auc > 0.5 else "낮을수록 결함"})
    st.dataframe(pd.DataFrame(rows).sort_values("단일 AUC", key=lambda c: (c - 0.5).abs(),
                                                ascending=False))
    st.caption("AUC 0.5는 구분 못 함, 1에 가까우면 높을수록 결함, 0에 가까우면 낮을수록 결함입니다. "
               "센서를 여러 개 검정했으므로 p-value가 작아 보여도 우연일 수 있습니다.")

    st.subheader("단조 제약 부스팅으로 가설을 직접 검증")
    mono = run_mono_cv(X, y, folds, repeats)
    labels = {"plus": "모든 센서 '높을수록 위험(+)' 강제", "auto": "센서별 방향 추정",
              "none": "제약 없음"}
    res = {k_: [average_precision_score(y, mono[k_][r]) for r in range(repeats)] for k_ in mono}
    tbl = pd.DataFrame({labels[k_]: [f"{np.mean(v):.3f} ± {np.std(v):.3f}"] for k_, v in res.items()},
                       index=["PR-AUC"]).T
    st.dataframe(tbl)
    fig = go.Figure(go.Bar(x=[labels[k_] for k_ in res], y=[np.mean(v) for v in res.values()],
                           error_y=dict(type="data", array=[np.std(v) for v in res.values()]),
                           marker_color=["#E45756", "#F58518", "#4C78A8"]))
    fig.update_layout(yaxis_title="PR-AUC", yaxis_range=[0, 1])
    st.plotly_chart(fig)
    plus_m, none_m = np.mean(res["plus"]), np.mean(res["none"])
    if plus_m < none_m - np.std(res["none"]):
        st.error("'모든 센서가 높을수록 위험'이라고 강제하면 성능이 눈에 띄게 떨어집니다. "
                 "일부 센서는 **낮을수록 위험**(또는 비선형)일 가능성이 큽니다. "
                 "위 표에서 방향이 반대인 센서를 확인하세요.")
    else:
        st.success("'모든 센서가 높을수록 위험'이라는 제약을 걸어도 성능이 거의 유지됩니다. 가설과 모순되지 않습니다.")
    st.caption("해석: 가설(+) 제약 모델의 성능이 제약 없는 모델과 비슷하면 '높을수록 위험'이 데이터와 맞고, "
               "크게 떨어지면 맞지 않는다는 뜻입니다.")

# ------------------------------------------------------------------ 5. 예측 확률
with tab5:
    pick = st.selectbox("모델 선택", MODELS[1:], index=3)
    p = mean_p[pick]
    fig = go.Figure()
    bins = dict(start=0, end=1, size=0.04)
    fig.add_trace(go.Histogram(x=p[y == 0], xbins=bins, name="정상", marker_color="#4C78A8", opacity=0.7))
    fig.add_trace(go.Histogram(x=p[y == 1], xbins=bins, name="결함", marker_color="#E45756", opacity=0.9))
    fig.add_vline(x=thr, line_dash="dash", annotation_text=f"임계값 {thr:.2f}")
    fig.update_layout(barmode="overlay", yaxis_type="log", xaxis_title="예측 결함 확률(교차검증)",
                      yaxis_title="웨이퍼 수 (로그)", title=pick)
    st.plotly_chart(fig)
    rank = pd.Series(p).rank(ascending=False, method="min").astype(int)
    out = df.assign(예측확률=np.round(p, 3), 확률순위=rank.values)
    keep = [c for c in ["wafer_id", "process_step"] if c in out.columns] + [LABEL, "예측확률", "확률순위"]
    st.subheader(f"실제 결함 웨이퍼의 확률 순위 (전체 {n:,}개 중)")
    st.dataframe(out[out[LABEL] == 1].sort_values("확률순위")[keep].reset_index(drop=True))
    st.caption("결함 웨이퍼가 순위 상위에 몰려 있으면 모델이 잘 구분한다는 뜻입니다.")
