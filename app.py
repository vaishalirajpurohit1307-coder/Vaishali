import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# -------------------- Helpers --------------------
@st.cache_data
def load_data_from_csv(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded)

def preprocess_df(df: pd.DataFrame, target_col='Attrition'):
    # Copy to avoid mutation
    df = df.copy()
    if target_col in df.columns:
        y_raw = df[target_col].astype(str)
        X_raw = df.drop(columns=[target_col])
    else:
        y_raw = None
        X_raw = df
    # Identify numeric and categorical
    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_raw.select_dtypes(include=['object','category','bool']).columns.tolist()
    # Imputers
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_imputer, num_cols),
        ('cat', Pipeline([('impute', cat_imputer), ('ohe', ohe)]), cat_cols)
    ], remainder='drop')
    X_proc = preprocessor.fit_transform(X_raw)
    # build feature names
    feature_names = []
    feature_names += num_cols
    if cat_cols:
        cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(cat_cols)
        feature_names += cat_feature_names.tolist()
    X = pd.DataFrame(X_proc, columns=feature_names)
    if y_raw is not None:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        return X, y, preprocessor, le, num_cols, cat_cols, feature_names
    else:
        return X, None, preprocessor, None, num_cols, cat_cols, feature_names

def train_and_evaluate(X, y, random_state=42):
    # Train-test split with stratify
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=random_state, stratify=y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    }
    results = {}
    for name, model in models.items():
        # cross-val predictions on train to get training confusion
        y_train_cv_pred = cross_val_predict(model, X_train, y_train, cv=skf, method='predict', n_jobs=1)
        train_acc = accuracy_score(y_train, y_train_cv_pred)
        train_cm = confusion_matrix(y_train, y_train_cv_pred)
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, zero_division=0)
        recall = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_test_proba)
        except Exception:
            auc = np.nan
        test_cm = confusion_matrix(y_test, y_test_pred)
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'train_cm': train_cm,
            'test_acc': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'test_cm': test_cm,
            'fpr': fpr,
            'tpr': tpr
        }
    return results

def plot_confusion_matrix(cm, labels, title="Confusion matrix"):
    # returns plotly figure
    z = cm
    x = labels
    y = labels
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale='Blues', showscale=True, hovertemplate='Count: %{z}<extra></extra>'))
    fig.update_layout(title=title, xaxis_title='Predicted', yaxis_title='True')
    # add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            fig.add_annotation(x=x[j], y=y[i], text=str(cm[i,j]), showarrow=False, font=dict(color='black'))
    return fig

def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='predictions')
    return output.getvalue()

# -------------------- Streamlit UI --------------------
st.set_page_config(layout='wide', page_title="Attrition Dashboard")

st.title("Employee Attrition — Interactive Dashboard & Modelling")
st.markdown("Upload `EA.csv` or use the sample file. The app will let HR explore attrition drivers, train models, and predict on new data.")

# Sidebar for dataset upload / sample
st.sidebar.header("Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (EA.csv recommended)", type=['csv'])
use_sample = False
if uploaded_file is None:
    st.sidebar.info("No file uploaded — using embedded sample from dataset included with the repo if present.")
    # try to load /mnt/data/EA.csv if present in working dir on Streamlit cloud repo
    try:
        sample_df = pd.read_csv('EA.csv')
        use_sample = True
    except Exception:
        sample_df = None
else:
    sample_df = load_data_from_csv(uploaded_file)
    st.sidebar.success("File uploaded")

if sample_df is None:
    st.warning("No data available. Upload a CSV file named `EA.csv` or upload via the sidebar to use the app.")
    st.stop()

# Prepare original dataframe and do some basic cleaning
df_raw = sample_df.copy()
if 'Attrition' in df_raw.columns:
    df_raw['Attrition'] = df_raw['Attrition'].astype(str)
# Sidebar filters for job role and satisfaction slider
job_roles = sorted(df_raw['JobRole'].dropna().unique().tolist()) if 'JobRole' in df_raw.columns else []
selected_roles = st.sidebar.multiselect("Filter: Job Role (multi-select)", options=job_roles, default=job_roles)
satisfaction_col = None
# find a satisfaction-like column
for c in ['JobSatisfaction','EnvironmentSatisfaction','RelationshipSatisfaction','WorkLifeBalance','PerformanceRating']:
    if c in df_raw.columns:
        satisfaction_col = c
        break
if satisfaction_col:
    sat_min = int(df_raw[satisfaction_col].min()); sat_max = int(df_raw[satisfaction_col].max())
    sat_range = st.sidebar.slider(f"Filter: {satisfaction_col} range", min_value=sat_min, max_value=sat_max, value=(sat_min, sat_max))
else:
    sat_range = None

# Filtered dataframe for dashboard views
filtered = df_raw.copy()
if selected_roles:
    filtered = filtered[filtered['JobRole'].isin(selected_roles)]
if satisfaction_col and sat_range is not None:
    filtered = filtered[(filtered[satisfaction_col] >= sat_range[0]) & (filtered[satisfaction_col] <= sat_range[1])]

# Tabs: Dashboard, Modelling, Predict
tab = st.tabs(["Dashboard", "Model Playground", "Predict & Download"])

# -------------------- Dashboard tab --------------------
with tab[0]:
    st.header("Dashboard — HR insights")
    st.markdown("Interactive charts to help HR take action for retention. Use the sidebar filters (JobRole multi-select and satisfaction slider).")

    # Chart 1: Attrition rate by JobRole (bar + percent)
    if 'Attrition' in filtered.columns and 'JobRole' in filtered.columns:
        grp = filtered.groupby(['JobRole','Attrition']).size().reset_index(name='count')
        total_by_role = grp.groupby('JobRole')['count'].sum().reset_index(name='total')
        merged = grp.merge(total_by_role, on='JobRole')
        merged['pct'] = merged['count'] / merged['total']
        fig1 = px.bar(merged, x='JobRole', y='pct', color='Attrition', barmode='stack',
                      title='Attrition rate by Job Role (stacked)', labels={'pct':'Proportion'})
        fig1.update_layout(xaxis_tickangle=-45, height=450)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Attrition or JobRole column missing for Chart 1.")

    # Chart 2: Income vs Attrition (violin + box using plotly)
    if 'MonthlyIncome' in filtered.columns and 'Attrition' in filtered.columns:
        fig2 = px.violin(filtered, x='Attrition', y='MonthlyIncome', box=True, points='all',
                         title='Monthly Income distribution by Attrition (shows spread and outliers)')
        st.plotly_chart(fig2, use_container_width=True)
    # Chart 3: OverTime & Attrition by JobRole (heatmap-like pivot)
    if 'OverTime' in filtered.columns and 'JobRole' in filtered.columns:
        pivot = pd.crosstab(filtered['JobRole'], filtered['OverTime'], values=None, aggfunc='count').fillna(0)
        pivot = pivot.reindex(index=sorted(pivot.index))
        fig3 = px.imshow(pivot, labels=dict(x='OverTime', y='JobRole', color='count'), title='Counts by JobRole vs OverTime')
        st.plotly_chart(fig3, use_container_width=True)

    # Chart 4: Correlation matrix (numerical features) - heatmap for HR to see multicollinearity
    num_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 2:
        corr = filtered[num_cols].corr()
        fig4 = px.imshow(corr, title='Correlation matrix (numeric features) — check multicollinearity')
        st.plotly_chart(fig4, use_container_width=True)

    # Chart 5: Attrition drivers - interactive feature importance surrogate using RandomForest on filtered view
    st.subheader("Surrogate model: Decision importance for current filtered slice")
    with st.spinner("Training a quick surrogate RandomForest for feature importances..."):
        try:
            # require Attrition for this
            if 'Attrition' in filtered.columns:
                X_slice, y_slice, preproc_slice, le_slice, num_cols_slice, cat_cols_slice, feat_names_slice = preprocess_df(filtered, target_col='Attrition')
                rf = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=1)
                rf.fit(X_slice, y_slice)
                importances = pd.Series(rf.feature_importances_, index=feat_names_slice).sort_values(ascending=False).head(15)
                fig5 = go.Figure([go.Bar(x=importances.values[::-1], y=importances.index[::-1], orientation='h')])
                fig5.update_layout(title='Top 15 feature importances (surrogate RandomForest)', xaxis_title='Importance')
                st.plotly_chart(fig5, use_container_width=True)
                st.markdown("**Actionable insight examples:** Observe high importance features and run targeted interventions (e.g., compensation review for high MonthlyIncome-related attrition, manager training if YearsWithCurrManager is important).")
            else:
                st.info("Attrition not present — cannot create surrogate feature importance.")
        except Exception as e:
            st.error(f"Failed to compute surrogate model: {e}")

# -------------------- Model Playground tab --------------------
with tab[1]:
    st.header("Model Playground — train & evaluate models")
    st.markdown("Click the button below to train Decision Tree, Random Forest, and Gradient Boosting with stratified 5-fold CV (training confusion via CV predictions) and evaluate on a hold-out test set. This will run in-memory and show metrics and plots.")

    if st.button("Train & Evaluate Models"):
        with st.spinner("Preprocessing and training... this may take ~10-60s depending on dataset size"):
            try:
                X, y, preproc, le, num_cols, cat_cols, feat_names = preprocess_df(df_raw, target_col='Attrition')
                results = train_and_evaluate(X, y)
                # Metrics table
                rows = []
                for name, r in results.items():
                    rows.append({
                        'Algorithm': name,
                        'Train Accuracy (cv5)': r['train_acc'],
                        'Test Accuracy': r['test_acc'],
                        'Precision (test)': r['precision'],
                        'Recall (test)': r['recall'],
                        'F1-score (test)': r['f1'],
                        'AUC (test)': r['auc']
                    })
                df_metrics = pd.DataFrame(rows).round(3)
                st.subheader("Model performance summary")
                st.dataframe(df_metrics)

                # ROC combined
                fig = go.Figure()
                for name, r in results.items():
                    fig.add_trace(go.Scatter(x=r['fpr'], y=r['tpr'], mode='lines', name=f"{name} (AUC={r['auc']:.3f})"))
                fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash'))
                fig.update_layout(title='ROC curves', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=500)
                st.plotly_chart(fig, use_container_width=True)

                # Confusion matrices
                st.subheader("Confusion matrices (train via CV predictions and test)")
                cols = st.columns(3)
                for i, (name, r) in enumerate(results.items()):
                    with cols[i]:
                        st.markdown(f"**{name} - Train (CV)**")
                        st.plotly_chart(plot_confusion_matrix(r['train_cm'], labels=le.inverse_transform([0,1]) if le else ['No','Yes']), use_container_width=True)
                        st.markdown(f"**{name} - Test**")
                        st.plotly_chart(plot_confusion_matrix(r['test_cm'], labels=le.inverse_transform([0,1]) if le else ['No','Yes']), use_container_width=True)

                # Feature importances (if available)
                st.subheader("Feature importances (top 15)")
                for name, r in results.items():
                    model = r['model']
                    if hasattr(model, 'feature_importances_'):
                        importances = pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=False).head(15)
                        fig_imp = go.Figure([go.Bar(x=importances.values[::-1], y=importances.index[::-1], orientation='h')])
                        fig_imp.update_layout(title=f"{name} - Top 15 feature importances", xaxis_title='Importance', height=400)
                        st.plotly_chart(fig_imp, use_container_width=True)
            except Exception as e:
                st.error(f"Training failed: {e}")

# -------------------- Predict & Download tab --------------------
with tab[2]:
    st.header("Upload new data and predict Attrition")
    st.markdown("Upload a dataset with same schema (features used in model). The app will preprocess using the pipeline fitted in the last training run (if you trained earlier) or it will train a fresh model on the current EA data first. You can download predictions as an Excel file.")

    upload_new = st.file_uploader("Upload new CSV to predict (no target column required)", type=['csv'], key='predict_upload')
    if upload_new is not None:
        new_df = pd.read_csv(upload_new)
        st.write("Preview of uploaded data:", new_df.head())
        # Ensure we have trained model in session by running a light training if not present
        try:
            X_master, y_master, preproc_master, le_master, num_cols_master, cat_cols_master, feat_names_master = preprocess_df(df_raw, target_col='Attrition')
            # Train final GBRT model on full dataset
            final_model = GradientBoostingClassifier(n_estimators=200, random_state=42)
            final_model.fit(X_master, y_master)
            # Preprocess new data using the same preprocessor we built earlier -- BUT our preprocess_df returned its own preprocessor fitted on df_raw.
            # For simplicity re-fit a preprocessor on df_raw and use it to transform new data.
            # (A production app should persist the fitted preprocessor and trained model.)
            preprocessor = ColumnTransformer(transformers=[
                ('num', SimpleImputer(strategy='mean'), df_raw.select_dtypes(include=[np.number]).columns.tolist()),
                ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))]), df_raw.select_dtypes(include=['object','category','bool']).columns.tolist())
            ], remainder='drop')
            # fit the preprocessor on df_raw
            preprocessor.fit(df_raw.drop(columns=['Attrition']) if 'Attrition' in df_raw.columns else df_raw)
            # transform uploaded file (only keep columns that were seen in original df)
            # align columns: take the original df columns (without target) as baseline
            baseline_cols = (df_raw.drop(columns=['Attrition']) if 'Attrition' in df_raw.columns else df_raw).columns.tolist()
            new_df_aligned = new_df.copy()
            # add missing columns with NaN
            for c in baseline_cols:
                if c not in new_df_aligned.columns:
                    new_df_aligned[c] = np.nan
            new_df_aligned = new_df_aligned[baseline_cols]
            X_new = preprocessor.transform(new_df_aligned)
            # Get feature names for the transformed new data
            num_cols = baseline_cols if len(baseline_cols)>0 else []
            # predict
            preds_proba = final_model.predict_proba(X_new)[:,1]
            preds = final_model.predict(X_new)
            # Convert preds to original labels if possible using the earlier LabelEncoder
            if le_master is not None:
                pred_labels = le_master.inverse_transform(preds)
            else:
                pred_labels = preds.astype(str)
            out = new_df.copy()
            out['PredictedAttrition'] = pred_labels
            out['Predicted_Probability'] = preds_proba
            st.success("Prediction finished. Download below.")
            st.dataframe(out.head())
            # provide download button
            xlsx_bytes = to_excel_bytes(out)
            st.download_button("Download predictions (Excel)", data=xlsx_bytes, file_name="predictions_with_attrition.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ — upload your `EA.csv` or run on Streamlit Cloud (connect your GitHub repo).")
