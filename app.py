import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
from sklearn.inspection import permutation_importance

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="AI Biomarker Validation",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 AI-Discovered Biomarker Statistical Validation")

st.info(
    "Research prototype: This application demonstrates how AI-predicted "
    "patterns can be statistically evaluated for robustness and reliability. "
    "It is not a clinical diagnostic tool."
)


# =========================================================
# SESSION STATE
# =========================================================

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "model" not in st.session_state:
    st.session_state.model = None

if "X" not in st.session_state:
    st.session_state.X = None

if "y" not in st.session_state:
    st.session_state.y = None

if "selected_features" not in st.session_state:
    st.session_state.selected_features = []

if "cv" not in st.session_state:
    st.session_state.cv = None

if "model_name" not in st.session_state:
    st.session_state.model_name = None


# =========================================================
# FILE UPLOAD
# =========================================================

st.sidebar.header("1. Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is None:

    st.write("### Getting Started")

    st.write(
        "Upload a CSV or Excel dataset using the sidebar. "
        "The dataset should contain patient/sample features and a target column."
    )

    example_data = pd.DataFrame({
        "Biomarker_A": [12.1, 15.2, 10.5, 18.4, 11.7],
        "Biomarker_B": [4.2, 7.1, 3.8, 8.2, 4.5],
        "Age": [62, 71, 58, 75, 65],
        "Diagnosis": ["CJD", "CJD", "Control", "CJD", "Control"]
    })

    st.write("### Example structure")
    st.dataframe(example_data, use_container_width=True)

    st.stop()


# =========================================================
# READ DATASET
# =========================================================

try:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_excel(uploaded_file)

except Exception as e:

    st.error(f"Could not read the dataset: {e}")
    st.stop()


# =========================================================
# DATASET OVERVIEW
# =========================================================

st.header("2. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Samples", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

with col4:
    st.metric(
        "Numeric Features",
        len(df.select_dtypes(include=np.number).columns)
    )


st.subheader("Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# =========================================================
# MISSING VALUES
# =========================================================

st.subheader("Missing Values")

missing_values = df.isnull().sum()

missing_table = pd.DataFrame({
    "Column": missing_values.index,
    "Missing Values": missing_values.values
})

missing_table = missing_table[
    missing_table["Missing Values"] > 0
]

if len(missing_table) == 0:

    st.success("No missing values detected.")

else:

    st.dataframe(
        missing_table,
        use_container_width=True
    )


# =========================================================
# TARGET VARIABLE
# =========================================================

st.header("3. Select Target Variable")

target_column = st.selectbox(
    "Select the column containing the outcome/diagnosis:",
    df.columns
)

target_values = df[target_column].dropna().unique()

if len(target_values) != 2:

    st.error(
        "This prototype currently requires exactly two target classes "
        "(for example, CJD and Control)."
    )

    st.write("Detected classes:", list(target_values))

    st.stop()


st.write(
    "Detected target classes:",
    list(target_values)
)


# =========================================================
# FEATURES
# =========================================================

st.header("4. Prepare Features")

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

numeric_features = [
    column for column in numeric_columns
    if column != target_column
]

if len(numeric_features) == 0:

    st.error(
        "No numeric features were found."
    )

    st.stop()


selected_features = st.multiselect(
    "Select features for AI analysis:",
    numeric_features,
    default=numeric_features
)

if len(selected_features) == 0:

    st.warning(
        "Please select at least one feature."
    )

    st.stop()


# =========================================================
# PREPARE DATA
# =========================================================

analysis_df = df[
    selected_features + [target_column]
].copy()

analysis_df = analysis_df.dropna()

if len(analysis_df) < 10:

    st.error(
        "Too few complete samples remain after removing missing values."
    )

    st.stop()


X = analysis_df[selected_features]

y_original = analysis_df[target_column]

classes = list(y_original.unique())

class_mapping = {
    classes[0]: 0,
    classes[1]: 1
}

y = y_original.map(class_mapping)

st.write("### Target Encoding")

mapping_table = pd.DataFrame({
    "Original Class": list(class_mapping.keys()),
    "Encoded Value": list(class_mapping.values())
})

st.dataframe(
    mapping_table,
    use_container_width=True
)


# =========================================================
# AI MODEL
# =========================================================

st.header("5. AI Prediction")

model_name = st.selectbox(
    "Choose AI model:",
    [
        "Logistic Regression",
        "Random Forest",
        "Support Vector Machine"
    ]
)


def create_model(name):

    if name == "Logistic Regression":

        return Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    random_state=42
                )
            )
        ])

    elif name == "Random Forest":

        return RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

    else:

        return Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                SVC(
                    probability=True,
                    random_state=42
                )
            )
        ])


# =========================================================
# RUN AI ANALYSIS
# =========================================================

if st.button("🚀 Run AI Analysis"):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    model = create_model(model_name)

    with st.spinner("Training AI model..."):

        model.fit(
            X_train,
            y_train
        )

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    try:

        auc = roc_auc_score(
            y_test,
            probabilities
        )

    except:

        auc = np.nan


    # Save important information
    st.session_state.analysis_done = True
    st.session_state.model = model
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.selected_features = selected_features
    st.session_state.model_name = model_name

    st.session_state.accuracy = accuracy
    st.session_state.precision = precision
    st.session_state.recall = recall
    st.session_state.f1 = f1
    st.session_state.auc = auc

    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.predictions = predictions
    st.session_state.probabilities = probabilities


# =========================================================
# DISPLAY AI RESULTS
# =========================================================

if st.session_state.analysis_done:

    st.subheader("AI Prediction Performance")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Accuracy",
            f"{st.session_state.accuracy:.3f}"
        )

    with c2:
        st.metric(
            "Precision",
            f"{st.session_state.precision:.3f}"
        )

    with c3:
        st.metric(
            "Recall",
            f"{st.session_state.recall:.3f}"
        )

    with c4:
        st.metric(
            "F1 Score",
            f"{st.session_state.f1:.3f}"
        )

    with c5:

        if np.isnan(st.session_state.auc):
            st.metric("ROC-AUC", "N/A")

        else:
            st.metric(
                "ROC-AUC",
                f"{st.session_state.auc:.3f}"
            )


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        st.session_state.y_test,
        st.session_state.predictions
    )

    fig_cm, ax_cm = plt.subplots()

    ax_cm.imshow(cm)

    ax_cm.set_xlabel("Predicted Class")
    ax_cm.set_ylabel("Actual Class")
    ax_cm.set_title("Confusion Matrix")

    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])

    for i in range(2):

        for j in range(2):

            ax_cm.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    st.pyplot(fig_cm)


    # =====================================================
    # ROC CURVE
    # =====================================================

    if not np.isnan(st.session_state.auc):

        st.subheader("ROC Curve")

        fpr, tpr, _ = roc_curve(
            st.session_state.y_test,
            st.session_state.probabilities
        )

        fig_roc, ax_roc = plt.subplots()

        ax_roc.plot(
            fpr,
            tpr,
            label=f"ROC-AUC = {st.session_state.auc:.3f}"
        )

        ax_roc.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        ax_roc.set_xlabel(
            "False Positive Rate"
        )

        ax_roc.set_ylabel(
            "True Positive Rate"
        )

        ax_roc.set_title(
            "ROC Curve"
        )

        ax_roc.legend()

        st.pyplot(fig_roc)


    # =====================================================
    # CROSS VALIDATION
    # =====================================================

    st.header("6. Cross-Validation")

    st.write(
        "Cross-validation checks whether model performance remains "
        "consistent across different subsets of the dataset."
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    if st.session_state.cv is None:

        try:

            cv_scores = cross_val_score(
                st.session_state.model,
                st.session_state.X,
                st.session_state.y,
                cv=cv,
                scoring="roc_auc"
            )

            st.session_state.cv = cv_scores

        except Exception as e:

            st.error(
                f"Cross-validation failed: {e}"
            )


    if st.session_state.cv is not None:

        st.write(
            "Individual ROC-AUC scores:"
        )

        st.write(
            np.round(
                st.session_state.cv,
                3
            )
        )

        st.metric(
            "Mean Cross-Validated ROC-AUC",
            f"{st.session_state.cv.mean():.3f}"
        )

        st.metric(
            "Standard Deviation",
            f"{st.session_state.cv.std():.3f}"
        )


    # =====================================================
    # PERMUTATION TEST
    # =====================================================

    st.header("7. Permutation Testing")

    st.write(
        "Permutation testing asks whether the observed AI performance "
        "could occur by chance when the outcome labels are randomly shuffled."
    )

    number_of_permutations = st.slider(
        "Number of permutations:",
        20,
        200,
        50,
        10,
        key="permutation_number"
    )


    if st.button(
        "Run Permutation Test",
        key="run_permutation"
    ):

        with st.spinner(
            "Running permutation test..."
        ):

            actual_score = st.session_state.cv.mean()

            permutation_scores = []

            rng = np.random.RandomState(42)

            for _ in range(number_of_permutations):

                shuffled_y = rng.permutation(
                    st.session_state.y.values
                )

                try:

                    score = cross_val_score(
                        st.session_state.model,
                        st.session_state.X,
                        shuffled_y,
                        cv=cv,
                        scoring="roc_auc"
                    ).mean()

                    permutation_scores.append(
                        score
                    )

                except:

                    continue


            permutation_scores = np.array(
                permutation_scores
            )

            if len(permutation_scores) > 0:

                p_value = (
                    np.sum(
                        permutation_scores >= actual_score
                    ) + 1
                ) / (
                    len(permutation_scores) + 1
                )

                st.session_state.permutation_scores = permutation_scores
                st.session_state.permutation_p = p_value

            else:

                st.warning(
                    "No valid permutation results were obtained."
                )


    if "permutation_scores" in st.session_state:

        st.subheader("Permutation Test Results")

        st.metric(
            "Observed ROC-AUC",
            f"{st.session_state.cv.mean():.3f}"
        )

        st.metric(
            "Permutation p-value",
            f"{st.session_state.permutation_p:.4f}"
        )

        fig_perm, ax_perm = plt.subplots()

        ax_perm.hist(
            st.session_state.permutation_scores,
            bins=15
        )

        ax_perm.axvline(
            st.session_state.cv.mean(),
            linestyle="--",
            label="Observed ROC-AUC"
        )

        ax_perm.set_xlabel(
            "ROC-AUC under shuffled labels"
        )

        ax_perm.set_ylabel(
            "Frequency"
        )

        ax_perm.set_title(
            "Permutation Null Distribution"
        )

        ax_perm.legend()

        st.pyplot(fig_perm)


    # =====================================================
    # BOOTSTRAP
    # =====================================================

    st.header("8. Bootstrap Analysis")

    st.write(
        "Bootstrap analysis repeatedly resamples the dataset to estimate "
        "the stability of model performance."
    )

    number_of_bootstraps = st.slider(
        "Number of bootstrap samples:",
        20,
        200,
        50,
        10,
        key="bootstrap_number"
    )


    if st.button(
        "Run Bootstrap Analysis",
        key="run_bootstrap"
    ):

        with st.spinner(
            "Running bootstrap analysis..."
        ):

            bootstrap_scores = []

            rng = np.random.RandomState(42)

            for _ in range(number_of_bootstraps):

                indices = rng.choice(
                    len(st.session_state.X),
                    size=len(st.session_state.X),
                    replace=True
                )

                X_boot = st.session_state.X.iloc[
                    indices
                ]

                y_boot = st.session_state.y.iloc[
                    indices
                ]

                if y_boot.nunique() < 2:
                    continue

                try:

                    score = cross_val_score(
                        st.session_state.model,
                        X_boot,
                        y_boot,
                        cv=3,
                        scoring="roc_auc"
                    ).mean()

                    bootstrap_scores.append(
                        score
                    )

                except:

                    continue


            bootstrap_scores = np.array(
                bootstrap_scores
            )

            if len(bootstrap_scores) > 0:

                lower = np.percentile(
                    bootstrap_scores,
                    2.5
                )

                upper = np.percentile(
                    bootstrap_scores,
                    97.5
                )

                st.session_state.bootstrap_scores = bootstrap_scores
                st.session_state.bootstrap_lower = lower
                st.session_state.bootstrap_upper = upper


    if "bootstrap_scores" in st.session_state:

        st.subheader("Bootstrap Results")

        st.metric(
            "Mean Bootstrap ROC-AUC",
            f"{st.session_state.bootstrap_scores.mean():.3f}"
        )

        st.write(
            f"Approximate 95% bootstrap interval: "
            f"{st.session_state.bootstrap_lower:.3f} – "
            f"{st.session_state.bootstrap_upper:.3f}"
        )

        fig_boot, ax_boot = plt.subplots()

        ax_boot.hist(
            st.session_state.bootstrap_scores,
            bins=15
        )

        ax_boot.axvline(
            st.session_state.bootstrap_lower,
            linestyle="--",
            label="2.5th percentile"
        )

        ax_boot.axvline(
            st.session_state.bootstrap_upper,
            linestyle="--",
            label="97.5th percentile"
        )

        ax_boot.set_xlabel(
            "Bootstrap ROC-AUC"
        )

        ax_boot.set_ylabel(
            "Frequency"
        )

        ax_boot.set_title(
            "Bootstrap Distribution"
        )

        ax_boot.legend()

        st.pyplot(fig_boot)


    # =====================================================
    # FDR
    # =====================================================

    st.header("9. False Discovery Rate (FDR) Analysis")

    st.write(
        "FDR correction helps control false discoveries when multiple "
        "features are tested simultaneously."
    )

    fdr_results = []

    for feature in st.session_state.selected_features:

        class_0 = st.session_state.X.loc[
            st.session_state.y == 0,
            feature
        ]

        class_1 = st.session_state.X.loc[
            st.session_state.y == 1,
            feature
        ]

        if len(class_0) > 1 and len(class_1) > 1:

            try:

                statistic, p_value = ttest_ind(
                    class_0,
                    class_1,
                    equal_var=False,
                    nan_policy="omit"
                )

                fdr_results.append({
                    "Feature": feature,
                    "t-statistic": statistic,
                    "p-value": p_value
                })

            except:

                pass


    if len(fdr_results) > 0:

        fdr_df = pd.DataFrame(
            fdr_results
        )

        reject, corrected_p, _, _ = multipletests(
            fdr_df["p-value"],
            method="fdr_bh"
        )

        fdr_df["FDR-adjusted p-value"] = corrected_p

        fdr_df["Significant after FDR"] = reject

        fdr_df = fdr_df.sort_values(
            "FDR-adjusted p-value"
        )

        st.dataframe(
            fdr_df,
            use_container_width=True
        )


    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    try:

        if st.session_state.model_name == "Random Forest":

            importance = (
                st.session_state.model
                .feature_importances_
            )

        elif st.session_state.model_name == "Logistic Regression":

            classifier = (
                st.session_state.model
                .named_steps["classifier"]
            )

            importance = np.abs(
                classifier.coef_[0]
            )

        else:

            permutation = permutation_importance(
                st.session_state.model,
                st.session_state.X_test,
                st.session_state.y_test,
                n_repeats=10,
                random_state=42,
                scoring="roc_auc"
            )

            importance = (
                permutation.importances_mean
            )

        importance_df = pd.DataFrame({
            "Feature": st.session_state.selected_features,
            "Importance": importance
        })

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False
        )

        st.dataframe(
            importance_df,
            use_container_width=True
        )

        fig_imp, ax_imp = plt.subplots()

        top_features = importance_df.head(15)

        ax_imp.barh(
            top_features["Feature"],
            top_features["Importance"]
        )

        ax_imp.set_xlabel(
            "Importance"
        )

        ax_imp.set_title(
            "Most Important Features"
        )

        ax_imp.invert_yaxis()

        st.pyplot(fig_imp)

    except Exception as e:

        st.warning(
            f"Feature importance could not be calculated: {e}"
        )


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    st.header("11. Statistical Validation Summary")

    st.write(
        "The AI model identifies predictive patterns in the supplied dataset. "
        "However, model performance alone does not establish that a biomarker "
        "pattern is reliable."
    )

    st.write(
        "Cross-validation evaluates whether model performance remains "
        "consistent across different subsets of the data."
    )

    st.write(
        "Permutation testing compares the observed model performance with "
        "performance obtained after randomly shuffling outcome labels."
    )

    st.write(
        "Bootstrap analysis evaluates the stability of model performance "
        "under repeated resampling."
    )

    st.write(
        "FDR correction helps address the problem of multiple feature-level "
        "statistical tests."
    )

    st.success(
        "Analysis completed successfully."
    )