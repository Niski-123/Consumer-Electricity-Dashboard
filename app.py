
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# PROFESSIONAL VISUALIZATION SETTINGS
sns.set_theme(
    style="whitegrid",
    context="notebook",
    font_scale=1.2
)

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 120,
    "savefig.dpi": 150,

    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",

    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.titlepad": 16,

    "axes.labelsize": 14,
    "axes.labelweight": "medium",

    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    "legend.fontsize": 12,

    "grid.alpha": 0.25,
    "grid.linestyle": "--",

    "axes.spines.top": False,
    "axes.spines.right": False,

    "font.family": "DejaVu Sans"
})




# Professional Light Chart Theme
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#CBD5E1",
    "axes.labelcolor": "#334155",
    "text.color": "#172554",
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "axes.titlecolor": "#172554",
    "grid.color": "#E2E8F0",
    "font.size": 10
})



from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# Page configuration
st.set_page_config(
    page_title="Consumer Electricity Dashboard",
    page_icon="⚡",
    layout="wide"
)

# Dashboard styling
st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FA;
        color: #172554;
    }

    h1, h2, h3 {
        color: #172554 !important;
    }

    p, label, span {
        color: #334155;
    }

    .metric-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #DCE5EF;
        text-align: center;
    }

    .metric-value {
        color: #0284C7;
        font-size: 30px;
        font-weight: bold;
    }

    .metric-label {
        color: #64748B;
        font-size: 14px;
    }

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #DCE5EF;
        padding: 15px;
        border-radius: 12px;
    }

    [data-testid="stMetricValue"] {
        color: #0284C7;
    }

    .stApp {
        font-size: 18px;
    }

    .stMarkdown p {
        font-size: 17px !important;
        line-height: 1.7;
    }

    h1 {
        font-size: 38px !important;
    }

    h2 {
        font-size: 30px !important;
    }

    h3 {
        font-size: 23px !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 30px !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 15px !important;
    }

    button[data-baseweb="tab"] {
        font-size: 16px !important;
    }

    .stCaption {
        font-size: 14px !important;
    }


    /* Medium and readable typography */
    .stApp {
        font-size: 16px !important;
    }

    .stMarkdown p,
    .stMarkdown li {
        font-size: 16px !important;
        color: #334155 !important;
        line-height: 1.6 !important;
    }

    h1 {
        font-size: 34px !important;
        color: #172554 !important;
    }

    h2 {
        font-size: 27px !important;
        color: #172554 !important;
    }

    h3 {
        font-size: 21px !important;
        color: #172554 !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 27px !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 14px !important;
    }

    /* Fix success and information message readability */
    [data-testid="stAlert"] {
        background-color: #E0F2FE !important;
        border: 1px solid #BAE6FD !important;
    }

    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span {
        color: #0C4A6E !important;
        font-size: 16px !important;
    }

    /* Improve tab text */
    button[data-baseweb="tab"] {
        font-size: 15px !important;
    }

    /* Improve table text */
    [data-testid="stDataFrame"] {
        font-size: 15px !important;
    }

</style>
""", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv("/content/CONSUMER_ELECTRICITY_PRO.csv")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Convert AC values
df["Has_AC"] = df["Has_AC"].replace({
    "Yes": 1,
    "No": 0
})

df["Has_AC"] = pd.to_numeric(df["Has_AC"], errors="coerce")

# Remove incomplete records
df = df.dropna().copy()

# Exclude incomplete April 8
df = df[
    df["Date"].dt.date != pd.Timestamp("2025-04-08").date()
].copy()

# Household-level dataset
household_df = df.groupby("Household_ID").agg({
    "Energy_Consumption_kWh": "mean",
    "Household_Size": "first",
    "Avg_Temperature_C": "mean",
    "Has_AC": "first",
    "Peak_Hours_Usage_kWh": "mean"
}).reset_index()

# Features
features = [
    "Energy_Consumption_kWh",
    "Household_Size",
    "Avg_Temperature_C",
    "Has_AC",
    "Peak_Hours_Usage_kWh"
]

X = household_df[features]

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

household_df["Cluster"] = kmeans.fit_predict(X_scaled)

# Evaluation
silhouette = silhouette_score(
    X_scaled,
    household_df["Cluster"],
    sample_size=min(3000, len(X_scaled)),
    random_state=42
)

db_score = davies_bouldin_score(
    X_scaled,
    household_df["Cluster"]
)

ch_score = calinski_harabasz_score(
    X_scaled,
    household_df["Cluster"]
)

# Dashboard title
st.title("⚡ Consumer Electricity Usage Dashboard")

st.markdown(
    "### Consumer Electricity Usage – Machine Learning Clustering"
)

st.write(
    "An interactive dashboard to analyze household electricity "
    "consumption and identify consumer groups using K-Means clustering."
)


# KPI Cards
total_households = len(household_df)
average_usage = household_df["Energy_Consumption_kWh"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Households Analyzed", f"{total_households:,}")
col2.metric("Optimal Clusters (K)", "4")
col3.metric("Silhouette Score", f"{silhouette:.4f}")
col4.metric("Average Usage", f"{average_usage:.2f} kWh")

st.divider()

# Dashboard navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "EDA Analysis",
    "Model Evaluation",
    "Clustering Results",
    "Dataset"
])


# Overview section
with tab1:

    st.header("Project Overview")

    st.write("""
    This project uses **K-Means Clustering**, an unsupervised
    machine learning algorithm, to identify different household
    electricity consumption patterns.

    The objective is to group households based on their electricity
    usage, household size, average temperature, AC availability
    and peak-hour consumption.
    """)

    st.subheader("Project Objectives")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        - Analyze household electricity consumption.
        - Identify different consumer groups.
        - Study AC and non-AC consumption patterns.
        """)

    with col2:
        st.markdown("""
        - Apply feature scaling.
        - Select the optimal number of clusters.
        - Evaluate the K-Means model.
        """)

    st.divider()

    st.subheader("Machine Learning Workflow")

    st.markdown("""
    **1. Data Collection** → **2. Data Cleaning** →
    **3. Feature Selection** → **4. Standardization** →
    **5. K-Means Clustering** → **6. Model Evaluation**
    """)

    st.divider()

    st.subheader("Identified Consumer Groups")

    cluster_names = {
        0: "Very High Consumption - AC",
        1: "Low Consumption - Non-AC",
        2: "Moderate-Low Consumption - AC",
        3: "High Consumption - Non-AC"
    }

    cluster_summary = household_df.groupby("Cluster").agg(
        Households=("Household_ID", "count"),
        Average_Consumption=("Energy_Consumption_kWh", "mean")
    ).round(2)

    cluster_summary["Consumer Group"] = [
        cluster_names.get(int(i), f"Cluster {i}")
        for i in cluster_summary.index
    ]

    cluster_summary = cluster_summary[
        ["Consumer Group", "Households", "Average_Consumption"]
    ]

    cluster_summary.index.name = "Cluster"

    st.dataframe(
        cluster_summary,
        use_container_width=True
    )

    st.divider()

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Silhouette Score", f"{silhouette:.4f}")
    col2.metric("Number of Clusters", "4")
    col3.metric("Households Analyzed", f"{len(household_df):,}")

    st.markdown("""
    <div style="
        background-color: #E0F2FE;
        color: #0C4A6E;
        padding: 16px 20px;
        border-left: 5px solid #0284C7;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 500;
    ">
        K-Means successfully identified four household
        electricity consumption groups.
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# TAB 2: EXPLORATORY DATA ANALYSIS
# ==========================================

with tab2:

    st.header("Exploratory Data Analysis")

    st.write(
        "Explore household electricity consumption patterns "
        "using the visualizations created during data analysis."
    )

    # 1. Electricity Consumption Distribution
    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.hist(
            household_df["Energy_Consumption_kWh"],
            bins=30,
            color="#3498DB",
            edgecolor="white"
        )

        ax.axvline(
            household_df["Energy_Consumption_kWh"].mean(),
            color="red",
            linestyle="--",
            label="Average Consumption"
        )

        ax.set_title("1. Electricity Consumption Distribution",
                     fontweight="bold")
        ax.set_xlabel("Electricity Consumption (kWh)")
        ax.set_ylabel("Number of Households")
        ax.legend()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # 2. Consumption Categories
    with col2:

        q1 = household_df["Energy_Consumption_kWh"].quantile(1/3)
        q2 = household_df["Energy_Consumption_kWh"].quantile(2/3)

        categories = pd.cut(
            household_df["Energy_Consumption_kWh"],
            bins=[-np.inf, q1, q2, np.inf],
            labels=["Low", "Medium", "High"]
        )

        counts = categories.value_counts().reindex(
            ["Low", "Medium", "High"]
        )

        fig, ax = plt.subplots(figsize=(9, 6))

        bars = ax.bar(
            counts.index,
            counts.values,
            color=["#2ECC71", "#F1C40F", "#E74C3C"]
        )

        for bar in bars:
            value = int(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width()/2,
                value + 25,
                f"{value:,}",
                ha="center",
                fontweight="bold"
            )

        ax.set_title("2. Low, Medium and High Consumption",
                     fontweight="bold")
        ax.set_ylabel("Number of Households")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # 3. Average Consumption by Family Size
    col1, col2 = st.columns(2)

    with col1:

        family_usage = household_df.groupby(
            "Household_Size"
        )["Energy_Consumption_kWh"].mean()

        fig, ax = plt.subplots(figsize=(9, 6))

        bars = ax.bar(
            family_usage.index.astype(str),
            family_usage.values,
            color="#3498DB"
        )

        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.15,
                f"{bar.get_height():.2f}",
                ha="center"
            )

        ax.set_title("3. Average Consumption by Family Size",
                     fontweight="bold")
        ax.set_xlabel("Family Members")
        ax.set_ylabel("Average Consumption (kWh)")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # 4. AC vs Non-AC Comparison
    with col2:

        ac_usage = household_df.groupby(
            "Has_AC"
        )["Energy_Consumption_kWh"].mean()

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.bar(
            ["Without AC", "With AC"],
            [ac_usage.get(0, 0), ac_usage.get(1, 0)],
            color=["#3498DB", "#E67E22"]
        )

        ax.set_title("4. AC vs Non-AC Consumption",
                     fontweight="bold")
        ax.set_ylabel("Average Consumption (kWh)")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # 5. Daily Average Consumption Trend
    col1, col2 = st.columns(2)

    with col1:

        daily_usage = df.groupby("Date")[
            "Energy_Consumption_kWh"
        ].mean()

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.plot(
            daily_usage.index,
            daily_usage.values,
            marker="o",
            linewidth=2.5,
            color="#8E44AD"
        )

        for x, y in zip(daily_usage.index, daily_usage.values):
            ax.annotate(
                f"{y:.2f}",
                (x, y),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center"
            )

        ax.set_title("5. Daily Average Electricity Consumption",
                     fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Average Consumption (kWh)")
        ax.tick_params(axis="x", rotation=30)

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # 6. Total vs Peak-Hour Consumption
    with col2:

        sample = household_df.sample(
            min(2500, len(household_df)),
            random_state=42
        )

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.scatter(
            sample["Energy_Consumption_kWh"],
            sample["Peak_Hours_Usage_kWh"],
            alpha=0.4,
            color="#E67E22",
            s=18
        )

        ax.set_title("6. Total vs Peak-Hour Consumption",
                     fontweight="bold")
        ax.set_xlabel("Total Consumption (kWh)")
        ax.set_ylabel("Peak-Hour Usage (kWh)")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


    st.success("All 6 EDA visualizations displayed successfully!")



with tab3:

    st.header("Feature Scaling and Model Evaluation")

    # 1. Feature Scaling
    st.subheader("Before and After Feature Scaling")

    # Preserve original feature names after scaling
    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=features
    )

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))

    sns.boxplot(data=X, ax=ax[0], color="#60A5FA")
    ax[0].set_title("Before Scaling", fontsize=15, fontweight="bold")
    ax[0].set_ylabel("Feature Values")
    ax[0].tick_params(axis="x", rotation=35, labelsize=9)

    sns.boxplot(data=X_scaled_df, ax=ax[1], color="#34D399")
    ax[1].set_title("After Scaling", fontsize=15, fontweight="bold")
    ax[1].set_ylabel("Standardized Values")
    ax[1].tick_params(axis="x", rotation=35, labelsize=9)

    fig.tight_layout()
    st.pyplot(fig, width=550)
    plt.close(fig)

    st.divider()

    # 2. Elbow Method
    st.subheader("Elbow Method")

    k_values = [1, 2, 3, 4, 5, 6]
    inertia_values = [64285, 36757, 27170, 20327, 17612, 15360]

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.plot(
        k_values,
        inertia_values,
        marker="o",
        linewidth=2,
        markersize=7,
        color="#2563EB"
    )

    for x, y in zip(k_values, inertia_values):
        ax.annotate(
            f"{y:,}",
            (x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=10
        )

    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    st.pyplot(fig, width=550)
    plt.close(fig)

    st.info("The elbow point suggests selecting K = 4.")

    st.divider()

    # 3. Silhouette Score
    st.subheader("Silhouette Score Analysis")

    k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    silhouette_values = [
        0.3694, 0.3735, 0.4136, 0.3910,
        0.3727, 0.3623, 0.3448, 0.3485, 0.3638
    ]

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.plot(
        k_values,
        silhouette_values,
        marker="o",
        linewidth=2,
        markersize=7,
        color="#7C3AED"
    )

    for x, y in zip(k_values, silhouette_values):
        ax.annotate(
            f"{y:.4f}",
            (x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9
        )

    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title(
        "Silhouette Score for Different K Values",
        fontsize=15,
        fontweight="bold"
    )
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    st.pyplot(fig, width=550)
    plt.close(fig)

    st.success("K = 4 gives the highest Silhouette Score.")

    st.divider()

    # 4. Final Model Evaluation
    st.subheader("Final Model Evaluation")

    col1, col2, col3 = st.columns(3)

    col1.metric("Silhouette Score", "0.4136")
    col2.metric("Davies-Bouldin Index", "0.9712")
    col3.metric("Calinski-Harabasz Score", "9265.06")



with tab4:

    st.header("Customer Clustering Results")

    # 1. Household Distribution
    st.subheader("Households in Each Cluster")

    cluster_counts = household_df["Cluster"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 3.5))

    bars = ax.bar(
        cluster_counts.index.astype(str),
        cluster_counts.values,
        color=["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"],
        width=0.6
    )

    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Number of Households", fontsize=12)
    ax.set_title("Household Distribution Across Clusters",
                 fontsize=15, fontweight="bold")

    ax.bar_label(bars, fmt="%.0f", padding=4, fontsize=10)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()
    st.pyplot(fig, width=550)
    plt.close(fig)

    st.divider()

    # 2. Average Consumption
    st.subheader("Average Electricity Consumption by Cluster")

    cluster_avg = household_df.groupby("Cluster")[
        "Energy_Consumption_kWh"
    ].mean()

    fig, ax = plt.subplots(figsize=(6, 3.5))

    bars = ax.bar(
        cluster_avg.index.astype(str),
        cluster_avg.values,
        color=["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"],
        width=0.6
    )

    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Average Consumption (kWh)", fontsize=12)
    ax.set_title("Average Consumption Across Clusters",
                 fontsize=15, fontweight="bold")

    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=10)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()
    st.pyplot(fig, width=550)
    plt.close(fig)

    st.divider()

    # 3. PCA Visualization
    st.subheader("PCA Visualization of Clusters")

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PCA 1": pca_result[:, 0],
        "PCA 2": pca_result[:, 1],
        "Cluster": household_df["Cluster"].values
    })

    fig, ax = plt.subplots(figsize=(7, 4))

    cluster_colors = [
        "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"
    ]

    for cluster in sorted(pca_df["Cluster"].unique()):

        data = pca_df[pca_df["Cluster"] == cluster]

        ax.scatter(
            data["PCA 1"],
            data["PCA 2"],
            label=f"Cluster {cluster}",
            color=cluster_colors[int(cluster)],
            alpha=0.45,
            s=9
        )

    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)
    ax.set_title("Household Clusters Using PCA",
                 fontsize=15, fontweight="bold")

    ax.legend(title="Consumer Groups", fontsize=10)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    st.pyplot(fig, width=550)
    plt.close(fig)

    st.divider()

    # 4. Cluster Profile
    st.subheader("Cluster Profile Comparison")

    cluster_profile = household_df.groupby("Cluster").agg({
        "Energy_Consumption_kWh": "mean",
        "Household_Size": "mean",
        "Has_AC": "mean",
        "Peak_Hours_Usage_kWh": "mean"
    }).round(2)

    cluster_profile.index.name = "Cluster"

    st.dataframe(
        cluster_profile,
        use_container_width=True
    )

    st.caption(
        "Cluster profiles show the average characteristics "
        "of households belonging to each group."
    )



with tab5:

    st.header("Dataset Explorer")

    st.write(
        "Explore, search and download the processed "
        "household electricity dataset."
    )

    # Dataset statistics
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Households", f"{len(household_df):,}")
    col2.metric("Total Features", len(household_df.columns))
    col3.metric("Total Clusters", household_df["Cluster"].nunique())

    st.divider()

    # Search
    st.subheader("Search Household Records")

    search_text = st.text_input(
        "Search by household ID, consumption, or other values",
        placeholder="Enter a value to search..."
    )

    if search_text:
        filtered_df = household_df[
            household_df.astype(str).apply(
                lambda row: row.str.contains(
                    search_text,
                    case=False,
                    na=False
                ).any(),
                axis=1
            )
        ]
    else:
        filtered_df = household_df

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=450
    )

    st.caption(f"Displaying {len(filtered_df):,} household records")

    st.divider()

    # Download
    st.subheader("Download Dataset")

    csv_data = household_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Processed Dataset (CSV)",
        data=csv_data,
        file_name="consumer_electricity_clustered.csv",
        mime="text/csv"
    )

