import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Life Expectancy Dashboard", layout="wide")

# Load the data
data_file = "data.csv"
try:
    df = pd.read_csv(data_file)
    st.success("Data loaded successfully.")
except FileNotFoundError:
    st.error(f"Error: File '{data_file}' not found.")
    st.stop()

# Clean column names
df.columns = df.columns.str.strip()
target = 'Life expectancy'
if target not in df.columns:
    st.error(f"Error: '{target}' column not found in the dataset.")
    st.stop()

# Outlier removal function
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# Remove outliers
df = remove_outliers(df, target)

# Basic statistics
mean = df[target].mean()
median = df[target].median()
mode = df[target].mode()[0]
std_dev = df[target].std()
variance = df[target].var()
range_val = df[target].max() - df[target].min()
skewness = df[target].skew()
kurtosis = df[target].kurt()
missing_vals = df[target].isnull().sum()

# Title
st.title("📊 Life Expectancy Analysis Dashboard")

# Subheader for stats
st.subheader("📈 Summary Statistics")

# Row 1
col1, col2, col3 = st.columns(3)
col1.metric("📏 Mean", f"{mean:.2f}")
col2.metric("📍 Median", f"{median:.2f}")
col3.metric("🎯 Mode", f"{mode:.2f}")

# Row 2
col4, col5, col6 = st.columns(3)
col4.metric("📊 Std Deviation", f"{std_dev:.2f}")
col5.metric("📐 Variance", f"{variance:.2f}")
col6.metric("📶 Range", f"{range_val:.2f}")

# Row 3
col7, col8, col9 = st.columns(3)
col7.metric("↕️ Skewness", f"{skewness:.2f}")
col8.metric("🎲 Kurtosis", f"{kurtosis:.2f}")
col9.metric("❓ Missing Values", f"{missing_vals}")


# 📉 Linear Regression: GDP vs Life Expectancy
df['GDP'] = pd.to_numeric(df['GDP'], errors='coerce')
df_clean = df[[target, 'GDP']].dropna()
df_clean = remove_outliers(df_clean, 'GDP')

X = sm.add_constant(df_clean['GDP'])
y = df_clean[target]
model = sm.OLS(y, X).fit()
slope = model.params[1]
intercept = model.params[0]
r_value = df_clean.corr().loc['GDP', target]

st.subheader("📉 Linear Regression: GDP and Life Expectancy")

# Split into two columns
col1, col2 = st.columns(2)

# Left: Regression Plot
with col1:
    st.markdown("#### 📈 Regression Plot")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(x='GDP', y=target, data=df_clean, ax=ax)
    ax.set_xlabel("GDP")
    ax.set_ylabel("Life Expectancy (years)")
    ax.set_title("GDP vs Life Expectancy (Regression Line)")
    st.pyplot(fig)

# Right: Stats and Summary
with col2:
    st.markdown("#### 📋 Regression Summary")
    st.markdown(model.summary().as_html(), unsafe_allow_html=True)

st.markdown("#### 🧾 Equation of the Line")
st.latex(f"\\text{{Life Expectancy}} = {slope:.6f} \\cdot \\text{{GDP}} + {intercept:.2f}")

# Get correlation coefficient
# Calculate correlation manually and extract values
x = df_clean['GDP']
y = df_clean[target]

x_mean = x.mean()
y_mean = y.mean()

# Step-by-step values for Pearson correlation
numerator = ((x - x_mean) * (y - y_mean)).sum()
x_denom = np.sqrt(((x - x_mean) ** 2).sum())
y_denom = np.sqrt(((y - y_mean) ** 2).sum())
denominator = x_denom * y_denom
manual_r = numerator / denominator

# Correlation (also from pandas to confirm)
correlation = df_clean.corr().loc['GDP', target]

# Display correlation value
st.subheader("🔗 Correlation Between GDP and Life Expectancy")
st.markdown(f"**Correlation Coefficient (r):** {correlation:.3f}")

# Explanation with formula and components
with st.expander("ℹ️ What does the correlation coefficient mean?"):
    st.markdown(f"""
    The **correlation coefficient `r = {correlation:.3f}`** measures how strongly and in what direction GDP and Life Expectancy move together.

    ### 📐 Formula:
    \[
    r = \\frac{{\\sum (x_i - \\bar{{x}})(y_i - \\bar{{y}})}}{{\\sqrt{{\\sum (x_i - \\bar{{x}})^2}} \\cdot \\sqrt{{\\sum (y_i - \\bar{{y}})^2}}}}
    \]

    ### 🔢 Step-by-Step Values:
    - **Mean of GDP (𝑥̄)** = {x_mean:.2f}  
    - **Mean of Life Expectancy (𝑦̄)** = {y_mean:.2f}  
    - **Numerator (Covariance Sum)** = {numerator:.2e}  
    - **Denominator (Product of Standard Deviations)** = {denominator:.2e}  
    - **r = Numerator ÷ Denominator =** {manual_r:.6f}

    ### 📈 Interpretation Scale:
    | `r` value       | Strength of Relationship |
    |-----------------|--------------------------|
    | 0.0 to ±0.2     | Very weak                |
    | ±0.2 to ±0.4    | Weak                     |
    | ±0.4 to ±0.6    | Moderate                 |
    | ±0.6 to ±0.8    | Strong                   |
    | ±0.8 to ±1.0    | Very strong              |

    Since **r = {correlation:.3f}**, this suggests a **moderate positive correlation** between GDP and life expectancy — as GDP increases, life expectancy tends to increase as well.
    """)



# One-sample t-test
chosen_value = 70
t_stat, p_val = stats.ttest_1samp(df[target].dropna(), chosen_value)

with st.expander("🧪 One-Sample t-Test – Is Life Expectancy Different from 70?"):
    st.markdown("""
    This test checks if the **average life expectancy** in our data is **different from 70 years**.

    It answers the question:  
    🧠 *"Is the average life expectancy in the world higher or lower than 70?"*

    Here's what the results mean:
    - **T-Statistic** tells us how far our average is from 70.
    - **P-Value** tells us if that difference is likely just by chance.
      - If the p-value is less than 0.05, the result is **statistically significant** – meaning it's very unlikely to be random.

    Below are the results:
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric(label="🎯 Tested Against", value=f"{chosen_value}")
    col2.metric(label="📊 T-Statistic", value=f"{t_stat:.4f}")
    col3.metric(label="🧪 P-Value", value=f"{p_val:.4f}")

    if p_val < 0.05:
        st.success("✅ Conclusion: The average life expectancy is **significantly different** from 70.")
    else:
        st.info("ℹ️ Conclusion: There's **no strong evidence** that life expectancy differs from 70.")



# Confidence Interval
mean_life = np.mean(df[target].dropna())
std_life = np.std(df[target].dropna(), ddof=1)
n = len(df[target].dropna())
conf_int = stats.t.interval(confidence=0.95, df=n-1, loc=mean_life, scale=std_life/np.sqrt(n))

with st.expander("📏 95% Confidence Interval (Click to Learn More)"):
    st.markdown("This interval shows the range where 95% of the countries average ages lie, based on the sample.")

    col1, col2, col3 = st.columns(3)
    col1.metric(label="Confidence Interval", value=f"({conf_int[0]:.2f}, {conf_int[1]:.2f})")
    col2.metric(label="Mean", value=f"{mean_life:.2f}")
    col3.metric(label="Sample Size (n)", value=n)


with st.expander("❓ Missing Values in the Dataset (Click to Expand)"):
    st.markdown("""
    Missing values occur when some data entries are not available or were not recorded.

    🔎 **Why this matters:**  
    Missing data can affect analysis and models. We usually fix this by removing, filling in, or estimating those values.

    Below is a list of columns with missing values:
    """)
    
    st.dataframe(df.isnull().sum().to_frame(name='🔧 Missing Values'))


st.header("📊 Correlation Tests Between GDP and Life Expectancy")

# Pearson Correlation
pearson_r, pearson_p = stats.pearsonr(df_clean['GDP'], df_clean[target])
# Spearman Correlation
spearman_r, spearman_p = stats.spearmanr(df_clean['GDP'], df_clean[target])
# Kendall Tau Correlation
kendall_r, kendall_p = stats.kendalltau(df_clean['GDP'], df_clean[target])

# Display metrics
col1, col2, col3 = st.columns(3)
col1.metric("🔹 Pearson r", f"{pearson_r:.3f}", f"p = {pearson_p:.4f}")
col2.metric("🔸 Spearman ρ", f"{spearman_r:.3f}", f"p = {spearman_p:.4f}")
col3.metric("🟠 Kendall τ", f"{kendall_r:.3f}", f"p = {kendall_p:.4f}")

# Side-by-side layout for heatmap and interpretation
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("📈 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df_clean.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("GDP and Life Expectancy Correlation Matrix")
    st.pyplot(fig)

with right_col:
    st.subheader("📘 Interpretation of Correlation Results")
    st.markdown(f"""
    Each test examines how GDP relates to life expectancy:

    | Test         | Coefficient | p-value    | Interpretation Type       |
    |--------------|-------------|------------|----------------------------|
    | Pearson r    | {pearson_r:.3f}       | {pearson_p:.4f}   | Measures **linear** correlation. |
    | Spearman ρ   | {spearman_r:.3f}       | {spearman_p:.4f}   | Measures **monotonic** relationships using **ranks**. |
    | Kendall τ    | {kendall_r:.3f}       | {kendall_p:.4f}   | Measures **ordinal** association. |

    ✅ All p-values are less than 0.05, so the correlations are **statistically significant**.

    💡 **Conclusion**: GDP and life expectancy are positively correlated across all three tests.
    """)



# Regression analysis
if 'GDP' not in df.columns:
    st.error("Error: 'GDP' column not found in the dataset.")
    st.stop()

df['GDP'] = pd.to_numeric(df['GDP'], errors='coerce')
df_clean = df[[target, 'GDP']].dropna()
df_clean = remove_outliers(df_clean, 'GDP')

# Linear regression model
X = sm.add_constant(df_clean['GDP'])
y = df_clean[target]
model = sm.OLS(y, X).fit()


# Linear regression model
X = sm.add_constant(df_clean['GDP'])
y = df_clean[target]
model = sm.OLS(y, X).fit()





# 📊 Interactive Dropdown Visualizations
st.subheader("📊 Explore Visualizations")

# Dropdown options and explanations
plot_options = {
    "Life Expectancy Distribution": "Distribution of life expectancy values across all countries.",
    "GDP vs Life Expectancy (Regression Line)": "Linear relationship between GDP and life expectancy.",
    "GDP Histogram": "GDP distribution showing economic disparities.",
    "Boxplot of Life Expectancy by Status": "Compare life expectancy between developing and developed nations.",
    "GDP vs Life Expectancy (Scatter)": "Simple scatterplot showing GDP vs Life Expectancy.",
    "Year vs Life Expectancy": "How life expectancy has changed over years.",
    "Schooling vs Life Expectancy": "Relationship between education and life expectancy.",
    "Adult Mortality vs Life Expectancy": "How adult mortality affects life expectancy.",
    "Alcohol vs Life Expectancy": "Influence of alcohol consumption on life expectancy.",
    "BMI vs Life Expectancy": "Relation between body mass index and life expectancy.",
    "HIV/AIDS vs Life Expectancy": "Impact of HIV/AIDS cases on life expectancy.",
    "Polio vs Life Expectancy": "Effect of polio immunization rates on life expectancy.",
    "Diphtheria vs Life Expectancy": "Diphtheria immunization coverage vs life expectancy.",
    "Hepatitis B vs Life Expectancy": "Impact of Hepatitis B vaccination rates.",
    "Total Expenditure vs Life Expectancy": "Government health spending's influence.",
    "Percentage Expenditure vs Life Expectancy": "Health expenditure as % of GDP vs life expectancy.",
    "Infant Deaths vs Life Expectancy": "Higher infant deaths often correlate with lower life expectancy.",
    "Under-five Deaths vs Life Expectancy": "Under-five deaths' impact on national life expectancy.",
    "Population vs Life Expectancy": "Explore whether population size impacts life expectancy."
}

selected_plot = st.selectbox("📌 Choose a graph to view:", options=list(plot_options.keys()))

# Split columns
col1, col2 = st.columns([1.5, 1])

# 📊 Left: Render Plot
with col1:
    fig, ax = plt.subplots(figsize=(6, 4))

    if selected_plot == "Life Expectancy Distribution":
        sns.histplot(df[target], kde=True, ax=ax)
        ax.set_xlabel("Life Expectancy (years)")
        ax.set_ylabel("Number of Records")

    elif selected_plot == "GDP vs Life Expectancy (Regression Line)":
        df_clean = df[['GDP', target]].dropna()
        sns.regplot(x='GDP', y=target, data=df_clean, ax=ax)
        ax.set_xlabel("GDP")
        ax.set_ylabel("Life Expectancy (years)")

    elif selected_plot == "GDP Histogram":
        df_clean = df[['GDP']].dropna()
        sns.histplot(df_clean['GDP'], bins=30, kde=True, ax=ax)
        ax.set_xlabel("GDP")
        ax.set_ylabel("Number of Records")

    elif selected_plot == "Boxplot of Life Expectancy by Status":
        sns.boxplot(x='Status', y=target, data=df, ax=ax)
        ax.set_xlabel("Country Status")
        ax.set_ylabel("Life Expectancy (years)")

    else:
        x_col = selected_plot.split(" vs ")[0].replace(" (1-19 years)", "")
        if x_col in df.columns:
            df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
            df_clean = df[[x_col, target]].dropna()
            sns.scatterplot(x=x_col, y=target, data=df_clean, ax=ax)
            ax.set_xlabel(x_col)
            ax.set_ylabel("Life Expectancy (years)")
        else:
            st.warning(f"🛑 Column '{x_col}' not found in the dataset.")

    ax.set_title(selected_plot)
    st.pyplot(fig)

# 🧠 Right: Explanation + Stats
with col2:
    st.markdown(f"### ℹ️ About the Plot")
    st.write(f"**{selected_plot}**")
    st.write(plot_options[selected_plot])

    if selected_plot == "Life Expectancy Distribution":
        x_col = target
    elif selected_plot == "GDP Histogram":
        x_col = "GDP"
    elif selected_plot == "Boxplot of Life Expectancy by Status":
        x_col = None
    else:
        x_col = selected_plot.split(" vs ")[0].strip()

    if x_col in df.columns:
        col_data = pd.to_numeric(df[x_col], errors='coerce').dropna()
        if not col_data.empty:
            st.markdown(f"""
            #### 📌 Stats for **{x_col}**:
            - **Mean:** {col_data.mean():.2f}
            - **Median:** {col_data.median():.2f}
            - **Std Dev:** {col_data.std():.2f}
            - **Min:** {col_data.min():.2f}
            - **Max:** {col_data.max():.2f}
            - **Missing Values:** {df[x_col].isnull().sum()}
            """)

