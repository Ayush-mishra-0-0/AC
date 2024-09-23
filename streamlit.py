import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from scipy import stats
from datetime import datetime, timedelta
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from statsmodels.tsa import seasonal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from mlxtend.frequent_patterns import apriori, association_rules

# Set page configuration
st.set_page_config(page_title="Sales Analysis Dashboard", layout="wide")

# Function to load data
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    return df

# Load data
df = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Customer Behavior", "Sales Trends", "SKU Performance", "Time-based Analysis", "SKU Analysis", "Customer Segment Analysis", "Order Size Analysis"])

# Main content
st.title("Sales Analysis Dashboard")

if page == "Data Overview":
    st.header("Data Overview")
    
    # Display first few rows
    st.subheader("Sample Data")
    st.dataframe(df.head())
    
    # Missing values
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values)
    
    # Duplicate rows
    st.subheader("Duplicate Rows")
    duplicate_rows = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicate_rows}")
    
    # Data types
    st.subheader("Data Types")
    st.write(df.dtypes)
    
    # Unique values
    st.subheader("Unique Values in Each Column")
    for col in df.columns:
        st.write(f"{col}: {df[col].nunique()} unique values")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Distribution of GMV
    st.subheader("Distribution of GMV")
    fig = px.histogram(df, x="placed_gmv", nbins=50, title="Distribution of GMV")
    st.plotly_chart(fig)

elif page == "Customer Behavior":
    st.header("Customer Behavior Analysis")
    
    # Purchase Frequency
    st.subheader("Purchase Frequency")
    purchase_frequency = df.groupby('user_id')['order_id'].count().reset_index().rename(columns={'order_id': 'order_count'})
    purchase_frequency = purchase_frequency.sort_values(by='order_count', ascending=False)
    st.write(purchase_frequency.head())
    st.write(purchase_frequency['order_count'].describe())
    
    fig = px.histogram(purchase_frequency, x="order_count", nbins=50, title="Distribution of Purchase Frequency")
    st.plotly_chart(fig)
    
    # Top Customers by GMV
    st.subheader("Top Customers by GMV")
    customer_gmv = df.groupby('user_id')['placed_gmv'].sum().reset_index()
    top_customers_gmv = customer_gmv.sort_values(by='placed_gmv', ascending=False)
    st.write(top_customers_gmv.head())
    st.write(top_customers_gmv['placed_gmv'].describe())
    
    fig = px.bar(top_customers_gmv.head(20), x='user_id', y='placed_gmv', title="Top 20 Customers by GMV")
    st.plotly_chart(fig)
    
    # RFM Analysis
    st.subheader("RFM Analysis")
    
    most_recent_date = df['order_date'].max()
    recency_df = df.groupby('user_id').agg({'order_date': lambda x: (most_recent_date - x.max()).days}).reset_index()
    recency_df = recency_df.rename(columns={'order_date': 'recency'})
    frequency_df = purchase_frequency.rename(columns={'order_count': 'frequency'})
    monetary_df = df.groupby('user_id').agg({'placed_gmv': 'sum'}).reset_index().rename(columns={'placed_gmv': 'monetary'})
    rfm_df = recency_df.merge(frequency_df, on='user_id').merge(monetary_df, on='user_id')
    
    rfm_df['recency_score'] = pd.qcut(rfm_df['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm_df['frequency_score'] = pd.qcut(rfm_df['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm_df['monetary_score'] = pd.qcut(rfm_df['monetary'], 5, labels=[1, 2, 3, 4, 5])
    rfm_df['RFM_score'] = rfm_df['recency_score'].astype(str) + rfm_df['frequency_score'].astype(str) + rfm_df['monetary_score'].astype(str)
    
    st.write(rfm_df.head())
    
    fig = px.scatter_3d(rfm_df, x='recency', y='frequency', z='monetary', color='RFM_score', title="3D RFM Analysis")
    st.plotly_chart(fig)
    
    # Customer Retention and Churn
    st.subheader("Customer Retention and Churn")
    
    repeat_customers = df.groupby('user_id')['order_id'].nunique().reset_index()
    repeat_customers['is_repeat'] = repeat_customers['order_id'] > 1
    retention_rate = repeat_customers['is_repeat'].mean() * 100
    st.write(f"Customer Retention Rate: {retention_rate:.2f}%")
    
    churn_threshold_days = 90
    churned_customers = rfm_df[rfm_df['recency'] > churn_threshold_days]
    churn_rate = len(churned_customers) / len(rfm_df) * 100
    st.write(f"Customer Churn Rate: {churn_rate:.2f}%")

    fig = go.Figure(data=[go.Pie(labels=['Retained', 'Churned'], values=[100-churn_rate, churn_rate])])
    fig.update_layout(title_text="Customer Retention vs Churn")
    st.plotly_chart(fig)

elif page == "Sales Trends":
    st.header("Sales Trends Analysis")
    
    # Daily Sales Trend
    st.subheader("Daily Sales Trend")
    daily_sales = df.groupby('order_date')['placed_gmv'].sum().reset_index()
    daily_sales['7_day_MA'] = daily_sales['placed_gmv'].rolling(window=7).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_sales['order_date'], y=daily_sales['placed_gmv'], mode='lines', name='Daily Sales'))
    fig.add_trace(go.Scatter(x=daily_sales['order_date'], y=daily_sales['7_day_MA'], mode='lines', name='7-day Moving Average'))
    fig.update_layout(title='Daily Sales Trend with 7-day Moving Average', xaxis_title='Date', yaxis_title='Total GMV')
    st.plotly_chart(fig)
    
    # Weekly Sales Trend
    st.subheader("Weekly Sales Trend")
    df['Week'] = df['order_date'].dt.to_period('W')
    weekly_sales = df.groupby('Week')['placed_gmv'].sum().reset_index()
    weekly_sales['Week'] = weekly_sales['Week'].astype(str)
    
    fig = px.line(weekly_sales, x='Week', y='placed_gmv', title='Weekly Sales Trend')
    st.plotly_chart(fig)
    
    # Monthly Sales Trend
    st.subheader("Monthly Sales Trend")
    df['Month'] = df['order_date'].dt.to_period('M')
    monthly_sales = df.groupby('Month')['placed_gmv'].sum().reset_index()
    monthly_sales['Month'] = monthly_sales['Month'].astype(str)
    
    fig = px.line(monthly_sales, x='Month', y='placed_gmv', title='Monthly Sales Trend')
    st.plotly_chart(fig)
    
    # Year-over-Year Growth
    st.subheader("Year-over-Year Growth")
    df['Year'] = df['order_date'].dt.year
    yearly_sales = df.groupby('Year')['placed_gmv'].sum().reset_index()
    yearly_sales['YoY_Growth'] = yearly_sales['placed_gmv'].pct_change()
    
    fig = px.bar(yearly_sales, x='Year', y='YoY_Growth', text='YoY_Growth')
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig.update_layout(title='Year-over-Year Sales Growth', yaxis_title='Growth Rate')
    st.plotly_chart(fig)
    
    # Average Order Value (AOV) Trends
    st.subheader("Average Order Value (AOV) Trends")
    df['AOV'] = df.groupby('order_id')['placed_gmv'].transform('sum')
    daily_aov = df.groupby('order_date')['AOV'].mean().reset_index()
    daily_aov['7_day_MA'] = daily_aov['AOV'].rolling(window=7).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_aov['order_date'], y=daily_aov['AOV'], mode='lines', name='Daily AOV'))
    fig.add_trace(go.Scatter(x=daily_aov['order_date'], y=daily_aov['7_day_MA'], mode='lines', name='7-day Moving Average'))
    fig.update_layout(title='Daily Average Order Value (AOV) Trend with 7-day Moving Average', xaxis_title='Date', yaxis_title='AOV')
    st.plotly_chart(fig)
elif page == "SKU Performance":
    st.header("SKU Performance Analysis")
    
    # Top-selling SKUs
    st.subheader("Top-selling SKUs")
    metric = st.selectbox("Select metric", ["quantity", "placed_gmv"])
    top_n = st.slider("Select number of top SKUs", 5, 20, 10)
    
    top_skus = df.groupby('sku_id')[metric].sum().sort_values(ascending=False).head(top_n)
    
    fig = px.bar(top_skus, x=top_skus.index, y=top_skus.values, title=f'Top {top_n} SKUs by {metric}')
    fig.update_xaxes(title='SKU ID')
    fig.update_yaxes(title=metric)
    st.plotly_chart(fig)
    
    st.write(top_skus)
    
    # SKU Diversity
    st.subheader("SKU Diversity in Orders")
    df['unique_skus'] = df.groupby('order_id')['sku_id'].transform('nunique')
    
    fig = px.bar(df['unique_skus'].value_counts().sort_index(), title='Distribution of Unique SKUs per Order')
    fig.update_xaxes(title='Number of Unique SKUs')
    fig.update_yaxes(title='Number of Orders')
    st.plotly_chart(fig)
    
    st.write("SKU Diversity Statistics:")
    st.write(df['unique_skus'].describe())
    
    # ABC Analysis
    st.subheader("ABC Analysis")
    
    def abc_analysis(df, value_column):
        sku_total = df.groupby('sku_id')[value_column].sum().sort_values(ascending=False)
        sku_total_cum = sku_total.cumsum() / sku_total.sum()
        
        sku_categories = pd.cut(sku_total_cum, bins=[0, 0.8, 0.95, 1], labels=['A', 'B', 'C'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(sku_total_cum))), y=sku_total_cum, mode='lines'))
        fig.add_shape(type="line", x0=0, y0=0.8, x1=len(sku_total_cum), y1=0.8, line=dict(color="Red", width=2, dash="dash"))
        fig.add_shape(type="line", x0=0, y0=0.95, x1=len(sku_total_cum), y1=0.95, line=dict(color="Red", width=2, dash="dash"))
        fig.update_layout(title='ABC Analysis Cumulative Curve', xaxis_title='Number of SKUs', 
                          yaxis_title=f'Cumulative Percentage of {value_column}')
        fig.add_annotation(x=len(sku_total_cum) * 0.05, y=0.82, text="Category A", showarrow=False)
        fig.add_annotation(x=len(sku_total_cum) * 0.5, y=0.96, text="Category B", showarrow=False)
        fig.add_annotation(x=len(sku_total_cum) * 0.8, y=0.97, text="Category C", showarrow=False)
        st.plotly_chart(fig)
        
        category_counts = sku_categories.value_counts().sort_index()
        st.write("ABC Analysis Results:")
        st.write(category_counts)
        st.write("Percentage of SKUs in each category:")
        st.write(category_counts / len(sku_categories) * 100)
        
        return sku_categories
    
    sku_categories = abc_analysis(df, 'placed_gmv')
    
    # SKU Correlation Analysis
    st.subheader("SKU Correlation Analysis")
    
    top_n_corr = st.slider("Select number of top SKUs for correlation", 5, 20, 10)
    top_skus = df.groupby('sku_id')['placed_gmv'].sum().nlargest(top_n_corr).index
    sku_sales = df[df['sku_id'].isin(top_skus)].pivot_table(values='quantity', index='order_date', columns='sku_id', aggfunc='sum').fillna(0)
    
    corr_matrix = sku_sales.corr()
    
    fig = px.imshow(corr_matrix, title=f'Correlation Heatmap of Top {top_n_corr} SKUs')
    st.plotly_chart(fig)
    
    # Market Basket Analysis
    st.subheader("Market Basket Analysis")

    # Prepare data for market basket analysis
    basket = df.groupby(['order_id', 'sku_id'])['quantity'].sum().unstack().fillna(0)
    basket_sets = (basket > 0).astype(int)

    min_support = st.slider("Minimum support", 0.01, 0.1, 0.02, 0.01)

    # Generate frequent itemsets
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # Display the association rules
    st.write("Generated Association Rules")
    st.dataframe(rules)

    # Filter rules by confidence and lift
    st.subheader("Filter Association Rules by Confidence and Lift")
    min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.01)
    min_lift = st.slider("Minimum Lift", 0.0, 10.0, 1.0, 0.1)

    filtered_rules = rules[(rules['confidence'] >= min_confidence) & (rules['lift'] >= min_lift)]

    st.write(f"Filtered Rules (Confidence >= {min_confidence} and Lift >= {min_lift})")
    st.dataframe(filtered_rules)

    # Plot Lift vs Confidence Scatter plot for filtered rules using Plotly
    st.subheader("Association Rules Visualization: Lift vs Confidence")

    if not filtered_rules.empty:
        fig = px.scatter(filtered_rules, x='confidence', y='lift',
                        size='support', color='lift',
                        labels={'confidence': 'Confidence', 'lift': 'Lift'},
                        title='Lift vs Confidence (Bubble Size = Support)')
        st.plotly_chart(fig)
    else:
        st.write("No rules found for the selected confidence and lift thresholds.")

    # Show top 5 association rules with the highest lift
    st.subheader("Top 5 Rules by Lift")
    top_5_rules = rules.sort_values(by='lift', ascending=False).head()
    st.write(top_5_rules)

elif page == "Time-based Analysis":
    st.header("Time-based Analysis")
    
    # Ensure your date column is in datetime format
    df['order_date'] = pd.to_datetime(df['order_date'])  # Replace 'order_date' with the actual date column name

    # Create a 'month' column in string format
    df['month'] = df['order_date'].dt.strftime('%Y-%m')  # Format the month as 'YYYY-MM'
    
    # Top 5 SKUs
    top_5_skus = df.groupby('sku_id')['quantity'].sum().nlargest(5).index
    
    # Monthly Sales for Top 5 SKUs
    sku_monthly_sales = df[df['sku_id'].isin(top_5_skus)].groupby(['month', 'sku_id'])['quantity'].sum().unstack()
    
    # Line Plot: Monthly Sales Trends for Top 5 SKUs
    fig1 = px.line(sku_monthly_sales, markers=True, labels={'value': 'Quantity Sold', 'variable': 'SKU ID'},
                   title='Monthly Sales Trends for Top 5 SKUs')
    fig1.update_layout(xaxis_title="Month", yaxis_title="Quantity Sold")
    st.plotly_chart(fig1)

    # Total Monthly GMV
    monthly_gmv = df.groupby('month')['placed_gmv'].sum()
    
    # Line Plot: Total Monthly GMV
    fig2 = px.line(monthly_gmv, x=monthly_gmv.index, y=monthly_gmv.values, labels={'x': 'Month', 'y': 'Total GMV'},
                   title='Total Monthly GMV')
    fig2.update_layout(xaxis_title="Month", yaxis_title="Total GMV")
    st.plotly_chart(fig2)

    # SKU Quantity Breakdown (Stacked Area Plot)
    fig3 = px.area(sku_monthly_sales, title='SKU Quantity Breakdown Over Time', 
                   labels={'value': 'Quantity Sold', 'month': 'Month', 'sku_id': 'SKU ID'},
                   markers=True)
    fig3.update_layout(xaxis_title="Month", yaxis_title="Quantity Sold")
    st.plotly_chart(fig3)

    # Monthly Average Order Value
    monthly_avg_order_value = df.groupby('month')['placed_gmv'].mean()
    
    # Line Plot: Monthly Average Order Value
    fig4 = px.line(monthly_avg_order_value, x=monthly_avg_order_value.index, y=monthly_avg_order_value.values,
                   labels={'x': 'Month', 'y': 'Average Order Value'},
                   title='Monthly Average Order Value')
    fig4.update_layout(xaxis_title="Month", yaxis_title="Average Order Value")
    st.plotly_chart(fig4)



# SKU Correlation, Performance, Seasonality, and Price Analysis combined
elif page == "SKU Analysis":
    st.header("SKU Analysis")
    
    # SKU Correlation Analysis
    st.subheader("SKU Pairs-Correlation Analysis")
    
    sku_correlation = df.pivot_table(values='quantity', index='order_id', columns='sku_id', aggfunc='sum', fill_value=0)
    correlation_matrix = sku_correlation.corr()
    
    fig = px.imshow(correlation_matrix, color_continuous_scale='RdBu', zmin=-1, zmax=1, title="SKU Correlation Heatmap")
    st.plotly_chart(fig)
    
    corr_unstack = correlation_matrix.unstack()
    top_corr = corr_unstack[corr_unstack < 1].nlargest(10)
    st.subheader("Top 10 Correlated SKU Pairs:")
    st.write(top_corr)
    
    # SKU Performance over time
    st.subheader("SKU Performance over Time")
    
    yearly_sku_performance = df.groupby(['year', 'sku_id'])['placed_gmv'].sum().unstack()
    growth_rates = yearly_sku_performance.pct_change().mean().sort_values(ascending=False)
    
    fig = px.bar(growth_rates.head(10), labels={'value': 'Average Yearly Growth Rate', 'index': 'SKU ID'},
                 title='Top 10 SKUs by Average Yearly Growth Rate')
    fig.update_layout(xaxis_title="SKU ID", yaxis_title="Average Yearly Growth Rate")
    st.plotly_chart(fig)
    
    st.subheader("Top 10 SKUs by Average Yearly Growth Rate:")
    st.write(growth_rates.head(10))
    
    # SKU Seasonality Analysis
    st.subheader("SKU Seasonality")
    
    def detect_seasonality(series):
        decomposition = seasonal.seasonal_decompose(series, model='additive', period=12)
        return decomposition.seasonal.abs().mean()
    
    sku_monthly_sales = df.groupby(['month', 'sku_id'])['quantity'].sum().unstack()
    sku_seasonality = sku_monthly_sales.apply(detect_seasonality).sort_values(ascending=False)
    
    fig = px.bar(sku_seasonality.head(10), labels={'value': 'Seasonality Score', 'index': 'SKU ID'},
                 title='Top 10 SKUs by Seasonality Score')
    fig.update_layout(xaxis_title="SKU ID", yaxis_title="Seasonality Score")
    st.plotly_chart(fig)
    
    st.subheader("Top 10 SKUs by Seasonality Score:")
    st.write(sku_seasonality.head(10))
    
    # SKU Price Analysis
    st.subheader("SKU Price Analysis")
    
    sku_price_stats = df.groupby('sku_id')['unit_price'].agg(['mean', 'min', 'max', 'std'])
    sku_price_stats['coefficient_of_variation'] = sku_price_stats['std'] / sku_price_stats['mean']
    
    fig = px.bar(sku_price_stats.sort_values('coefficient_of_variation', ascending=False).head(10)['coefficient_of_variation'],
                 labels={'value': 'Coefficient of Variation', 'index': 'SKU ID'}, 
                 title='Top 10 SKUs by Price Variability')
    fig.update_layout(xaxis_title="SKU ID", yaxis_title="Coefficient of Variation")
    st.plotly_chart(fig)
    
    st.subheader("Top 10 SKUs by Price Variability:")
    st.write(sku_price_stats.sort_values('coefficient_of_variation', ascending=False).head(10))

elif page == "Customer Segment Analysis":
    st.header("SKU Performance by Customer Segment")
    
    # Categorize customers into segments based on their total GMV
    df['customer_segment'] = pd.qcut(df.groupby('user_id')['placed_gmv'].sum(), q=3, labels=['Low Value', 'Medium Value', 'High Value'])
    
    # Calculate SKU performance by customer segment
    segment_sku_performance = df.groupby(['customer_segment', 'sku_id'])['placed_gmv'].sum().unstack()
    segment_sku_performance = segment_sku_performance.div(segment_sku_performance.sum(axis=1), axis=0)
    
    # Use Plotly for the heatmap
    fig = px.imshow(segment_sku_performance, color_continuous_scale='YlOrRd', labels={'color':'Performance'}, 
                    title="SKU Performance by Customer Segment")
    st.plotly_chart(fig)
    
    # Display top 5 SKUs for each customer segment
    st.subheader("Top 5 SKUs for each Customer Segment:")
    for segment in segment_sku_performance.index:
        st.write(f"\n{segment}:")
        top_5_skus = segment_sku_performance.loc[segment].nlargest(5)
        st.write(top_5_skus)


elif page == "Order Size Analysis":
    st.header("Patterns in Multi-Item Orders")

    # a. Analyze Percentage of Multi-Item Orders
    order_item_count = df.groupby('order_id')['sku_id'].nunique().reset_index(name='unique_items')
    multi_item_percentage = (order_item_count['unique_items'] > 1).mean() * 100
    st.write(f"**Percentage of multi-item orders:** {multi_item_percentage:.2f}%")

    # b. Examine GMV Contribution from Multi-Item Orders
    order_summary = df.groupby('order_id').agg({
        'sku_id': 'nunique',
        'placed_gmv': 'sum'
    }).reset_index()

    order_summary['is_multi_item'] = order_summary['sku_id'] > 1
    gmv_contribution = order_summary.groupby('is_multi_item')['placed_gmv'].sum()
    total_gmv = gmv_contribution.sum()
    gmv_percentage = (gmv_contribution / total_gmv * 100).reset_index()
    gmv_percentage.columns = ['is_multi_item', 'gmv_percentage']

    st.subheader("GMV Contribution")
    st.write(gmv_percentage)

    # Visualization of GMV Contribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='is_multi_item', y='gmv_percentage', data=gmv_percentage, ax=ax)
    ax.set_title('GMV Contribution: Single-Item vs Multi-Item Orders')
    ax.set_xlabel('Order Type')
    ax.set_ylabel('Percentage of Total GMV')
    ax.set_xticklabels(['Single-Item', 'Multi-Item'])
    st.pyplot(fig)

    # Filter multi-item orders
    multi_item_orders = order_item_count[order_item_count['unique_items'] > 1]['order_id']
    multi_item_df = df[df['order_id'].isin(multi_item_orders)]

    # Function to generate SKU pairs
    def get_sku_pairs(group):
        skus = group['sku_id'].unique()
        skus_sorted = sorted(skus)
        return list(combinations(skus_sorted, 2))

    # Apply function to each order
    sku_pairs = multi_item_df.groupby('order_id').apply(get_sku_pairs)

    # Flatten the list of SKU pairs and count occurrences
    sku_pair_counts = pd.Series([pair for pairs in sku_pairs for pair in pairs]).value_counts().head(10)

    st.subheader("Top 10 SKU pairs in Multi-Item Orders")
    st.write(sku_pair_counts)

    # Visualization of top SKU pairs
    fig, ax = plt.subplots(figsize=(12, 6))
    sku_pair_counts.plot(kind='bar', ax=ax)
    ax.set_title('Top 10 SKU Pairs in Multi-Item Orders')
    ax.set_xlabel('SKU Pair')
    ax.set_ylabel('Frequency')
    ax.set_xticklabels(sku_pair_counts.index.astype(str), rotation=45, ha='right')
    st.pyplot(fig)

    # 4. Analyze Order Value per Item
    df['value_per_item'] = df['placed_gmv'] / df['quantity']

    # Overall statistics of value per item
    value_per_item_stats = df['value_per_item'].describe()
    st.subheader("Value per Item Statistics")
    st.write(value_per_item_stats)

    # Top 10 SKUs by average value per item
    top_value_skus = df.groupby('sku_id')['value_per_item'].mean().sort_values(ascending=False).head(10)
    st.subheader("Top 10 SKUs by Average Value per Item")
    st.write(top_value_skus)

    # Visualization of value per item distribution
    st.subheader("Distribution of Value per Item")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(df['value_per_item'], kde=True, ax=ax)
    ax.set_title('Distribution of Value per Item')
    ax.set_xlabel('Value per Item')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Comparison of value per item in single-item vs multi-item orders
    df['is_multi_item'] = df['order_id'].isin(multi_item_orders)
    value_per_item_comparison = df.groupby('is_multi_item')['value_per_item'].mean()

    st.subheader("Average Value per Item: Single-Item vs Multi-Item Orders")
    st.write(value_per_item_comparison)

    # Visualization of value per item comparison
    st.subheader("Value per Item: Single-Item vs Multi-Item Orders")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='is_multi_item', y='value_per_item', data=df, ax=ax)
    ax.set_title('Value per Item: Single-Item vs Multi-Item Orders')
    ax.set_xlabel('Order Type')
    ax.set_ylabel('Value per Item')
    ax.set_xticklabels(['Single-Item', 'Multi-Item'])
    st.pyplot(fig)