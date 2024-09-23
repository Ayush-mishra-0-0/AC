import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from scipy import stats
from datetime import datetime, timedelta


# # Data Preperation and Overview

df= pd.read_csv('data.csv')

df.head(5)

missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Check for duplicate rows
duplicate_rows = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}")


# Check data types of each column
data_types = df.dtypes
print("Data types of columns:\n", data_types)

for col in df.columns:
    print(col, df[col].nunique())

# df['warehouse_name'].value_counts()

# Get basic summary statistics for numerical columns
summary_stats = df.describe()
summary_stats

# ## Convert order_date to datetime format

df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

print(df.dtypes)


# # Customer Behavior Analysis

# ## 1. Analysis costumer purchase Frequency

# Calculate the purchase frequency for each customer
purchase_frequency = df.groupby('user_id')['order_id'].count().reset_index().rename(columns={'order_id': 'order_count'})

# Sort customers by order count in descending order
purchase_frequency = purchase_frequency.sort_values(by='order_count', ascending=False)


purchase_frequency

purchase_frequency['order_count'].describe()

# ## 2. Identify Top Customers by GMV and Order Frequency

# Calculate the total GMV for each customer
customer_gmv = df.groupby('user_id')['placed_gmv'].sum().reset_index()

# Sort customers by GMV in descending order
top_customers_gmv = customer_gmv.sort_values(by='placed_gmv', ascending=False)

top_customers_gmv

top_customers_gmv['placed_gmv'].describe()

# Sort customers by the number of orders placed
top_customers_frequency = purchase_frequency.head()

# Display top 5 customers by order frequency
print(top_customers_frequency)


# ## 3. Segment Customers Based on RFM (Recency, Frequency, Monetary) Analysis
# RFM analysis is a powerful way to segment customers based on their behavior.
# 
# Recency: When the customer last made a purchase.
# Frequency: How often the customer made purchases.
# Monetary: How much money the customer has spent.

most_recent_date = df['order_date'].max()

# Calculate Recency
recency_df = df.groupby('user_id').agg({'order_date': lambda x: (most_recent_date - x.max()).days}).reset_index()
recency_df = recency_df.rename(columns={'order_date': 'recency'})

# Frequency: Number of orders placed by the customer
frequency_df = purchase_frequency.rename(columns={'order_count': 'frequency'})

# Monetary: Total GMV per customer
monetary_df = df.groupby('user_id').agg({'placed_gmv': 'sum'}).reset_index().rename(columns={'placed_gmv': 'monetary'})

# Merge recency, frequency, and monetary data
rfm_df = recency_df.merge(frequency_df, on='user_id').merge(monetary_df, on='user_id')

# Display the RFM table
print(rfm_df.head())


# b. RFM Segmentation
# We can create a segmentation based on RFM by assigning scores (e.g., 1-5) for each metric:

# RFM Scoring (1 to 5)
rfm_df['recency_score'] = pd.qcut(rfm_df['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm_df['frequency_score'] = pd.qcut(rfm_df['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm_df['monetary_score'] = pd.qcut(rfm_df['monetary'], 5, labels=[1, 2, 3, 4, 5])

# Calculate the overall RFM score
rfm_df['RFM_score'] = rfm_df['recency_score'].astype(str) + rfm_df['frequency_score'].astype(str) + rfm_df['monetary_score'].astype(str)

rfm_df.head()

# ## 4. Examine Customer Retention and Churn Rates
# For customer retention, we need to analyze how many customers are placing repeat orders and identify potential churn (customers who haven't ordered in a long time).

# a. Customer Retention Rate

# Identify repeat customers
repeat_customers = df.groupby('user_id')['order_id'].nunique().reset_index()
repeat_customers['is_repeat'] = repeat_customers['order_id'] > 1

# Calculate retention rate
retention_rate = repeat_customers['is_repeat'].mean() * 100
print(f"Customer Retention Rate: {retention_rate:.2f}%")


# b. Customer Churn Rate
# You can calculate churn based on customers who haven't made a purchase recently, compared to the total customer base.

# Assume customers who have not purchased in the last 90 days are considered "churned"
churn_threshold_days = 90
churned_customers = rfm_df[rfm_df['recency'] > churn_threshold_days]

# Churn rate calculation
churn_rate = len(churned_customers) / len(rfm_df) * 100
print(f"Customer Churn Rate: {churn_rate:.2f}%")


# # Sales Trends Analysis

# ### 1. Analyze Daily, Weekly, and Monthly Sales Trends  
# 
# a. Daily Sales Trends
# We will group the dataset by order_date to analyze daily trends.

df.head()

# Assuming the data is already loaded into a DataFrame called 'df'
# and 'order_date' is converted to datetime

# Daily sales trend
daily_sales = df.groupby('order_date')['placed_gmv'].sum().reset_index()
daily_sales['7_day_MA'] = daily_sales['placed_gmv'].rolling(window=7).mean()

plt.figure(figsize=(15, 6))
plt.plot(daily_sales['order_date'], daily_sales['placed_gmv'], label='Daily Sales')
plt.plot(daily_sales['order_date'], daily_sales['7_day_MA'], label='7-day Moving Average', color='red')
plt.title('Daily Sales Trend with 7-day Moving Average')
plt.xlabel('Date')
plt.ylabel('Total GMV')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Weekly sales trend
df['Week'] = df['order_date'].dt.to_period('W')
weekly_sales = df.groupby('Week')['placed_gmv'].sum().reset_index()
weekly_sales['Week'] = weekly_sales['Week'].astype(str)

plt.figure(figsize=(15, 6))
plt.plot(weekly_sales['Week'], weekly_sales['placed_gmv'])
plt.title('Weekly Sales Trend')
plt.xlabel('Week')
plt.ylabel('Total GMV')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Monthly sales trend
df['Month'] = df['order_date'].dt.to_period('M')
monthly_sales = df.groupby('Month')['placed_gmv'].sum().reset_index()
monthly_sales['Month'] = monthly_sales['Month'].astype(str)

plt.figure(figsize=(15, 6))
plt.plot(monthly_sales['Month'], monthly_sales['placed_gmv'])
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total GMV')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Print summary statistics
print("Daily Sales Summary:")
print(daily_sales['placed_gmv'].describe())
print("\nWeekly Sales Summary:")
print(weekly_sales['placed_gmv'].describe())
print("\nMonthly Sales Summary:")
print(monthly_sales['placed_gmv'].describe())

# ## 2. Identify Peak Sales Periods and Seasonality
# You can use the monthly or weekly sales trends to identify periods of high sales.

# 2. Identify peak sales periods and seasonality
# Peak sales periods
peak_daily = daily_sales.nlargest(10, 'placed_gmv')
peak_weekly = weekly_sales.nlargest(10, 'placed_gmv')
peak_monthly = monthly_sales.nlargest(10, 'placed_gmv')

print("Top 10 Peak Sales Days:")
print(peak_daily[['order_date', 'placed_gmv']])
print("\nTop 10 Peak Sales Weeks:")
print(peak_weekly[['Week', 'placed_gmv']])
print("\nTop 10 Peak Sales Months:")
print(peak_monthly[['Month', 'placed_gmv']])

# Seasonality analysis
df['Month'] = df['order_date'].dt.month
df['Year'] = df['order_date'].dt.year

monthly_seasonal = df.groupby(['Year', 'Month'])['placed_gmv'].sum().reset_index()
monthly_seasonal_pivot = monthly_seasonal.pivot(index='Month', columns='Year', values='placed_gmv')

plt.figure(figsize=(12, 6))
sns.heatmap(monthly_seasonal_pivot, cmap='YlOrRd', annot=True, fmt='.0f')
plt.title('Monthly Sales Heatmap')
plt.xlabel('Year')
plt.ylabel('Month')
plt.tight_layout()
plt.show()

# Calculate month-over-month growth
monthly_seasonal['MoM_Growth'] = monthly_seasonal.groupby('Year')['placed_gmv'].pct_change()

plt.figure(figsize=(12, 6))
for year in monthly_seasonal['Year'].unique():
    year_data = monthly_seasonal[monthly_seasonal['Year'] == year]
    plt.plot(year_data['Month'], year_data['MoM_Growth'], label=str(year))

plt.title('Month-over-Month Growth by Year')
plt.xlabel('Month')
plt.ylabel('Growth Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Identify top 5 peak sales periods
peak_sales_periods = monthly_sales.sort_values(by='placed_gmv', ascending=False).head(5)
print("Top 5 Peak Sales Periods:\n", peak_sales_periods)

# ## 3. Calculate and Visualize Year-over-Year (YoY) Growth
# To calculate year-over-year growth, we need to compare sales for each year.

df.head()

# 3. Calculate and visualize year-over-year growth
yearly_sales = df.groupby('Year')['placed_gmv'].sum().reset_index()
yearly_sales['YoY_Growth'] = yearly_sales['placed_gmv'].pct_change()

plt.figure(figsize=(10, 6))
plt.bar(yearly_sales['Year'], yearly_sales['YoY_Growth'])
plt.title('Year-over-Year Sales Growth')
plt.xlabel('Year')
plt.ylabel('Growth Rate')
plt.axhline(y=0, color='r', linestyle='-')
for i, v in enumerate(yearly_sales['YoY_Growth']):
    plt.text(yearly_sales['Year'][i], v, f'{v:.2%}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

print("Yearly Sales and Growth:")
print(yearly_sales)

# Calculate and visualize cumulative year-over-year growth
yearly_sales['Cumulative_Growth'] = (1 + yearly_sales['YoY_Growth']).cumprod() - 1

plt.figure(figsize=(10, 6))
plt.plot(yearly_sales['Year'], yearly_sales['Cumulative_Growth'], marker='o')
plt.title('Cumulative Year-over-Year Sales Growth')
plt.xlabel('Year')
plt.ylabel('Cumulative Growth Rate')
plt.axhline(y=0, color='r', linestyle='-')
for i, v in enumerate(yearly_sales['Cumulative_Growth']):
    plt.text(yearly_sales['Year'][i], v, f'{v:.2%}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

print("Yearly Sales and Cumulative Growth:")
print(yearly_sales)


# Extract year from order_date
df['year'] = df['order_date'].dt.year

# Group by year to calculate total GMV per year
yearly_sales = df.groupby('year')['placed_gmv'].sum().reset_index()

# Calculate Year-over-Year Growth
yearly_sales['YoY_Growth'] = yearly_sales['placed_gmv'].pct_change() * 100

# Plotting Year-over-Year Growth
plt.figure(figsize=(10, 6))
sns.barplot(x='year', y='YoY_Growth', data=yearly_sales)
plt.title('Year-over-Year Growth')
plt.xlabel('Year')
plt.ylabel('YoY Growth (%)')
plt.show()

# Display the Year-over-Year growth table
print(yearly_sales)

# ----------------------------------------------------------------------------------------------------------------------------------
# ## 4. Analyze Average Order Value (AOV) Trends
# We can calculate AOV (Average Order Value) by dividing the total GMV by the number of orders for a given time period.


# 4. Analyze average order value (AOV) trends
# Calculate AOV
df['AOV'] = df.groupby('order_id')['placed_gmv'].transform('sum')

# Daily AOV trend
daily_aov = df.groupby('order_date')['AOV'].mean().reset_index()
daily_aov['7_day_MA'] = daily_aov['AOV'].rolling(window=7).mean()

plt.figure(figsize=(15, 6))
plt.plot(daily_aov['order_date'], daily_aov['AOV'], label='Daily AOV')
plt.plot(daily_aov['order_date'], daily_aov['7_day_MA'], label='7-day Moving Average', color='red')
plt.title('Daily Average Order Value (AOV) Trend with 7-day Moving Average')
plt.xlabel('Date')
plt.ylabel('AOV')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Monthly AOV trend
monthly_aov = df.groupby('Month')['AOV'].mean().reset_index()
monthly_aov['Month'] = monthly_aov['Month'].astype(str)

plt.figure(figsize=(15, 6))
plt.plot(monthly_aov['Month'], monthly_aov['AOV'])
plt.title('Monthly Average Order Value (AOV) Trend')
plt.xlabel('Month')
plt.ylabel('AOV')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Yearly AOV trend
yearly_aov = df.groupby('Year')['AOV'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(yearly_aov['Year'], yearly_aov['AOV'])
plt.title('Yearly Average Order Value (AOV) Trend')
plt.xlabel('Year')
plt.ylabel('AOV')
for i, v in enumerate(yearly_aov['AOV']):
    plt.text(yearly_aov['Year'][i], v, f'{v:.2f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

print("AOV Summary Statistics:")
print(df['AOV'].describe())

# Calculate correlation between AOV and total daily sales
aov_sales_corr = daily_aov.merge(daily_sales, on='order_date')
correlation = aov_sales_corr['AOV'].corr(aov_sales_corr['placed_gmv'])
print(f"\nCorrelation between daily AOV and total sales: {correlation:.2f}")

# # 4. SKU Performance Analysis
# 
# Identify top-selling SKUs by quantity and GMV
# Analyze SKU diversity in orders
# Perform ABC analysis to categorize SKUs
# Examine SKU purchase patterns and correlations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

# Identify top-selling SKUs by quantity and GMV:
# 
# Creates bar plots for top 10 SKUs by quantity and GMV
# Prints the top 10 SKUs for each metric
# 
# 
# Analyze SKU diversity in orders:
# 
# Calculates the number of unique SKUs per order
# Plots the distribution of unique SKUs per order
# Prints summary statistics for SKU diversity
# 
# 
# Perform ABC analysis to categorize SKUs:
# 
# Categorizes SKUs into A, B, and C categories based on cumulative GMV
# Plots the cumulative curve for ABC analysis
# Prints the number and percentage of SKUs in each category
# 
# Examine SKU purchase patterns and correlations:
# 4.1 Time-based analysis:
# - Plots monthly sales trends for top 5 SKUs
# 4.2 Correlation analysis:
# - Creates a correlation heatmap for SKUs
# - Prints top 10 correlated SKU pairs
# 4.3 Market Basket Analysis:
# - Performs association rule mining
# - Prints top 10 association rules
# 4.4 SKU Performance over time:
# - Calculates and plots top 10 SKUs by average yearly growth rate
# 4.5 SKU Seasonality:
# - Detects seasonality for each SKU
# - Plots and prints top 10 SKUs by seasonality score
# 4.6 SKU Price Analysis:
# - Calculates price variability for each SKU
# - Plots and prints top 10 SKUs by price variability
# 4.7 SKU Performance by Customer Segment:
# - Segments customers into low, medium, and high value
# - Creates a heatmap of SKU performance by customer segment
# - Prints top 5 SKUs for each customer segment

# ## 1. Identify top-selling SKUs by quantity and GMV


def analyze_top_skus(df, metric, top_n=10):
    top_skus = df.groupby('sku_id')[metric].sum().sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 6))
    top_skus.plot(kind='bar')
    plt.title(f'Top {top_n} SKUs by {metric}')
    plt.xlabel('SKU ID')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    print(f"\nTop {top_n} SKUs by {metric}:")
    print(top_skus)

analyze_top_skus(df, 'quantity')
analyze_top_skus(df, 'placed_gmv')

# ## 2. Analyze SKU diversity in orders


df['unique_skus'] = df.groupby('order_id')['sku_id'].transform('nunique')

plt.figure(figsize=(10, 6))
df['unique_skus'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Unique SKUs per Order')
plt.xlabel('Number of Unique SKUs')
plt.ylabel('Number of Orders')
plt.tight_layout()
plt.show()

print("\nSKU Diversity Statistics:")
print(df['unique_skus'].describe())

# ## 3. Perform ABC analysis to categorize SKUs


def abc_analysis(df, value_column):
    sku_total = df.groupby('sku_id')[value_column].sum().sort_values(ascending=False)
    sku_total_cum = sku_total.cumsum() / sku_total.sum()
    
    sku_categories = pd.cut(sku_total_cum, bins=[0, 0.8, 0.95, 1], labels=['A', 'B', 'C'])
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(sku_total_cum)), sku_total_cum, 'b-')
    plt.title('ABC Analysis Cumulative Curve')
    plt.xlabel('Number of SKUs')
    plt.ylabel('Cumulative Percentage of ' + value_column)
    plt.axhline(y=0.8, color='r', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.text(len(sku_total_cum)*0.05, 0.82, 'Category A', fontsize=12, color='red')
    plt.text(len(sku_total_cum)*0.5, 0.96, 'Category B', fontsize=12, color='red')
    plt.text(len(sku_total_cum)*0.8, 0.97, 'Category C', fontsize=12, color='red')
    plt.tight_layout()
    plt.show()
    
    category_counts = sku_categories.value_counts().sort_index()
    print("\nABC Analysis Results:")
    print(category_counts)
    print("\nPercentage of SKUs in each category:")
    print(category_counts / len(sku_categories) * 100)
    
    return sku_categories

sku_categories = abc_analysis(df, 'placed_gmv')

# ## 4. Another analysis

# ### 4.1 Time-based analysis

# 4. Examine SKU purchase patterns and correlations

df['order_date'] = pd.to_datetime(df['order_date'])
df['month'] = df['order_date'].dt.to_period('M')

top_5_skus = df.groupby('sku_id')['quantity'].sum().nlargest(5).index

sku_monthly_sales = df[df['sku_id'].isin(top_5_skus)].groupby(['month', 'sku_id'])['quantity'].sum().unstack()

plt.figure(figsize=(15, 8))
sku_monthly_sales.plot(marker='o')
plt.title('Monthly Sales Trends for Top 5 SKUs')
plt.xlabel('Month')
plt.ylabel('Quantity Sold')
plt.legend(title='SKU ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ### 4.2 Correlation analysis


sku_correlation = df.pivot_table(values='quantity', index='order_id', columns='sku_id', aggfunc='sum', fill_value=0)
correlation_matrix = sku_correlation.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('SKU Correlation Heatmap')
plt.tight_layout()
plt.show()

# Find top correlated pairs
corr_unstack = correlation_matrix.unstack()
top_corr = corr_unstack[corr_unstack < 1].nlargest(10)

print("\nTop 10 Correlated SKU Pairs:")
print(top_corr)


# ### 4.3 Market Basket Analysis


def one_hot_encode(x):
    return 1 if x > 0 else 0

basket = df.groupby(['order_id', 'sku_id'])['quantity'].sum().unstack().applymap(one_hot_encode)

frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

print("\nTop 10 Association Rules:")
print(rules.sort_values('lift', ascending=False).head(10))



# ### 4.4 SKU Performance over time

df['year'] = df['order_date'].dt.year
yearly_sku_performance = df.groupby(['year', 'sku_id'])['placed_gmv'].sum().unstack()

growth_rates = yearly_sku_performance.pct_change().mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
growth_rates.head(10).plot(kind='bar')
plt.title('Top 10 SKUs by Average Yearly Growth Rate')
plt.xlabel('SKU ID')
plt.ylabel('Average Yearly Growth Rate')
plt.tight_layout()
plt.show()

print("\nTop 10 SKUs by Average Yearly Growth Rate:")
print(growth_rates.head(10))


# ### 4.5 SKU Seasonality


def detect_seasonality(series):
    decomposition = stats.seasonal_decompose(series, model='additive', period=12)
    return decomposition.seasonal.abs().mean()

sku_seasonality = sku_monthly_sales.apply(detect_seasonality).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sku_seasonality.head(10).plot(kind='bar')
plt.title('Top 10 SKUs by Seasonality Score')
plt.xlabel('SKU ID')
plt.ylabel('Seasonality Score')
plt.tight_layout()
plt.show()

print("\nTop 10 SKUs by Seasonality Score:")
print(sku_seasonality.head(10))


# ### 4.6 SKU Price Analysis


df['unit_price'] = df['placed_gmv'] / df['quantity']

sku_price_stats = df.groupby('sku_id')['unit_price'].agg(['mean', 'min', 'max', 'std'])
sku_price_stats['coefficient_of_variation'] = sku_price_stats['std'] / sku_price_stats['mean']

plt.figure(figsize=(12, 6))
sku_price_stats.sort_values('coefficient_of_variation', ascending=False).head(10)['coefficient_of_variation'].plot(kind='bar')
plt.title('Top 10 SKUs by Price Variability')
plt.xlabel('SKU ID')
plt.ylabel('Coefficient of Variation')
plt.tight_layout()
plt.show()

print("\nTop 10 SKUs by Price Variability:")
print(sku_price_stats.sort_values('coefficient_of_variation', ascending=False).head(10))


# ### 4.7 SKU Performance by Customer Segment


df['customer_segment'] = pd.qcut(df.groupby('user_id')['placed_gmv'].sum(), q=3, labels=['Low Value', 'Medium Value', 'High Value'])

segment_sku_performance = df.groupby(['customer_segment', 'sku_id'])['placed_gmv'].sum().unstack()
segment_sku_performance = segment_sku_performance.div(segment_sku_performance.sum(axis=1), axis=0)

plt.figure(figsize=(15, 8))
sns.heatmap(segment_sku_performance, annot=False, cmap='YlOrRd')
plt.title('SKU Performance by Customer Segment')
plt.tight_layout()
plt.show()

print("\nTop 5 SKUs for each Customer Segment:")
for segment in segment_sku_performance.index:
    print(f"\n{segment}:")
    print(segment_sku_performance.loc[segment].nlargest(5))


order_sizes = df.groupby('order_id')['quantity'].sum().reset_index(name='items_per_order')

# Basic statistics
basic_stats = order_sizes['items_per_order'].describe()
print("Basic statistics of order sizes:")
print(basic_stats)

# Distribution of order sizes
size_distribution = order_sizes['items_per_order'].value_counts().sort_index()
size_distribution_pct = size_distribution / len(order_sizes) * 100

print("\nDistribution of order sizes:")
print(size_distribution_pct)

# Visualization of order size distribution
plt.figure(figsize=(12, 6))
sns.histplot(order_sizes['items_per_order'], kde=True)
plt.title('Distribution of Order Sizes')
plt.xlabel('Items per Order')
plt.ylabel('Frequency')
plt.show()

# 2. Examine the relationship between order size and GMV

# Calculate total GMV per order
order_gmv = df.groupby('order_id')['placed_gmv'].sum().reset_index(name='total_gmv')

# Merge order sizes and GMV
order_summary = pd.merge(order_sizes, order_gmv, on='order_id')

# Calculate average GMV per order size
avg_gmv_by_size = order_summary.groupby('items_per_order')['total_gmv'].mean().reset_index()
print("\nAverage GMV by order size:")
print(avg_gmv_by_size)

# Correlation between order size and GMV
correlation = order_summary['items_per_order'].corr(order_summary['total_gmv'])
print(f"\nCorrelation between order size and GMV: {correlation:.2f}")

# Visualization of relationship between order size and GMV
plt.figure(figsize=(12, 6))
sns.scatterplot(data=order_summary, x='items_per_order', y='total_gmv')
plt.title('Relationship between Order Size and GMV')
plt.xlabel('Items per Order')
plt.ylabel('Total GMV')
plt.show()

# 3. Identify patterns in multi-item orders

# Identify multi-item orders
multi_item_orders = df.groupby('order_id')['sku_id'].nunique().reset_index(name='unique_items')
multi_item_orders = multi_item_orders[multi_item_orders['unique_items'] > 1]['order_id']

# Most common product combinations in multi-item orders
multi_item_df = df[df['order_id'].isin(multi_item_orders)]
combinations = multi_item_df.groupby('order_id')['sku_id'].apply(lambda x: tuple(sorted(x)))
common_combinations = combinations.value_counts().head(10)

print("\nMost common product combinations in multi-item orders:")
print(common_combinations)

# Compare single-item vs multi-item orders
order_summary['is_multi_item'] = order_summary['items_per_order'] > 1
single_vs_multi = order_summary.groupby('is_multi_item').agg({
    'total_gmv': 'mean',
    'order_id': 'count'
}).rename(columns={'total_gmv': 'avg_gmv', 'order_id': 'order_count'})

print("\nComparison of single-item vs multi-item orders:")
print(single_vs_multi)

# Seasonal patterns in multi-item orders
df['order_date'] = pd.to_datetime(df['order_date'])
df['month'] = df['order_date'].dt.month

seasonal_patterns = df.groupby(['order_id', 'month']).agg({
    'quantity': 'sum',
    'placed_gmv': 'sum'
}).reset_index()

seasonal_patterns = seasonal_patterns.groupby('month').agg({
    'quantity': 'mean',
    'placed_gmv': 'mean'
}).reset_index()

print("\nSeasonal patterns in orders:")
print(seasonal_patterns)

# Visualization of seasonal patterns
plt.figure(figsize=(12, 6))
sns.lineplot(data=seasonal_patterns, x='month', y='quantity', label='Avg Items per Order')
sns.lineplot(data=seasonal_patterns, x='month', y='placed_gmv', label='Avg GMV')
plt.title('Seasonal Patterns in Orders')
plt.xlabel('Month')
plt.ylabel('Value')
plt.legend()
plt.show()

# ## 3. Identify Patterns in Multi-Item Orders

# ### a. Analyze Percentage of Multi-Item Orders

# a. Analyze Percentage of Multi-Item Orders
order_item_count = df.groupby('order_id')['sku_id'].nunique().reset_index(name='unique_items')
multi_item_percentage = (order_item_count['unique_items'] > 1).mean() * 100

print(f"\nPercentage of multi-item orders: {multi_item_percentage:.2f}%")



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

print("\nGMV Contribution:")
print(gmv_percentage)

# Visualization of GMV Contribution
plt.figure(figsize=(10, 6))
sns.barplot(x='is_multi_item', y='gmv_percentage', data=gmv_percentage)
plt.title('GMV Contribution: Single-Item vs Multi-Item Orders')
plt.xlabel('Order Type')
plt.ylabel('Percentage of Total GMV')
plt.xticks([0, 1], ['Single-Item', 'Multi-Item'])
plt.show()

# Filter multi-item orders
multi_item_orders = order_sizes[order_sizes['num_items'] > 1]['order_id']
multi_item_df = df[df['order_id'].isin(multi_item_orders)]

# Define the function to generate SKU pairs
def get_sku_pairs(group):
    skus = group['sku_id'].unique()
    skus_sorted = sorted(skus)  # Sort the SKUs before creating combinations
    return list(combinations(skus_sorted, 2))

# Apply the function to each order
sku_pairs = multi_item_df.groupby('order_id').apply(get_sku_pairs)

# Flatten the list of SKU pairs and count occurrences
sku_pair_counts = pd.Series([pair for pairs in sku_pairs for pair in pairs]).value_counts().head(10)

# Display top 10 SKU pairs in multi-item orders
print("\nTop 10 SKU pairs in multi-item orders:")
print(sku_pair_counts)

# Visualization of top SKU pairs
plt.figure(figsize=(12, 6))
sku_pair_counts.plot(kind='bar')
plt.title('Top 10 SKU Pairs in Multi-Item Orders')
plt.xlabel('SKU Pair')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# 4. Analyze Order Value per Item
df['value_per_item'] = df['placed_gmv'] / df['quantity']

# Overall statistics of value per item
value_per_item_stats = df['value_per_item'].describe()
print("\nValue per Item Statistics:")
print(value_per_item_stats)

# Top 10 SKUs by average value per item
top_value_skus = df.groupby('sku_id')['value_per_item'].mean().sort_values(ascending=False).head(10)
print("\nTop 10 SKUs by Average Value per Item:")
print(top_value_skus)

# Visualization of value per item distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['value_per_item'], kde=True)
plt.title('Distribution of Value per Item')
plt.xlabel('Value per Item')
plt.ylabel('Frequency')
plt.show()

# Comparison of value per item in single-item vs multi-item orders
df['is_multi_item'] = df['order_id'].isin(multi_item_orders)
value_per_item_comparison = df.groupby('is_multi_item')['value_per_item'].mean()

print("\nAverage Value per Item: Single-Item vs Multi-Item Orders")
print(value_per_item_comparison)

# Visualization of value per item comparison
plt.figure(figsize=(10, 6))
sns.boxplot(x='is_multi_item', y='value_per_item', data=df)
plt.title('Value per Item: Single-Item vs Multi-Item Orders')
plt.xlabel('Order Type')
plt.ylabel('Value per Item')
plt.xticks([0, 1], ['Single-Item', 'Multi-Item'])
plt.show()


