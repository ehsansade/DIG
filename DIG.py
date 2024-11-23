import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from fuzzywuzzy import process  # Ensure you install 'fuzzywuzzy'
import re
import numpy as np
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher


selection_state=0
question_counter=0
response_counter=0
# Define mapping of keywords to statistical functions
STAT_FUNCTIONS = {
    "mean": lambda x: x.mean(),
    "average": lambda x: x.mean(),
    "min": lambda x: x.min(),
    "minimum": lambda x: x.min(),
    "max": lambda x: x.max(),
    "maximum": lambda x: x.max(),
    "sum": lambda x: x.sum(),
    "total": lambda x: x.sum(),
    "count": lambda x: x.count(),
    "mode": lambda x: x.mode().iloc[0] if not x.mode().empty else "No mode found",
    "frequent": lambda x: x.mode().iloc[0] if not x.mode().empty else "No mode found",
    "median": lambda x: x.median(),
    "std": lambda x: x.std(),
    "variance": lambda x: x.var(),
    "unique": lambda x: x.unique(),
    "distinct": lambda x: x.unique()
    #r"\bcount\b.*\bunique\b": lambda x: x.value_counts()
    
}

# Define common chart types
CHART_TYPES = {
    "bar": "bar",
    "bar chart": "bar",
    "line": "line",
    "line chart": "line",
    "histogram": "histogram",
    "scatter": "scatter",
    "box":"box",
    "whisker":"box",
    "pie":"pie",
    "pie chart":"pie"
}

def find_most_similar_and_switch_general(list1, list2):
    
    def most_similar(target, choices):
        # Find the most similar string in choices to target
        similarity = [(item, SequenceMatcher(None, target, item).ratio()) for item in choices]
        return max(similarity, key=lambda x: x[1])[0]

    # Find the most similar items for each in list1
    matches = [most_similar(item, list2) for item in list1]
    
    # Find their indices in list2
    indices = [list2.index(match) for match in matches]

    # Check order of indices and rearrange list1 if necessary
    sorted_indices = sorted(range(len(indices)), key=lambda k: indices[k])
    sorted_list1 = [list1[i] for i in sorted_indices]
    
    return sorted_list1

 # Function to identify similar column names using fuzzy matching

     
def find_column_name(possible_names, available_columns):
    #col_name=[]
    score_f=0
    match_f=''
    for item in available_columns:  
        words = item.lower().split()
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words) - 1)]
        all_words=words+bigrams
        cols=possible_names
        
              
        for i in range(len(cols)):
         
            for w in all_words:
                score = fuzz.ratio(cols[i].lower(),w)
                if score>score_f:
                    score_f=score
                    match_f=item
    return match_f if score_f>60 else None        
# Function to extract column name from the question
def extract_column_name(question, columns):
    col_name=[]
    
    words = question.lower().split()
    bigrams = [' '.join(words[i:i+2]) for i in range(len(words) - 1)]
    all_words=words+bigrams
    cols=list(columns)
    final_score=[]
    for i in range(len(cols)):
        final_score.append(0)
        
    for i in range(len(cols)):
   
        for w in all_words:
            score = fuzz.ratio(cols[i].lower(),w)
            if score>final_score[i]:
                final_score[i]=score 
    max1_index=500   
    max2_index=500 
    if max(final_score)>70:           
        max1_index = final_score.index(max(final_score))
    # Temporarily replace the highest number with a very small value to find the second-highest
        temp = final_score[max1_index]
        final_score[max1_index] = float('-inf')
    if max(final_score)>70:
        max2_index = final_score.index(max(final_score))
    # Restore the original list
        final_score[max1_index] = temp 
    if max1_index<500:
        col_name.append(cols[max1_index])
    if max2_index<500:
        col_name.append(cols[max2_index])     
    if len(col_name)==2:
        col_name=find_most_similar_and_switch_general(col_name, all_words)
    return col_name

def extract_chart_type(question):
    """Extract chart types like 'bar', 'line' from the question."""
    for chart_name in CHART_TYPES:
        if re.search(rf"\b{chart_name}\b", question, re.IGNORECASE):
            return CHART_TYPES[chart_name]
    return None

def extract_date_range(question):
    """Extract date range from the question."""
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", question)
    return dates if len(dates) == 2 else None

def detect_summary_request(question):
    """Detect if the user is asking for summary information."""
    return bool(re.search(r"(rows|columns|data types|column names|correlation|heatmap|heat map)", question, re.IGNORECASE))

# Function to extract statistical operation from the question
def extract_stat_function(question):
    for keyword, func in STAT_FUNCTIONS.items():
        if re.search(rf"\b{keyword}\b", question, re.IGNORECASE):
            return func, keyword
    return None, None

# Function to detect if the question involves grouping
def extract_grouping_column(question, columns):
    # Look for phrases like "each" or "per", followed by a column name
    if 'group' in question:
        words_to_split = r'each|per|by'
    else:
        words_to_split = r'each|per'
        
    split_text = [s.strip() for s in re.split(words_to_split, question.lower())] 
    
    if len(split_text)>1:
        score_f=0
        match_f=''
        for col in list(columns):
        
            #match, score = process.extractOne(col, split_text[-1])
            score = fuzz.ratio(col.lower(),split_text[-1])
            if score>score_f and score>65:
                score_f=score
                match_f=col
        return match_f
    #for col in columns:
    #    if re.search(rf"(each|per|group\by)\s+{col}", question, re.IGNORECASE):
    #        return col
    return None

# Function to detect if the user asks about null values
def detect_null_check(question):
    return bool(re.search(r"\b(null|missing|na|nan|none)\b", question, re.IGNORECASE))


# Define categories for the dropdown
categories = [
    " ","Sales Analytics", "Customer/User Analytics", "Financial Analytics",
    "Marketing Analytics", "Service Analytics", "Game Analytics",
    "Healthcare Analytics", "Logistics Analytics", 
    "Social Media Analytics", "Risk Analytics"
]

# Initialize session state to store questions and responses
if 'history' not in st.session_state:
    st.session_state.history = []
# App title and file uploader
st.title("NLP-Powered Data Insights Generator")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Dropdown to select analytics category
#selected_category = st.selectbox("Select Analytics Category", categories)

# Display default analytics for the chosen category
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", data.head())
    
    ###
    col_names=data.columns
    date_cols=[]
    # Automatically detect and process any datetime columns
    for col in col_names:
    
        if 'date' in col.lower():
                # Convert 'order_date' to datetime
            data[col] = pd.to_datetime(data[col], errors='coerce')
            data[f'{col}_year'] = data[col].dt.year
            data[f'{col}_month'] = data[col].dt.month
            data[f'{col}_day'] = data[col].dt.day
            date_cols.append(col)
    ###
    
    # Ask the user what type of analytics they want
    analytics_type = st.selectbox(
        "What kind of analytics would you like to explore?",
        categories
    )
    st.write(f"### Analytics Report: {analytics_type}")

    # Example: Generate a simple visualization based on category
    if analytics_type == "Sales Analytics":
        

        # Sample column name mapping for potential variations
        column_mapping = {
            'order_date': ['order_date','Order Date', 'orderdate', 'Date','Date of Order', 'Purchase Date', 'Invoice Date', 'Transaction Date', 'Sale Date'],
            'product_category': ['product_category','Product Category', 'Category','Product','Group', 'Product Group','Category of Product', 'Item Group', 'Product Type', 'Item Category', 'Merchandise Category'],
            'total_sales': ['total_sales','Total Sales', 'Sales Amount', 'Revenue','Rev', 'Sales','Total Revenue', 'Transaction Value', 'Order Total', 'Invoice Amount'],
            'quantity_sold': ['quantity_sold','Quantity Sold', 'Units Sold','Items Sold', 'Number of Items Sold', 'Volume Sold', 'Quantity Purchased', 'Items Bought'],
            'region': ['region','Region', 'Area', 'Zone','Sales Region', 'Territory', 'Geographic Region'],
            'product': ['Product', 'Item', 'Product Name', 'Product ID','Item ID', 'Item Name', 'SKU', 'Goods', 'Merchandise'],
            }

        

        # Automatically map provided columns to available ones
        mapped_columns = {}
        for key, possible_names in column_mapping.items():
            match = find_column_name(possible_names, data.columns.tolist())
            if match:
                mapped_columns[key] = match


        # Ensure 'Order Date' is in datetime format
        if 'order_date' in mapped_columns:
            data[mapped_columns['order_date']] = pd.to_datetime(data[mapped_columns['order_date']])

        # Calculate Key Metrics
        if 'total_sales' in mapped_columns:
            
            total_revenue = data[mapped_columns['total_sales']].sum()
            avg_order_value = data[mapped_columns['total_sales']].mean()
            top_selling_products = (
                data.groupby(mapped_columns['product'])[mapped_columns['total_sales']]
                .sum()
                .sort_values(ascending=False)
                .head(5)
            )
            data['Month'] = data[mapped_columns['order_date']].dt.to_period('M')
            monthly_sales = data.groupby('Month')[mapped_columns['total_sales']].sum()
            sales_growth_rate = monthly_sales.pct_change().mean() * 100

            # Display Key Metrics
            st.subheader("Key Metrics")
            st.write(f"**Total Revenue**: ${total_revenue:,.2f}")
            st.write(f"**Average Order Value**: ${avg_order_value:,.2f}")
            st.write(f"**Sales Growth Rate**: {sales_growth_rate:.2f}%")
            st.write("**Top-Selling Products:**")
            st.write(top_selling_products)

        # Visualizations
        st.subheader("Visualizations")

        # 1. Bar Chart: Sales by Product Category
        if 'total_sales' in mapped_columns:
            if 'product_category' in mapped_columns:
                fig1, ax1 = plt.subplots()
                data.groupby(mapped_columns['product_category'])[mapped_columns['total_sales']].sum().plot(kind='bar', ax=ax1)
                ax1.set_title('Sales by Product Category')
                ax1.set_xlabel('Category')
                ax1.set_ylabel('Total Sales ($)')
                st.pyplot(fig1)
    
            # 2. Line Chart: Revenue Trends Over Time
            fig2, ax2 = plt.subplots()
            monthly_sales.plot(kind='line', marker='o', ax=ax2)
            ax2.set_title('Revenue Trends Over Time')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Total Sales ($)')
            st.pyplot(fig2)

            # 3. Geographic Sales Distribution (using Plotly)
            if 'region' in mapped_columns:
                fig3 = px.bar(
                    data.groupby(mapped_columns['region'])[mapped_columns['total_sales']].sum().reset_index(),
                    x=mapped_columns['region'], y=mapped_columns['total_sales'], 
                    title='Sales Distribution Across Regions',
                    labels={mapped_columns['total_sales']: 'Total Sales ($)'}
                )
                st.plotly_chart(fig3)
        
        selection_state=1

    elif analytics_type == "Customer/User Analytics":
       

        # Sample column name mapping for potential variations
        column_mapping = {
        'Customer_ID': ['Customer_ID','CustomerID', 'Cust_ID', 'Customer ID', 'CustID', 'UserID', 'User ID', 'ID'],
        'Age': ['Age','Customer_Age', 'User_Age', 'Age_Group', 'Years', 'Customer Age'],
        'Gender': ['Gender','Sex', 'Customer_Gender', 'User_Gender', 'Gender_Code', 'Gender ID', 'Sex_Code'],
        'Segment': ['Segment','Customer_Segment', 'User_Segment', 'Customer Group', 'Segment_Type', 'User_Group', 'Segment_Category'],
        'Region': [ 'Region','Location', 'Area', 'Zone', 'Region_Code', 'Customer_Region', 'Geographic_Region'],
        'CLV': ['CLV','Customer_Lifetime_Value', 'Lifetime_Value', 'Customer Value', 'CLTV', 'LTV'],
        'Churn_Flag': ['Churn_Flag','Churn', 'Churn_Status', 'Is_Churned', 'Churned', 'Churned_Flag', 'Customer_Churn'],
        'Retention_Score': ['Retention_Score','Retention', 'Retention_Rate', 'Retention_Index', 'Customer_Retention', 'Loyalty_Score'],
        'NPS': ['NPS','Net_Promoter_Score', 'NetPromoterScore', 'Customer_NPS', 'Promoter_Score', 'Loyalty_Index'],
        'Signup_Date': ['Signup_Date','Registration_Date', 'Sign_Up_Date', 'Join_Date', 'Customer_Join_Date', 'Created_Date', 'Account_Created'],
        'Last_Login': ['Last_Login','Last_Activity', 'Last_Active_Date', 'Last_Access', 'Last_Signin', 'Last_Login_Date', 'Recent_Login'],
        'Average_Session_Duration': ['Average_Session_Duration','Avg_Session_Time', 'Session_Duration', 'Average_Time_Spent', 'Avg_Duration', 'Avg_Session_Length'],
        'Purchase_Amount': [' Purchase_Amount','Total_Spend', 'Purchase_Value', 'Transaction_Amount', 'Amount_Spent', 'Customer_Spend', 'Spend']
    }

        
        # Automatically map provided columns to available ones
        mapped_columns = {}
        for key, possible_names in column_mapping.items():
            match = find_column_name(possible_names, data.columns.tolist())
            if match:
                mapped_columns[key] = match
        
        if 'Signup_Date' in mapped_columns:
            data[mapped_columns['Signup_Date']] = pd.to_datetime(data[mapped_columns['Signup_Date']])
        
        # Display Key Metrics
        st.subheader("Key Metrics")
        # Calculate Key Metrics
        if 'Churn_Flag' in mapped_columns:
            churn_rate = data[mapped_columns['Churn_Flag']].mean() * 100
            st.write(f"**Customer Churn Rate**: {churn_rate:.2f}%")
        if 'Retention_Score' in mapped_columns:
            retention_rate = (data[mapped_columns['Retention_Score']] > 70).mean() * 100
            st.write(f"**Retention Rate**: {retention_rate:.2f}%")
        if 'CLV' in mapped_columns:
            avg_clv = data[mapped_columns['CLV']].mean()
            st.write(f"**Average Customer Lifetime Value (CLV)**: ${avg_clv:.2f}")
        if 'NPS' in mapped_columns:
            nps = data[mapped_columns['NPS']].mean()
            st.write(f"**Net Promoter Score (NPS)**: {nps:.2f}")
       
       
       
       
       

        # Visualizations
        st.subheader("Visualizations")
        # Pie Chart: Customer Segmentation
        #st.write("### Customer Segmentation by Segment")
        if 'Segment' in mapped_columns:
            segment_counts = data[mapped_columns['Segment']].value_counts()
            fig, ax = plt.subplots()
            ax.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', startangle=140)
            ax.set_title('Customer Segmentation')
            st.pyplot(fig)
        
        # Heatmap: User Retention Across Time Periods
        #st.write("### User Retention Across Time Periods")
        if 'Signup_Date' in mapped_columns and 'Retention_Score' in mapped_columns:
            data['Signup_Month'] = data[mapped_columns['Signup_Date']].dt.to_period('M')
            retention_pivot = data.pivot_table(values='Retention_Score', index='Signup_Month', aggfunc='mean')
        
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(retention_pivot, cmap='YlGnBu', annot=True, fmt=".1f", ax=ax)
            ax.set_title('User Retention Across Time Periods')
            st.pyplot(fig)
        
        # Bar Chart: Customer Lifetime Value by Segment
        #st.write("### Average Customer Lifetime Value by Segment")
        if 'Segment' in mapped_columns and 'CLV' in mapped_columns:
            avg_clv_by_segment = data.groupby(mapped_columns['Segment'])[mapped_columns['CLV']].mean()
            fig, ax = plt.subplots()
            avg_clv_by_segment.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_title('Customer Lifetime Value by Segment')
            ax.set_xlabel('Segment')
            ax.set_ylabel('Average CLV')
            st.pyplot(fig)
        
        selection_state=1
        
    elif analytics_type == "Financial Analytics":
       

        # Sample column name mapping for potential variations
        column_mapping = {
        'Transaction_ID': ['Transaction_ID', 'Trans_ID', 'TransactionID', 'Txn_ID', 'Transaction Number', 'TxnNum'],
    'Customer_ID': ['Customer_ID', 'Cust_ID', 'CustomerID', 'Client_ID', 'User_ID', 'Client Number'],
    'Account_Type': ['Account_Type', 'Acct_Type', 'Account Category', 'Account Classification', 'AcctType'],
    'Loan_Amount': ['Loan_Amount', 'LoanAmt', 'Loan Principal', 'Principal Amount', 'Loan Total'],
    'Credit_Score': ['Credit_Score', 'Cred_Score', 'CreditRating', 'Score', 'Credit_Sc'],
    'Default_Flag': ['Default_Flag', 'Default_Status', 'Loan Default', 'Is_Defaulted', 'DefaultInd', 'Default Indicator'],
    'Transaction_Date': ['Transaction_Date', 'Trans_Date', 'TxnDate', 'Date_of_Transaction', 'TransactionDate'],
    'Transaction_Amount': ['Transaction_Amount', 'TxnAmt', 'TransactionValue', 'Trans_Amt', 'Amount'],
    'Interest_Rate': ['Interest_Rate', 'Int_Rate', 'Rate', 'Loan Interest', 'Interest %'],
    'Annual_Income': ['Annual_Income', 'Yearly Income', 'Income', 'AnnualSalary', 'Income/Year'],
    'Fraud_Flag': ['Fraud_Flag', 'Is_Fraud', 'Fraudulent', 'FraudAlert', 'FraudStatus'],
    'Monthly_Payment': ['Monthly_Payment', 'MonthlyPay', 'Monthly Installment', 'Payment_Per_Month', 'MonthlyRepayment'],
    'Tenure': ['Tenure', 'Loan_Tenure', 'Duration', 'Repayment Duration', 'Loan Term', 'Term Length']
    }

        
        # Automatically map provided columns to available ones
        mapped_columns = {}
        for key, possible_names in column_mapping.items():
            match = find_column_name(possible_names, data.columns.tolist())
            if match:
                mapped_columns[key] = match
        
        if 'Transaction_Date' in mapped_columns:
            data[mapped_columns['Transaction_Date']] = pd.to_datetime(data[mapped_columns['Transaction_Date']])
        
        # Display Key Metrics
        st.subheader("Key Metrics")
       # Key Metrics Calculation
        if 'Default_Flag' in mapped_columns:
            loan_default_rate = data[mapped_columns['Default_Flag']].mean()  # Percentage of defaults
            st.write(f"**Loan Default Rate**: {loan_default_rate:.2f}")
        if 'Interest_Rate' in mapped_columns and 'Loan_Amount' in mapped_columns:    
            profit_margin = (data[mapped_columns['Interest_Rate']].mean() / 100) * data[mapped_columns['Loan_Amount']].mean()  # Approximation of profit
            st.write(f"**Profit Margin**: {profit_margin:.2f}")
        if 'Fraud_Flag' in mapped_columns:
            fraud_detection_alerts = data[mapped_columns['Fraud_Flag']].sum()  # Total fraud alerts
            st.write(f"**Fraud Detection Alerts**: {fraud_detection_alerts:.2f}")
        if 'Transaction_Amount' in mapped_columns:
            revenue_growth = data[mapped_columns['Transaction_Amount']].pct_change().fillna(0).mean()  # Revenue growth as average % change
            st.write(f"**Average Revenue Growth**: {revenue_growth:.2f}%")
        
       
        # Visualizations
        st.subheader("Visualizations")
        if 'Fraud_Flag' in mapped_columns and 'Tenure' in mapped_columns and 'Account_Type' in mapped_columns:   
            # Heatmap for Fraud Detection Patterns by Account Type
            fraud_data = data.pivot_table(values=mapped_columns['Fraud_Flag'], index=mapped_columns['Tenure'], columns=mapped_columns['Account_Type'], aggfunc='sum', fill_value=0)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(fraud_data, annot=False, cmap="coolwarm", ax=ax)
            ax.set_title("Fraud Detection Patterns by Account Type")
            st.pyplot(fig)
        
        if 'Default_Flag' in mapped_columns and 'Account_Type' in mapped_columns:  
            # Bar Graph for Loan Default Rate by Account Type
            default_by_account = data.groupby(mapped_columns['Account_Type'])[mapped_columns['Default_Flag']].mean().reset_index(name=mapped_columns['Default_Flag'])
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=mapped_columns['Account_Type'], y=mapped_columns['Default_Flag'], data=default_by_account, color="skyblue", ax=ax)
            ax.set_title("Loan Default Rate by Account Type")
            ax.set_xlabel("Account Type")
            ax.set_ylabel("Default Rate")
            st.pyplot(fig)
        
        if 'Transaction_Date' in mapped_columns and 'Transaction_Amount' in mapped_columns:  
            # Line Chart for Financial Performance Over Time (using Transaction Amount as a proxy for revenue growth)
            data_sorted = data.sort_values(by='Transaction_Date').reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.plot(data_sorted[mapped_columns['Transaction_Date']], data_sorted[mapped_columns['Transaction_Amount']].pct_change().fillna(0), label='Revenue Growth (%)')
            ax.set_title("Financial Performance Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Revenue Growth (%)")
            st.pyplot(fig)
        
        
        selection_state=1
    
    elif analytics_type == "Marketing Analytics":
       

        # Sample column name mapping for potential variations
        column_mapping = {
        'Campaign_ID': ['Campaign_ID', 'CampaignID', 'Camp_ID', 'CampaignId', 'Campaign Code'],
    'Channel': ['Channel', 'Marketing_Channel', 'Ad_Channel', 'Source', 'Marketing Source'],
    'Target_Audience': ['Target_Audience', 'Audience', 'TargetGroup', 'Target Group', 'Demographic'],
    'Clicks': ['Clicks', 'Click_Count', 'Total Clicks', 'User Clicks'],
    'Impressions': ['Impressions', 'Impression_Count', 'Total Impressions', 'Views'],
    'Conversions': ['Conversions', 'Conversion_Count', 'Total Conversions', 'Actions'],
    'Conversion_Rate': ['Conversion_Rate', 'ConversionRate', 'CR', 'Conversion %', 'Conv Rate'],
    'Spend': ['Spend', 'Spending', 'Ad_Spend', 'Budget', 'Cost'],
    'Revenue': ['Revenue', 'Revenues', 'Sales', 'Income', 'Earnings'],
    'ROI': ['ROI', 'Return_on_Investment', 'Return', 'Profit Margin', 'Investment Return'],
    'Start_Date': ['Start_Date', 'StartDate', 'Beginning Date', 'Launch Date'],
    'End_Date': ['End_Date', 'EndDate', 'Finish Date', 'Completion Date']
    }

        
        # Automatically map provided columns to available ones
        mapped_columns = {}
        for key, possible_names in column_mapping.items():
            match = find_column_name(possible_names, data.columns.tolist())
            if match:
                mapped_columns[key] = match
        
        if 'Start_Date' in mapped_columns:
            data[mapped_columns['Start_Date']] = pd.to_datetime(data[mapped_columns['Start_Date']])
            
        if 'End_Date' in mapped_columns:
            data[mapped_columns['End_Date']] = pd.to_datetime(data[mapped_columns['End_Date']])
        
        # Display Key Metrics
        st.subheader("Key Metrics")
       #Adding calculated columns based on standardized names 
        if 'Conversions' in mapped_columns and 'Impressions' in mapped_columns:  
            data['Conversion_Rate'] = (data[mapped_columns['Conversions']] / data[mapped_columns['Impressions']]).round(2)
            conversion_rate = data['Conversion_Rate'].mean()
            st.write(f"**Average Conversion Rate**: {conversion_rate:.2f}")
            
        if 'Clicks' in mapped_columns and 'Impressions' in mapped_columns:  
            average_ctr = (data[mapped_columns['Clicks']] / data[mapped_columns['Impressions']]).mean()
            st.write(f"**Average Click-Through Rate (CTR)**: {average_ctr:.2f}")
            
        if 'Revenue' in mapped_columns and 'Spend' in mapped_columns:  
            data['ROI'] = ((data[mapped_columns['Revenue']] - data[mapped_columns['Spend']]) / data['Spend']).round(2)
            roi = data['ROI'].mean()
            st.write(f"**Average Return on Investment (ROI)**: {roi:.2f}")
            
        if 'Spend' in mapped_columns and 'Conversions' in mapped_columns:
            average_cac = data[mapped_columns['Spend']].sum() / data[mapped_columns['Conversions']].sum()
            st.write(f"**Customer Acquisition Cost (CAC)**: ${average_cac:.2f}")
      
    
        # Visualizations
        st.subheader("Visualizations")
        if 'Clicks' in mapped_columns and 'Impressions' in mapped_columns and 'Conversions' in mapped_columns:  
            # Funnel Chart for Conversion Rates through Marketing Stages
            stage_counts = data[[mapped_columns['Impressions'], mapped_columns['Clicks'], mapped_columns['Conversions']]].sum()
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.pie(stage_counts, labels=stage_counts.index, autopct='%1.1f%%', startangle=140)
            ax.set_title("Conversion Rates through Marketing Stages")
            st.pyplot(fig)
        
        if 'Clicks' in mapped_columns and 'Impressions' in mapped_columns and 'Channel' in mapped_columns: 
            # Bar Chart for Campaign Performance (CTR by Channel)
            ctr_by_channel = data.groupby(mapped_columns['Channel']).apply(lambda x: (x[mapped_columns['Clicks']].sum() / x[mapped_columns['Impressions']].sum())).reset_index(name='CTR')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=mapped_columns['Channel'], y='CTR', data=ctr_by_channel, color="skyblue",ax=ax)
            ax.set_title("Campaign Performance by Channel (Click-Through Rate)")
            ax.set_xlabel("Channel")
            ax.set_ylabel("Click-Through Rate")
            st.pyplot(fig)
        
        if 'Start_Date' in mapped_columns and 'Spend' in mapped_columns and 'Revenue' in mapped_columns:
            # Line Chart for Ad Spend vs Revenue Over Time
            data['Month_Year'] = data[mapped_columns['Start_Date']].dt.to_period('M')
            monthly_data = data.groupby('Month_Year').agg({mapped_columns['Spend']: 'mean', mapped_columns['Revenue']: 'mean'}).reset_index()
            # Convert 'Month_Year' to datetime for plotting
            monthly_data['Month_Year'] = monthly_data['Month_Year'].dt.to_timestamp()

            data_sorted_s = data.sort_values(by=mapped_columns['Start_Date']).reset_index(drop=True)
            #data_sorted_e = data.sort_values(by='End_Date').reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.plot(monthly_data['Month_Year'], monthly_data[mapped_columns['Spend']], label="Ad Spend")
            plt.plot(monthly_data['Month_Year'], monthly_data[mapped_columns['Revenue']], label="Revenue")
            ax.set_title("Monthly Average Ad Spend vs Revenue Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Amount ($)")
            ax.legend()
            st.pyplot(fig)
        
        
        selection_state=1
            
    elif analytics_type == "Service Analytics":
       

        # Sample column name mapping for potential variations
        column_mapping = {
        'Ticket_ID': ['Ticket_ID', 'TicketID', 'Ticket_Number', 'Case_ID', 'Case Number'],
    'Issue_Type': ['Issue_Type', 'Issue Category', 'Type_of_Issue', 'Problem_Type', 'Request_Type'],
    'Customer_ID': ['Customer_ID', 'Cust_ID', 'CustomerID', 'Client_ID', 'User_ID', 'Customer Number'],
    'Service_Team': ['Service_Team', 'Team', 'Support Team', 'Dept', 'Department'],
    'Resolution_Time': ['Resolution_Time', 'Time_to_Resolve', 'Time to Resolution', 'Duration', 'Resolution Duration'],
    'CSAT_Score': ['CSAT_Score', 'Customer Satisfaction', 'CSAT', 'Satisfaction Score', 'Customer_Rating'],
    'Open_Date': ['Open_Date', 'OpenDate', 'Date Opened', 'Start_Date', 'Case Open Date'],
    'Close_Date': ['Close_Date', 'CloseDate', 'End_Date', 'Date Closed', 'Case Close Date'],
    'Backlog_Volume': ['Backlog_Volume', 'Backlog', 'Pending Volume', 'Backlog Count', 'Backlog Count'],
    'Resolved_Flag': ['Resolved_Flag', 'Resolution_Status', 'Is_Resolved', 'Resolved?', 'Closed_Flag'],
    'First_Contact_Resolution': ['First_Contact_Resolution', 'FCR', 'First_Contact_Resolved', 'Resolved_on_First_Try', 'One_Call_Resolution']
    }

        
        # Automatically map provided columns to available ones
        mapped_columns = {}
        for key, possible_names in column_mapping.items():
            match = find_column_name(possible_names, data.columns.tolist())
            if match:
                mapped_columns[key] = match
        
        if 'Open_Date' in mapped_columns:
            data[mapped_columns['Open_Date']] = pd.to_datetime(data[mapped_columns['Open_Date']])
            
        if 'Close_Date' in mapped_columns:
            data[mapped_columns['Close_Date']] = pd.to_datetime(data[mapped_columns['Close_Date']])
            
        # Display Key Metrics
        st.subheader("Key Metrics")
       # Key Metrics Calculation
        if 'Resolution_Time' in mapped_columns:
            average_resolution_time = data[mapped_columns['Resolution_Time']].mean()
            st.write(f"**Average Resolution Time**: {average_resolution_time:.2f}")
        if 'CSAT_Score' in mapped_columns:
            average_csatisfaction_score = data[mapped_columns['CSAT_Score']].mean()
            st.write(f"**Customer Satisfaction Score (CSAT)**: {average_csatisfaction_score:.2f}")
        if 'Backlog_Volume' in mapped_columns:
            ticket_backlog_volume = data[mapped_columns['Backlog_Volume']].sum()
            st.write(f"**Ticket Backlog Volume**: {ticket_backlog_volume:.2f}")
        if 'First_Contact_Resolution' in mapped_columns:
            first_contact_resolution_rate = data[mapped_columns['First_Contact_Resolution']].mean()
            st.write(f"**First-Contact Resolution Rate**: {first_contact_resolution_rate:.2f}")
        
    
        # Visualizations
        st.subheader("Visualizations")
        if 'Open_Date' in mapped_columns and 'Resolution_Time' in mapped_columns:
            # Line Chart for Average Resolution Time Trends
            data_sorted = data.sort_values(by=mapped_columns['Open_Date']).reset_index(drop=True)
           
            fig = px.line(data_sorted, x=mapped_columns['Open_Date'], y=mapped_columns['Resolution_Time'],
                          labels={mapped_columns['Resolution_Time']: 'Resolution Time (hours)', 
                                  mapped_columns['Open_Date']: 'Open Date'},title="Resolution Time (hours)")
            st.plotly_chart(fig)
        
        if 'Service_Team' in mapped_columns and 'CSAT_Score' in mapped_columns:
            # Bar Graph for Customer Satisfaction Scores by Service Team
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=mapped_columns['Service_Team'], y=mapped_columns['CSAT_Score'], data=data, ci=None,ax=ax)
            ax.set_title("Customer Satisfaction Scores by Service Team")
            ax.set_xlabel("Service Team")
            ax.set_ylabel("CSAT Score")
            st.pyplot(fig)
        
        if 'Ticket_ID' in mapped_columns and 'Issue_Type' in mapped_columns and 'Service_Team' in mapped_columns:
            # Heatmap for Service Backlog by Category
            backlog_data = data.pivot_table(values=mapped_columns['Ticket_ID'], index=mapped_columns['Issue_Type'], columns=mapped_columns['Service_Team'], aggfunc='count', fill_value=0)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(backlog_data, annot=True, fmt="d", cmap="YlGnBu",ax=ax)
            ax.set_title("Service Backlog by Category")
            ax.set_xlabel("Service Team")
            ax.set_ylabel("Issue Type")
            st.pyplot(fig)
        
        selection_state=1 
        
    elif analytics_type == "Game Analytics":
       

        # Sample column name mapping for potential variations
        column_mapping = {
    'Date':['Date', 'Activity_Date', 'Play_Date', 'Session_Date', 'Log_Date','Timestamp', 'Event_Date',
            'Game_Date', 'Record_Date', 'Entry_Date', 'Creation_Date', 'Date_of_Play', 'Calendar_Date', 'Active_Date'],      
    'Player_ID': ['Player_ID', 'PlayerID', 'User_ID', 'Gamer_ID', 'UserID'],
    'Game_Session_ID': ['Game_Session_ID', 'SessionID', 'GameSessionID', 'Session_Number', 'Play_Session'],
    'Game_Level': ['Game_Level', 'Level', 'Stage', 'GameStage', 'Game_Level_ID'],
    'Session_Duration': ['Session_Duration', 'Duration', 'Play_Time', 'Session_Length', 'Time_Spent'],
    'In_Game_Purchases': ['In_Game_Purchases', 'Purchases', 'InGameSales', 'Microtransactions', 'InGamePurchases'],
    'Revenue': ['Revenue', 'Sales', 'Total_Revenue', 'Earnings', 'In_Game_Revenue','Rev'],
    'DAU': ['DAU', 'Daily_Active_Users', 'Active_Users_Daily', 'ActiveUserCount', 'UserActivity'],
    'Retention_Rate': ['Retention_Rate', 'User_Retention', 'Return_Rate', 'RetentionPercentage', 'Player_Retention'],
    'Player_Score': ['Player_Score', 'Score', 'User_Score', 'Game_Score', 'Points'],
    'Highest_Score': ['Highest_Score', 'Top_Score', 'Best_Score', 'MaxScore', 'Record_Score'],
    'Achievement_Unlocked': ['Achievement_Unlocked', 'Achievements', 'UnlockedAchievements', 'Trophies', 'Awards']
    }

        
        # Automatically map provided columns to available ones
        mapped_columns = {}
        for key, possible_names in column_mapping.items():
            match = find_column_name(possible_names, data.columns.tolist())
            if match:
                mapped_columns[key] = match
        
        if 'Date' in mapped_columns:
            data[mapped_columns['Date']] = pd.to_datetime(data[mapped_columns['Date']])
            
        # Display Key Metrics
        st.subheader("Key Metrics")     
        
        if 'Date' in mapped_columns and 'Player_ID' in mapped_columns:    
           # Key Metrics Calculation
          # Daily Active Users (DAU) - Count of unique players per day
            daily_dau = data.groupby(mapped_columns['Date'])[mapped_columns['Player_ID']].nunique()
            average_dau = daily_dau.mean()
            # Monthly Active Users (MAU) - Unique active users per month
            data['Month_Year'] = data[mapped_columns['Date']].dt.to_period('M')
            monthly_mau = data.groupby('Month_Year')[mapped_columns['Player_ID']].nunique().mean()
            
            # Time Series Line Chart for DAU and MAU Trends
            dau_mau_data = data.groupby(mapped_columns['Date']).agg({mapped_columns['Player_ID']: 'nunique'}).reset_index().rename(columns={mapped_columns['Player_ID']: 'DAU'})
            dau_mau_data['Month_Year'] = pd.to_datetime(dau_mau_data[mapped_columns['Date']]).dt.to_period('M')
            monthly_dau_mau = dau_mau_data.groupby('Month_Year')['DAU'].mean().reset_index()
            monthly_dau_mau['Month_Year'] = monthly_dau_mau['Month_Year'].dt.to_timestamp()
            
            #data_sorted = data.sort_values(by=mapped_columns['Open_Date']).reset_index(drop=True)
           
            # Melt the data for plotting both DAU and MAU in a single figure
            dau_mau_data['Metric'] = 'Daily Active Users (DAU)'
            monthly_dau_mau['Metric'] = 'Monthly Active Users (MAU)'
            monthly_dau_mau.rename(columns={'Month_Year': 'Date'}, inplace=True)
            
            combined_data = pd.concat([dau_mau_data[[mapped_columns['Date'], 'DAU', 'Metric']],
                                       monthly_dau_mau.rename(columns={'DAU': 'Value'})])

            
            st.write(f"**Average Daily Active Users (DAU)**: {average_dau:.2f}")
            st.write(f"**Monthly Active Users (MAU)**: {monthly_mau:.2f}")
            
        if 'Session_Duration' in mapped_columns:
            # Average Session Duration
            avg_session_duration = data[mapped_columns['Session_Duration']].mean()
        
        if 'Revenue' in mapped_columns:
            # In-Game Revenue - Total in-game revenue
            total_in_game_revenue = data[mapped_columns['Revenue']].sum()
            st.write(f"**Total In-Game Revenue**: ${total_in_game_revenue:.2f}")
        
        if 'Retention_Rate' in mapped_columns:
            # Player Retention Rate - Average retention rate
            average_retention_rate = data[mapped_columns['Retention_Rate']].mean()
            st.write(f"**Average Player Retention Rate**: {average_retention_rate:.2f}")
        
      
        if 'Player_ID' in mapped_columns:
            # Revenue per Active User (RPU)
            revenue_per_user = total_in_game_revenue / data[mapped_columns['Player_ID']].nunique()
            st.write(f"**Revenue per Active User (RPU)**: ${revenue_per_user:.2f}")
        
       
        
        # Visualizations
        st.subheader("Visualizations")
        
        if 'Date' in mapped_columns and 'Player_ID' in mapped_columns:  
            
            # Plotting with Plotly Express
            fig = px.line(combined_data, x=mapped_columns['Date'], y='DAU', color='Metric', 
                  labels={'DAU': 'Number of Users', mapped_columns['Date']: 'Date'},
                  title="DAU and MAU Trends Over Time")
            
            fig.update_layout(
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Number of Users")
    
            st.plotly_chart(fig)
       
        if 'Game_Level' in mapped_columns and 'Revenue' in mapped_columns:  
            # Revenue Distribution by Game Level
            
            revenue_by_level = data.groupby(mapped_columns['Game_Level'])[mapped_columns['Revenue']].sum().reset_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=mapped_columns['Game_Level'], y=mapped_columns['Revenue'], data=revenue_by_level, palette="viridis")
            ax.set_title("Revenue Distribution by Game Level")
            ax.set_xlabel("Game Level")
            ax.set_ylabel("Total Revenue ($)")
            st.pyplot(fig)

        if 'Session_Duration' in mapped_columns: 
            # Session Duration Distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data[mapped_columns['Session_Duration']], bins=30, kde=True)
            ax.set_title("Session Duration Distribution")
            ax.set_xlabel("Session Duration (minutes)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        if 'Game_Level' in mapped_columns and 'In_Game_Purchases' in mapped_columns:  
            # In-Game Purchase Frequency by User Group (Game Level as a proxy for User Group)
            purchase_by_group = data.groupby(mapped_columns['Game_Level'])[mapped_columns['In_Game_Purchases']].mean().reset_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=mapped_columns['Game_Level'], y=mapped_columns['In_Game_Purchases'], data=purchase_by_group, palette="coolwarm")
            ax.set_title("Average In-Game Purchases by Game Level")
            ax.set_xlabel("Game Level")
            ax.set_ylabel("Average In-Game Purchases ($)")
            st.pyplot(fig)

        if 'Player_ID' in mapped_columns and 'Highest_Score' in mapped_columns:  
            # Leaderboard: Top 10 Players by Highest Score
            top_players = data.groupby(mapped_columns['Player_ID'])[mapped_columns['Highest_Score']].max().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=top_players.index, y=top_players.values, palette="viridis")
            ax.set_title("Top 10 Players by Highest Score")
            ax.set_xlabel("Player ID")
            ax.set_ylabel("Highest Score")
            st.pyplot(fig)
        
        
        selection_state=1 
    
    
    elif analytics_type == "Healthcare Analytics":
       

        # Sample column name mapping for potential variations
        column_mapping = {
    'Patient_ID': ['Patient_ID', 'PatientID', 'Patient_No', 'Patient_Number', 'ID', 'Person_ID'],
    'Admission_Date': ['Admission_Date', 'AdmissionDate', 'Date_of_Admission', 'Entry_Date', 'CheckIn_Date'],
    'Discharge_Date': ['Discharge_Date', 'DischargeDate', 'Date_of_Discharge', 'Exit_Date', 'Release_Date'],
    'Department': ['Department', 'Dept', 'Ward', 'Unit', 'Section', 'Division'],
    'Treatment_Success': ['Treatment_Success', 'Success_Flag', 'Successful_Treatment', 'Outcome_Success', 'Treatment_Outcome'],
    'Readmission_Flag': ['Readmission_Flag', 'Readmission', 'Re_Admission', 'ReAdmit_Flag', 'Return_Flag','Readmitted','Returned'],
    'Critical_Condition': ['Critical_Condition', 'Critical_Flag', 'Severe_Condition', 'Emergency', 'Critical_Status'],
    'Hospital_Occupancy': ['Hospital_Occupancy', 'Occupancy', 'Occupancy_Rate', 'Bed_Occupancy', 'Occupancy_Level'],
    'Age': ['Age', 'Patient_Age', 'Years', 'Birth_Age', 'Age_in_Years'],
    'Gender': ['Gender', 'Sex', 'Patient_Gender', 'Patient_Sex', 'Gender_ID'],
    'Patient_Satisfaction_Score': ['Patient_Satisfaction_Score', 'Satisfaction', 'Satisfaction_Score', 'Patient_Score', 'Satisfaction_Level']
}


        
        # Automatically map provided columns to available ones
        mapped_columns = {}
        for key, possible_names in column_mapping.items():
            match = find_column_name(possible_names, data.columns.tolist())
            if match:
                mapped_columns[key] = match
        
        if 'Discharge_Date' in mapped_columns:
            data[mapped_columns['Discharge_Date']] = pd.to_datetime(data[mapped_columns['Discharge_Date']])
        if 'Admission_Date' in mapped_columns:
            data[mapped_columns['Admission_Date']] = pd.to_datetime(data[mapped_columns['Admission_Date']])
            
             
       # Standardizing boolean values to True/False
        def standardize_boolean(column):
            return column.replace({
                'Yes': True, 'No': False,
                1: True, 0: False
            })
        
        # Display Key Metrics
        st.subheader("Key Metrics")
       # Applying the standardization function to each relevant column
        if 'Treatment_Success' in mapped_columns:
            data[mapped_columns['Treatment_Success']] = standardize_boolean(data[mapped_columns['Treatment_Success']])
            treatment_success_rate = data[mapped_columns['Treatment_Success']].mean() * 100
            st.write(f"**Treatment Success Rate**: {treatment_success_rate:.2f}%")
        if 'Readmission_Flag' in mapped_columns:
            data[mapped_columns['Readmission_Flag']] = standardize_boolean(data[mapped_columns['Readmission_Flag']])
            readmission_rate = data[mapped_columns['Readmission_Flag']].mean() * 100
            st.write(f"**Readmission Rate**: {readmission_rate:.2f}%")
        if 'Critical_Condition' in mapped_columns:
            data[mapped_columns['Critical_Condition']] = standardize_boolean(data[mapped_columns['Critical_Condition']])
            critical_condition_rate = data[mapped_columns['Critical_Condition']].mean() * 100
            st.write(f"**Critical Condition Rate**: {critical_condition_rate:.2f}%")
       # Key Metrics Calculation
       # Calculate Length of Stay as a new column
        if 'Discharge_Date' in mapped_columns and 'Admission_Date' in mapped_columns:
            data['Length_of_Stay'] = (data[mapped_columns['Discharge_Date']] - data[mapped_columns['Admission_Date']]).dt.days.abs()
            average_length_of_stay = data['Length_of_Stay'].mean()
            st.write(f"**Average Length of Stay (days)**: {average_length_of_stay:.2f}")
       
    
        # Visualizations
        st.subheader("Visualizations")
        if 'Admission_Date' in mapped_columns and 'Hospital_Occupancy' in mapped_columns:
           # Line Chart for Hospital Occupancy Over Time
            occupancy_over_time = data.groupby(mapped_columns['Admission_Date'])[mapped_columns['Hospital_Occupancy']].mean().reset_index()
            fig1 = px.line(occupancy_over_time, x=mapped_columns['Admission_Date'], y=mapped_columns['Hospital_Occupancy'], title="Hospital Occupancy Over Time")
            fig1.update_layout(xaxis_title="Date", yaxis_title="Occupancy")
            st.plotly_chart(fig1)
        
        if 'Department' in mapped_columns and 'Treatment_Success' in mapped_columns:
            # Bar Chart for Treatment Success by Department
            success_by_department = data.groupby(mapped_columns['Department'])[mapped_columns['Treatment_Success']].mean().reset_index()
            success_by_department[mapped_columns['Treatment_Success']] *= 100  # Convert to percentage
            fig2 = px.bar(success_by_department, x=mapped_columns['Department'], y=mapped_columns['Treatment_Success'], title="Treatment Success Rate by Department",
                          labels={'Treatment_Success': 'Treatment Success Rate (%)'})
            st.plotly_chart(fig2)

        if 'Patient_Satisfaction_Score' in mapped_columns:
            # Histogram for Patient Satisfaction Score Distribution
            fig3 = px.histogram(data, x=mapped_columns['Patient_Satisfaction_Score'], nbins=5, title="Distribution of Patient Satisfaction Scores")
            fig3.update_layout(xaxis_title="Satisfaction Score", yaxis_title="Frequency")
            st.plotly_chart(fig3)
        
        if 'Critical_Condition' in mapped_columns:
            # Pie Chart for Critical vs Non-Critical Admissions
            critical_condition_counts = data[mapped_columns['Critical_Condition']].value_counts().reset_index()
            critical_condition_counts.columns = [mapped_columns['Critical_Condition'], 'Count']
            critical_condition_counts[mapped_columns['Critical_Condition']] = critical_condition_counts[mapped_columns['Critical_Condition']].replace({True: "Critical", False: "Non-Critical"})
            fig4 = px.pie(critical_condition_counts, values='Count', names=mapped_columns['Critical_Condition'], 
                          title="Critical vs Non-Critical Admissions", labels={mapped_columns['Critical_Condition']: 'Critical Condition'})
            
            st.plotly_chart(fig4)
       
        
        
        selection_state=1 
    
    elif analytics_type == "Logistics Analytics":
       

        # Sample column name mapping for potential variations
        column_mapping = {
    'Order_ID': ['Order_ID', 'OrderID', 'Order Number', 'OrderNum', 'ID_Order'],
    'Product_ID': ['Product_ID', 'ProductID', 'Product Number', 'Prod_ID', 'Item_ID'],
    'Warehouse_Location': ['Warehouse_Location', 'WarehouseLocation', 'Location', 'Warehouse', 'Storage_Location'],
    'On_Time_Delivery': ['On_Time_Delivery', 'OnTime', 'On Time', 'Delivery_OnTime', 'Timely_Delivery'],
    'Inventory_Level': ['Inventory_Level', 'Stock_Level', 'Inventory', 'Level_of_Inventory', 'StockAmount'],
    'Backorder_Flag': ['Backorder_Flag', 'Backordered', 'BackorderStatus', 'Out_of_Stock', 'Reorder_Needed'],
    'Supplier_ID': ['Supplier_ID', 'SupplierID', 'Supplier Number', 'Vendor_ID', 'VendorNumber'],
    'Delivery_Date': ['Delivery_Date', 'DeliveryDate', 'Date_of_Delivery', 'Ship_Date', 'Received_Date'],
    'Order_Date': ['Order_Date', 'OrderDate', 'Date_of_Order', 'Purchase_Date', 'Ordered_On'],
    'Shipping_Cost': ['Shipping_Cost', 'ShippingCost', 'Cost_of_Shipping', 'Freight_Cost', 'Delivery_Charge'],
    'Delivery_Time': ['Delivery_Time', 'Transit_Time', 'Shipping Duration', 'Time_to_Deliver', 'DeliveryDuration']
    }

        
        # Automatically map provided columns to available ones
        mapped_columns = {}
        for key, possible_names in column_mapping.items():
            match = find_column_name(possible_names, data.columns.tolist())
            if match:
                mapped_columns[key] = match
        
        if 'Order_Date' in mapped_columns:
            data[mapped_columns['Order_Date']] = pd.to_datetime(data[mapped_columns['Order_Date']])
        
        if 'Delivery_Date' in mapped_columns:
            data[mapped_columns['Delivery_Date']] = pd.to_datetime(data[mapped_columns['Delivery_Date']])     
             
       # Standardizing boolean values to True/False
        def standardize_boolean(column):
            return column.replace({
                'Yes': True, 'No': False,
                1: True, 0: False
            })
        
        # Display Key Metrics
        st.subheader("Key Metrics")
       # Applying the standardization function to each relevant column
        if 'On_Time_Delivery' in mapped_columns:
            data[mapped_columns['On_Time_Delivery']] = standardize_boolean(data[mapped_columns['On_Time_Delivery']])
            on_time_delivery_rate = data[mapped_columns['On_Time_Delivery']].mean() * 100
            st.write(f"**On-Time Delivery Rate**: {on_time_delivery_rate:.2f}%")
            
        if 'Inventory_Level' in mapped_columns:
             average_inventory_level = data[mapped_columns['Inventory_Level']].mean()
             st.write(f"**Average Inventory Level**: {average_inventory_level:.2f}")
        
        if 'Backorder_Flag' in mapped_columns:
            data[mapped_columns['Backorder_Flag']] = standardize_boolean(data[mapped_columns['Backorder_Flag']]) 
            backorder_rate = data[mapped_columns['Backorder_Flag']].mean() * 100
            st.write(f"**Backorder Rate**: {backorder_rate:.2f}%")
              
        if 'Delivery_Time' in mapped_columns:
            average_delivery_time = data[mapped_columns['Delivery_Time']].mean()
            st.write(f"**Average Delivery Time (days)**: {average_delivery_time:.2f}")
        
           
        # Visualizations
        st.subheader("Visualizations")
        if 'On_Time_Delivery' in mapped_columns and 'Warehouse_Location' in mapped_columns:
            # Bar Chart for On-Time Delivery Rate by Warehouse Location
            on_time_by_location = data.groupby(mapped_columns['Warehouse_Location'])[mapped_columns['On_Time_Delivery']].mean().reset_index()
            on_time_by_location[mapped_columns['On_Time_Delivery']] *= 100  # Convert to percentage
    
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=mapped_columns['Warehouse_Location'], y=mapped_columns['On_Time_Delivery'], data=on_time_by_location, palette='Blues')
            ax.set_title("On-Time Delivery Rate by Warehouse Location")
            ax.set_xlabel("Warehouse Location")
            ax.set_ylabel("On-Time Delivery Rate (%)")
            st.pyplot(fig)

        if 'Order_Date' in mapped_columns and 'Inventory_Level' in mapped_columns:      
            # Line Chart for Inventory Levels Over Time (simulated by Order Date)
            inventory_over_time = data.groupby(mapped_columns['Order_Date'])[mapped_columns['Inventory_Level']].mean().reset_index()
            fig2 = px.line(inventory_over_time, x=mapped_columns['Order_Date'], y=mapped_columns['Inventory_Level'],
                           title="Inventory Levels Over Time",
                           labels={mapped_columns['Order_Date']: 'Date', mapped_columns['Inventory_Level']: 'Average Inventory Level'})
            st.plotly_chart(fig2)
        
        if 'Backorder_Flag' in mapped_columns:    
            # Pie Chart for Backordered vs Non-Backordered Orders
            backorder_counts = data[mapped_columns['Backorder_Flag']].value_counts().reset_index()
            backorder_counts.columns = [mapped_columns['Backorder_Flag'], 'Count']
            fig3 = px.pie(backorder_counts, values='Count', names=mapped_columns['Backorder_Flag'],
                          title="Backordered vs Non-Backordered Orders",
                          labels={mapped_columns['Backorder_Flag']: 'Backorder Status'})
            fig3.update_traces(textinfo='percent+label')
            st.plotly_chart(fig3)

        if 'Delivery_Time' in mapped_columns: 
           # Histogram of Delivery Times
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data[mapped_columns['Delivery_Time']], bins=10, kde=True, color='purple')
            ax.set_title("Distribution of Delivery Times")
            ax.set_xlabel("Delivery Time (days)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
      
        
        selection_state=1 
        
    
    elif analytics_type == "Social Media Analytics":
       

        # Sample column name mapping for potential variations
        column_mapping = {
    'Post_ID': ['Post_ID', 'PostID', 'Post_ID_Number', 'ID_Post', 'Content_ID'],
    'User_ID': ['User_ID', 'UserID', 'Account_ID', 'User_Number', 'ID_User'],
    'Post_Date': ['Post_Date', 'Date_Posted', 'Content_Date', 'Date_of_Post', 'Timestamp'],
    'Platform': ['Platform', 'Social_Platform', 'SocialMedia', 'Channel', 'Source_Platform'],
    'Content_Type': ['Content_Type', 'Type_of_Content', 'ContentFormat', 'Media_Type', 'Post_Type'],
    'Engagements': ['Engagements', 'Total_Engagements', 'EngagementCount', 'Interactions', 'User_Engagement'],
    'Likes': ['Likes', 'Like_Count', 'Total_Likes', 'Number_of_Likes', 'Like_Engagements'],
    'Shares': ['Shares', 'Share_Count', 'Total_Shares', 'Number_of_Shares', 'Reposts'],
    'Comments': ['Comments', 'Comment_Count', 'Total_Comments', 'Number_of_Comments', 'Replies'],
    'Sentiment_Score': ['Sentiment_Score', 'Score_Sentiment', 'Emotion_Score', 'SentimentRating', 'Polarity'],
    'Sentiment_Label': ['Sentiment_Label', 'Label_Sentiment', 'Sentiment_Type', 'Emotion_Label', 'SentimentCategory'],
    'Hashtag': ['Hashtag', 'Tag', 'Post_Hashtag', 'Hash_Tag', 'Keyword'],
    'Total_Views': ['Total_Views', 'Views', 'View_Count', 'Total_Impressions', 'Impression_Count'],
    'Engagement_Rate': ['Engagement_Rate', 'EngagementPercentage', 'Rate_of_Engagement', 'User_Engagement_Rate', 'Interaction_Rate']
    }

        
        # Automatically map provided columns to available ones
        mapped_columns = {}
        for key, possible_names in column_mapping.items():
            match = find_column_name(possible_names, data.columns.tolist())
            if match:
                mapped_columns[key] = match
        
        if 'Post_Date' in mapped_columns:
            data[mapped_columns['Post_Date']] = pd.to_datetime(data[mapped_columns['Post_Date']])
        
             
      
        # Display Key Metrics
        st.subheader("Key Metrics")
        # Key Metrics Calculation
        if 'Engagement_Rate' in mapped_columns:
            average_engagement_rate = data[mapped_columns['Engagement_Rate']].mean()*100
            st.write(f"**Average Engagement Rate**: {average_engagement_rate:.2f}%")
        if 'Likes' in mapped_columns:
            total_likes = data[mapped_columns['Likes']].sum()
            st.write(f"**Total Likes**: {total_likes:.0f}")
        if 'Shares' in mapped_columns:    
            total_shares = data[mapped_columns['Shares']].sum()
            st.write(f"**Total Shares**: {total_shares:.0f}")
        if 'Comments' in mapped_columns:
            total_comments = data[mapped_columns['Comments']].sum()
            st.write(f"**Total Comments**: {total_comments:.0f}")
        if 'Sentiment_Label' in mapped_columns:
            sentiment_distribution = data[mapped_columns['Sentiment_Label']].value_counts()
            # Display Sentiment Distribution as a Markdown table
            st.write("**Sentiment Distribution**")
            st.markdown(
                sentiment_distribution.to_frame(name="Count")
                .rename_axis("Sentiment")
                .reset_index()
                .to_markdown(index=False)
            )
        if 'Hashtag' in mapped_columns and 'Engagements' in mapped_columns: 
            top_hashtags = data.groupby(mapped_columns['Hashtag'])[mapped_columns['Engagements']].sum().sort_values(ascending=False).head(5)
            # Display Top Hashtags by Engagement as a Markdown table
            st.write("**Top Hashtags by Engagement**")
            st.markdown(
                top_hashtags.to_frame(name="Total Engagements")
                .rename_axis("Hashtag")
                .reset_index()
                .to_markdown(index=False)
            )
        
    
        # Visualizations
        st.subheader("Visualizations")
        if 'Platform' in mapped_columns and 'Engagements' in mapped_columns: 
           # 1. Engagement by Platform
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=mapped_columns['Platform'], y=mapped_columns['Engagements'], data=data, palette='viridis')
            ax.set_title("Engagement by Platform")
            ax.set_xlabel("Platform")
            ax.set_ylabel("Total Engagements")
            st.pyplot(fig)

        if 'Sentiment_Label' in mapped_columns:       
           # 2. Sentiment Distribution Pie Chart using Plotly
            fig2 = px.pie(data, names=mapped_columns['Sentiment_Label'], title='Sentiment Distribution', 
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig2.update_traces(textinfo='percent+label')
            st.plotly_chart(fig2)
            
        if 'Engagement_Rate' in mapped_columns:
            # 3. Engagement Rate Distribution using Seaborn
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data[mapped_columns['Engagement_Rate']], bins=20, kde=True, color='teal')
            ax.set_title("Distribution of Engagement Rates")
            ax.set_xlabel("Engagement Rate")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        if 'Post_Date' in mapped_columns and 'Total_Views' in mapped_columns: 
                   # Ensure that the Post_Date column is sorted if not already
            data.sort_values(by=mapped_columns['Post_Date'], inplace=True)
            
            # Line Chart of Total Views Over Time using Plotly
            fig = px.line(data, x=mapped_columns['Post_Date'], y=mapped_columns['Total_Views'], 
                          title="Total Views Over Time",
                          labels={mapped_columns['Post_Date']: 'Date', mapped_columns['Total_Views']: 'Total Views'},
                          markers=False)
            fig.update_layout(xaxis_title="Date", yaxis_title="Total Views")
            st.plotly_chart(fig)
        
        selection_state=1 
        
    elif analytics_type == "Risk Analytics":
       

        # Sample column name mapping for potential variations
        column_mapping = {
    'Transaction_ID': ['TransactionID', 'Trans_ID', 'TransNumber', 'Transaction_No', 'ID_Transaction'],
    'Customer_ID': ['CustomerID', 'Cust_ID', 'ID_Customer', 'ClientID', 'Client_Number'],
    'Transaction_Date': ['TransactionDate', 'Date_of_Transaction', 'Trans_Date', 'Date', 'Transaction_Timestamp'],
    'Risk_Score': ['RiskScore', 'Score_Risk', 'Risk_Level', 'Transaction_Risk', 'Fraud_Risk_Score'],
    'Fraud_Flag': ['FraudFlag', 'IsFraud', 'Fraud_Status', 'Fraudulent', 'Fraud_Indicator'],
    'Location': ['Transaction_Location', 'Geo_Location', 'Place', 'Area', 'Region'],
    'Amount': ['Transaction_Amount', 'Amount_Transacted', 'Value', 'Total_Amount', 'Monetary_Value'],
    'Account_Type': ['AccountType', 'Acct_Type', 'Type_of_Account', 'AccountCategory', 'Client_Account_Type'],
    'Risk_Category': ['RiskCategory', 'Category_Risk', 'Transaction_Risk_Level', 'Risk_Classification', 'Fraud_Risk_Category'],
    'Anomaly_Score': ['AnomalyScore', 'Score_Anomaly', 'Transaction_Anomaly', 'Anomaly_Rating', 'Anomaly_Index'],
    'Alert_Flag': ['AlertFlag', 'IsAlert', 'Alert_Status', 'Transaction_Alert', 'Fraud_Alert']
    }

        
        # Automatically map provided columns to available ones
        mapped_columns = {}
        for key, possible_names in column_mapping.items():
            match = find_column_name(possible_names, data.columns.tolist())
            if match:
                mapped_columns[key] = match
        
        if 'Transaction_Date' in mapped_columns:
            data[mapped_columns['Transaction_Date']] = pd.to_datetime(data[mapped_columns['Transaction_Date']])
        
        # Standardizing boolean values to True/False
        def standardize_boolean(column):
            return column.replace({
                 'Yes': 1, 'No':0,
                 True: 1, False: 0
             })
        # Display Key Metrics
        st.subheader("Key Metrics")
        # Key Metrics Calculation
        if 'Risk_Score' in mapped_columns:
            average_risk_score = data[mapped_columns['Risk_Score']].mean()
            st.write(f"**Average Risk Score**: {average_risk_score:.2f}")
        # Applying the standardization function to each relevant column
        if 'Fraud_Flag' in mapped_columns:
            data[mapped_columns['Fraud_Flag']] = standardize_boolean(data[mapped_columns['Fraud_Flag']])
            fraud_rate = data[mapped_columns['Fraud_Flag']].mean() * 100
            st.write(f"**Fraud Rate**: {fraud_rate:.2f}%")
        if 'Alert_Flag' in mapped_columns:
            data[mapped_columns['Alert_Flag']] = standardize_boolean(data[mapped_columns['Alert_Flag']])      
            anomaly_alert_rate = data[mapped_columns['Alert_Flag']].mean() * 100
            st.write(f"**Anomaly Alert Rate**: {anomaly_alert_rate:.2f}%")   
        if 'Risk_Category' in mapped_columns:
            high_risk_transactions = (data[mapped_columns['Risk_Category']] == 'High').sum()
            st.write(f"**High-Risk Transactions**:{high_risk_transactions}")
        
       
        # Visualizations
        st.subheader("Visualizations")
        if 'Fraud_Flag' in mapped_columns:
           # Prepare data for the pie chart
            fraud_counts = data[mapped_columns['Fraud_Flag']].value_counts().reset_index()
            fraud_counts.columns = [mapped_columns['Fraud_Flag'], 'Count']
            fraud_counts[mapped_columns['Fraud_Flag']] = fraud_counts[mapped_columns['Fraud_Flag']].map({0: 'Non-Fraudulent', 1: 'Fraudulent'})
    
            # 1. Pie Chart for Fraud vs Non-Fraudulent Transactions
            fig = px.pie(fraud_counts, values='Count', names=mapped_columns['Fraud_Flag'],
                         title="Fraud vs Non-Fraudulent Transactions",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        
            # Update the pie chart layout
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig)
               
        if 'Risk_Score' in mapped_columns:      
            # 2. Histogram of Risk Scores
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data[mapped_columns['Risk_Score']], bins=10, kde=True, color='blue')
            ax.set_title("Distribution of Risk Scores")
            ax.set_xlabel("Risk Score")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        
        if 'Risk_Category' in mapped_columns and 'Account_Type' in mapped_columns: 
            # 3. Bar Chart for High-Risk Transactions by Account Type
            high_risk_by_account = data[data[mapped_columns['Risk_Category']] == 'High'].groupby(mapped_columns['Account_Type']).size().reset_index(name='Count')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=mapped_columns['Account_Type'], y='Count', data=high_risk_by_account, palette='Oranges')
            ax.set_title("High-Risk Transactions by Account Type")
            ax.set_xlabel("Account Type")
            ax.set_ylabel("Count")
            st.pyplot(fig)


        #data.sort_values(by='Post_Date', inplace=True)
        if 'Transaction_Date' in mapped_columns and 'Risk_Score' in mapped_columns: 
           # 4. Line Chart for Average Risk Score Over Time
            risk_score_over_time = data.groupby('Transaction_Date')['Risk_Score'].mean().reset_index()
            fig = px.line(risk_score_over_time, x=mapped_columns['Transaction_Date'], y=mapped_columns['Risk_Score'], 
                          title="Average Risk Score Over Time",
                          labels={mapped_columns['Transaction_Date']: 'Date', mapped_columns['Risk_Score']: 'Average Risk Score'},
                          markers=False)
            st.plotly_chart(fig)
        
        selection_state=1 
        
    # Continuously ask for questions
    while selection_state==1:
        # Display previous questions and responses from session history
        #for entry in st.session_state.history:
       #     st.markdown(f"**Q:** {entry['question']}")
       #     st.markdown(f"**A:** {entry['response']}")
        #data_f=data   
        # Ask a new question
        if question_counter==response_counter:
            if question_counter==0:
                user_question = st.text_input("Ask a question about your data or request a plot:", key=f"q{question_counter}")
            else:
                user_question = st.text_input("", key=f"q{question_counter}")
            question_counter+=1
        #if user_question.lower().strip() == "no" or user_question.strip() == "":
        #    st.write("Thank you! No more questions.")
        #    break  # Exit the loop if user inputs "no" or leaves it blank
        if user_question:
            if user_question.lower() in ['no','no more', 'thank you','done']:
                break
        # Extract column names and chart type from the question
            response_counter+=1
            #x_column = extract_column_name(user_question, data.columns)
            #y_column = extract_column_name(user_question, data.columns)
            chart_type = extract_chart_type(user_question)
        
            column_name = extract_column_name(user_question, data.columns)
            group_column = extract_grouping_column(user_question, data.columns)
            date_range = extract_date_range(user_question)
            
            #if len(column_name)>1:
             #   y_column=column_name[1]
              #  x_column=column_name[0]
                
            if len(column_name)>1:
                y_column=column_name[0]
                x_column=column_name[1]
            stat_func, stat_name = extract_stat_function(user_question)
            
            # Check if the question is about null values
            
            if date_range:
                if len(date_range)>1:
                    try:
                        data = data[
                            (data[date_cols[0]] >= date_range[0]) & (data[date_cols[0]] <= date_range[1])
                        ]
                        response = f"Filtered data between {date_range[0]} and {date_range[1]}:"
                        #st.write(filtered_data)
                    except Exception as e:
                        response = f"Error filtering by date: {e}"
                elif 'before' in user_question.lower():
                    try:
                        data = data[
                            (data[date_cols[0]] < date_range[0]) 
                        ]
                        response = f"Filtered data between {date_range[0]} and {date_range[1]}:"
                        #st.write(filtered_data)
                    except Exception as e:
                        response = f"Error filtering by date: {e}"
                        
                elif 'after' in user_question.lower():
                    try:
                        data = data[
                            (data[date_cols[0]] > date_range[0]) 
                        ]
                        response = f"Filtered data between {date_range[0]} and {date_range[1]}:"
                        #st.write(filtered_data)
                    except Exception as e:
                        response = f"Error filtering by date: {e}"    
    
            # Handle chart generation
            if chart_type and x_column and y_column:
                try:
                    if chart_type == "bar":
                        fig = px.bar(data, x=x_column, y=y_column, title=f"{y_column} by {x_column}")
                    elif chart_type == "line":
                        fig = px.line(data, x=x_column, y=y_column, title=f"{y_column} over {x_column}")
                    elif chart_type == "histogram":
                        fig = px.histogram(data, x=x_column, title=f"Histogram of {x_column}")
                    elif chart_type == "scatter":
                        fig = px.scatter(data, x=x_column, y=y_column, title=f"{y_column} vs {x_column}")
                    
                    elif chart_type == "pie":
                        fig = px.pie(data, names=x_column, values=y_column, title=f"{y_column} by {x_column}")
                        
                    elif chart_type == "box":
                        fig = px.box(data, x=x_column, y=y_column, title=f"{y_column} by {x_column}")
                    
                    response = f"Generated a {chart_type} chart of **{y_column}** vs **{x_column}**."
                    st.plotly_chart(fig)
    
                except Exception as e:
                    response = f"Error generating {chart_type} chart: {e}"
                    st.error(response)
           
            elif detect_null_check(user_question):
                 try:
                     if len(column_name)>0:
                         # Check for null values in the specified column
                         null_count = data[column_name[0]].isnull().sum()
                         if null_count > 0:
                             st.write(f"Number of null values in '{column_name[0]}': {null_count}")
                         else:
                             st.write(f"No missing values in '{column_name[0]}'.")
                     else:
                         # Show total null counts for all columns
                         null_summary = data.isnull().sum()
                         if null_summary.sum() > 0:
                             st.write("Missing values summary for all columns:")
                             st.write(null_summary[null_summary > 0])  # Display only columns with missing values
                         else:
                             st.write("No missing values in the entire dataset.")
                 except Exception as e:
                     response = f"Error reading date: {e}"
                     
                    # Handle summary-related questions
            elif detect_summary_request(user_question):
                if "rows" in user_question.lower():
                    response = f"Number of rows: {data.shape[0]}"
                elif "column" in user_question.lower() and  "num" in user_question.lower():
                    response = f"Number of columns: {len(col_names)}"
                elif "column" in user_question.lower() and "name" in user_question.lower():
                    response = f"Column names: {', '.join(data.columns)}"
                elif "data types" in user_question.lower():
                    response = f"Data types:\n{data.dtypes.to_string()}"
                elif "correlation" in user_question.lower():
                    st.subheader("Correlation Matrix")
                    numeric_df = data.select_dtypes(include=[np.number])
                    corr_matrix = numeric_df.corr()
                    st.write(corr_matrix)
                    response=False
                elif "heatmap" in user_question.lower() or "heat map" in user_question.lower():
                    numeric_df = data.select_dtypes(include=[np.number])
                    corr_matrix = numeric_df.corr()
                    fig, ax = plt.subplots()
                    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
                    response=False
                if response:
                    st.write(response)
                    
        
                 
                    
            elif column_name and stat_func:
                 #st.write(f"Performing **{stat_name}** on **{column_name}**...")
                 if group_column:
                     try:
                         # Perform the statistical operation
                         if column_name[0]!=group_column:
                             response =data.groupby(group_column)[column_name[0]].apply(stat_func)
                         else:
                             response =data.groupby(group_column)[column_name[1]].apply(stat_func)
                         #st.subheader("Result")
                         #st.write(f"{stat_name.capitalize()} of {column_name}: {result}")
                         st.write(f"{response}")
                     except Exception as e:
                         st.error(f"Error performing {stat_name} on {column_name[0]} by {group_column}: {e}")
                 else:
                     try:
                         # Perform the statistical operation
                         response = stat_func(data[column_name[0]])
                         #st.subheader("Result")
                         #st.write(f"{stat_name.capitalize()} of {column_name}: {result}")
                         st.write(f"{response}")
                     except Exception as e:
                         st.error(f"Error performing {stat_name} on {column_name}: {e}")
             #else:
             #    st.warning("Could not identify a column name or statistical function in your question.")
            else:
                #response = "Could not identify the chart type or column names. Please try again."
                response = "Could not undrestand the request. Please try again."
                st.warning(response)

            # Store the question and response in session history
            
            # Store the question and response
            if "history" not in st.session_state:
                st.session_state.history = []
                st.session_state.history.append({"question": user_question, "response": response})

            # Display question and response history
        for entry in st.session_state.history:
            st.markdown(f"**Q:** {entry['question']}")
            st.markdown(f"**A:** {entry['response']}")


       
   

        # Optional: Add logic to generate new charts based on user query
