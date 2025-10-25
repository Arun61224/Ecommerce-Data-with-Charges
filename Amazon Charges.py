import streamlit as st
import pandas as pd
import numpy as np

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Amazon Seller Reconciliation Dashboard")
st.title("💰 Amazon Seller Central Reconciliation Dashboard")
st.markdown("---")


# --- 1. Master Data Loading & Transformation (Cached for Speed) ---

# Load Raw Payment Data (for detailed breakdown)
@st.cache_data
def load_raw_payment_data(file_path):
    # This file was created in the previous step
    df = pd.read_csv(file_path)
    return df

# Load MTR Summary Data (for reconciliation summary)
@st.cache_data
def load_mtr_summary_data(file_path):
    # This file was created in the previous step
    df = pd.read_csv(file_path)
    return df

# Perform the main financial aggregation and merging
@st.cache_data
def create_final_reconciliation_df(df_payment_raw, df_logistics_master):
    # --- A. Create Financial Master Summary (Pivoting/Grouping) ---
    
    # 1. Total Principal (Revenue)
    principal_df = df_payment_raw[df_payment_raw['amount-description'] == 'Principal'].groupby('OrderID')['amount'].sum().reset_index(name='Total_Principal')
    
    # 2. Total Amazon Fees (Cost): Group all negative amounts except Tax/TCS/Reimbursement types
    
    # Identify Fee descriptions for accurate summation (using unique values from previous step)
    fee_descriptions = [
        'FBA Pick & Pack Fee', 'FBA Weight Handling Fee', 'Commission', 
        'Fixed closing fee', 'Technology Fee', 'Refund commission',
        'FBA Pick & Pack Fee CGST', 'FBA Pick & Pack Fee SGST', 
        'FBA Weight Handling Fee CGST', 'FBA Weight Handling Fee SGST',
        'Commission IGST', 'Fixed closing fee IGST', 'Technology Fee IGST',
        'Refund commission IGST', 'Commission CGST', 'Commission SGST',
        'Fixed closing fee CGST', 'Fixed closing fee SGST'
    ]
    
    df_fees = df_payment_raw[df_payment_raw['amount-description'].isin(fee_descriptions)]
    
    # Sum all fees for each order. Since fees are reported as negative, we take the absolute sum.
    fees_summary_df = df_fees.groupby('OrderID')['amount'].sum().reset_index(name='Total_Amazon_Fees')
    fees_summary_df['Total_Amazon_Fees'] = fees_summary_df['Total_Amazon_Fees'].abs()
    
    # 3. Combine Principal and Fees
    df_financial_master = pd.merge(principal_df, fees_summary_df, on='OrderID', how='outer').fillna(0)
    
    # --- B. Final Merge (Financial Master + Logistics Master) ---
    df_final = pd.merge(
        df_financial_master, 
        df_logistics_master, 
        on='OrderID', 
        how='outer'
    ).fillna(0)
    
    # --- C. Final Calculation ---
    df_final['Net_Settlement'] = df_final['Total_Principal'] - df_final['Total_Amazon_Fees'] - df_final['Total_MTR_Billed']
    
    # Add a status column for filtering
    df_final['Status'] = np.select(
        [df_final['Net_Settlement'] > 0, df_final['Net_Settlement'] < 0],
        ['Profit', 'Loss'],
        default='Break Even'
    )

    return df_final


# --- Load the Prepared DataFrames ---
try:
    df_payment_raw = load_raw_payment_data("combined_payment_raw_data.csv")
    df_logistics_master = load_mtr_summary_data("logistics_master_summary.csv")
    df_reconciliation = create_final_reconciliation_df(df_payment_raw, df_logistics_master)
    
except FileNotFoundError:
    st.error("Error: Required intermediate files (combined_payment_raw_data.csv or logistics_master_summary.csv) not found.")
    st.stop()


# --- Dashboard Layout and Filters ---

st.sidebar.header("Filter & Breakdown")

# Order ID Selection for Detailed Breakdown (Your specific request)
order_id_list = ['All Orders'] + sorted(df_reconciliation['OrderID'].unique().tolist())
selected_order_id = st.sidebar.selectbox("Select Order ID for Breakdown:", order_id_list)


# --- 2. Main Reconciliation Table ---
st.header("1. Order-Level Reconciliation Summary")

if selected_order_id != 'All Orders':
    df_display = df_reconciliation[df_reconciliation['OrderID'] == selected_order_id]
else:
    df_display = df_reconciliation.sort_values(by='Net_Settlement', ascending=False)

# Display Summary Metrics (KPI Cards)
total_orders = df_reconciliation.shape[0]
total_principal = df_reconciliation['Total_Principal'].sum()
total_fees = df_reconciliation['Total_Amazon_Fees'].sum()
total_mtr = df_reconciliation['Total_MTR_Billed'].sum()
net_settlement_value = df_reconciliation['Net_Settlement'].sum()

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
col_kpi1.metric("Total Reconciled Orders", f"{total_orders:,}")
col_kpi2.metric("Total Principal (Revenue)", f"INR {total_principal:,.2f}")
col_kpi3.metric("Total Amazon Fees", f"INR {total_fees:,.2f}")
col_kpi4.metric("Net Settlement Value", f"INR {net_settlement_value:,.2f}", 
                delta=f"MTR: -INR {total_mtr:,.2f}")

st.dataframe(df_display, use_container_width=True, hide_index=True)

# Allow download of the full reconciliation file
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

csv = convert_df_to_csv(df_reconciliation)
st.download_button(
    label="Download Full Reconciliation Report (.csv)",
    data=csv,
    file_name='full_amazon_reconciliation_report.csv',
    mime='text/csv',
)
st.markdown("---")

# --- 3. Detailed Charge Breakdown (Your Specific Request) ---
st.header("2. Detailed Charge Breakdown (Payment File)")

if selected_order_id != 'All Orders':
    st.subheader(f"Showing all charges for Order ID: `{selected_order_id}`")
    
    # Filter the raw payment data for the selected OrderID
    df_breakdown = df_payment_raw[df_payment_raw['OrderID'] == selected_order_id]
    
    # Select and display the relevant columns
    df_breakdown_display = df_breakdown[[
        'transaction-type', 
        'amount-type', 
        'amount-description', 
        'amount', 
        'posted-date'
    ]].sort_values(by='amount', ascending=False).reset_index(drop=True)
    
    st.dataframe(df_breakdown_display, use_container_width=True, hide_index=True)
    
    # Display MTR details for context
    mtr_row = df_logistics_master[df_logistics_master['OrderID'] == selected_order_id]
    if not mtr_row.empty:
        mtr_billed = mtr_row['Total_MTR_Billed'].iloc[0]
        st.info(f"**Logistics Charge (MTR):** INR {mtr_billed:,.2f} (from MTR Report)")
    else:
        st.warning("No MTR data found for this Order ID in the MTR files.")

else:
    st.info("Please select a specific Order ID from the sidebar dropdown to view the detailed charge breakdown.")