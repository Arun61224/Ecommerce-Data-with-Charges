import streamlit as st
import pandas as pd
import numpy as np
import io

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Amazon Seller Reconciliation Dashboard")
st.title("💰 Amazon Seller Central Reconciliation Dashboard (Detailed)")
st.markdown("---")


# --- 1. Data Processing Functions ---

@st.cache_data(show_spinner="Processing Payment Files and Creating Financial Master...")
def process_payment_files(uploaded_payment_files):
    """Reads all uploaded TXT payment files, creates the financial summary (Net Sales, Fees, Tax/TCS) and raw breakdown."""
    
    all_payment_data = []
    
    # Define Column Names manually 
    cols = ['settlement-id', 'settlement-start-date', 'settlement-end-date', 'deposit-date', 
            'total-amount', 'currency', 'transaction-type', 'order-id', 'merchant-order-id', 
            'adjustment-id', 'shipment-id', 'marketplace-name', 'amount-type', 
            'amount-description', 'amount', 'fulfillment-id', 'posted-date', 
            'posted-date-time', 'order-item-code', 'merchant-order-item-id', 
            'merchant-adjustment-item-id', 'sku', 'quantity-purchased', 'promotion-id'] # 24 columns

    for file in uploaded_payment_files:
        try:
            # FIX: Use header=1 to skip the primary header row and read files robustly
            df_temp = pd.read_csv(io.StringIO(file.getvalue().decode("latin-1")), 
                                  sep='\t', 
                                  skipinitialspace=True,
                                  header=1)
            
            # FIX: Normalize Column Count for files with extra columns
            if len(df_temp.columns) > len(cols):
                df_temp = df_temp.iloc[:, :len(cols)]
            df_temp.columns = cols 
            
            all_payment_data.append(df_temp)
        except Exception as e:
            st.error(f"Error reading {file.name}: The file structure is unexpected. Details: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    df_payment_raw = pd.concat(all_payment_data, ignore_index=True)
    df_payment_cleaned = df_payment_raw.dropna(subset=['order-id']).copy()
    
    df_payment_cleaned.rename(columns={'order-id': 'OrderID'}, inplace=True)
    df_payment_cleaned['OrderID'] = df_payment_cleaned['OrderID'].astype(str)
    df_payment_cleaned['amount'] = pd.to_numeric(df_payment_cleaned['amount'], errors='coerce').fillna(0)
    
    # Create Raw Payment Data (kept only for compatibility, not used in final export)
    charge_breakdown_cols = ['OrderID', 'transaction-type', 'marketplace-name', 'amount-type', 'amount-description', 'amount', 'posted-date', 'settlement-id']
    df_charge_breakdown = df_payment_cleaned[charge_breakdown_cols]
    
    # --- PIVOTING: Calculate CLASSIFIED AMOUNTS per OrderID (NEW BIFURCATION) ---
    
    # Helper function for fee calculation
    def calculate_fee_total(df, keyword, name):
        # Filter and sum the absolute value (since fees are negative)
        df_fee = df[df['amount-description'].str.contains(keyword, case=False, na=False)]
        df_summary = df_fee.groupby('OrderID')['amount'].sum().reset_index(name=name)
        df_summary[name] = df_summary[name].abs()
        return df_summary
        
    # 1. Net Settlement Amount (Total Payment)
    df_financial_master = df_charge_breakdown.groupby('OrderID')['amount'].sum().reset_index(name='Net_Payment_Fetched')
    
    # 2. Bifurcated Fees (Total Fees is removed, individual fees are added)
    df_comm = calculate_fee_total(df_charge_breakdown, 'Commission', 'Total_Commission_Fee')
    df_fixed = calculate_fee_total(df_charge_breakdown, 'Fixed closing fee', 'Total_Fixed_Closing_Fee')
    df_pick = calculate_fee_total(df_charge_breakdown, 'Pick & Pack Fee', 'Total_FBA_Pick_Pack_Fee')
    df_weight = calculate_fee_total(df_charge_breakdown, 'Weight Handling Fee', 'Total_FBA_Weight_Handling_Fee')
    df_tech = calculate_fee_total(df_charge_breakdown, 'Technology Fee', 'Total_Technology_Fee')
    
    # 3. Total Tax/TDS/TCS Components
    tax_descriptions = ['TCS', 'TDS', 'Tax']
    df_tax_summary = calculate_fee_total(df_charge_breakdown, '|'.join(tax_descriptions), 'Total_Tax_TCS_TDS')


    # --- FINAL FINANCIAL MASTER MERGE (Net Payment + Fees Bifurcation + Tax) ---
    
    # Merge individual fees and tax onto the Net Payment master
    for df in [df_comm, df_fixed, df_pick, df_weight, df_tech, df_tax_summary]: 
        df_financial_master = pd.merge(df_financial_master, df, on='OrderID', how='left').fillna(0)
    
    # Calculate the old Total_Fees for KPI card consistency
    df_financial_master['Total_Fees_KPI'] = (
        df_financial_master['Total_Commission_Fee'] +
        df_financial_master['Total_Fixed_Closing_Fee'] +
        df_financial_master['Total_FBA_Pick_Pack_Fee'] +
        df_financial_master['Total_FBA_Weight_Handling_Fee'] +
        df_financial_master['Total_Technology_Fee']
    )
    
    return df_financial_master, df_charge_breakdown 


@st.cache_data(show_spinner="Processing MTR Files and Creating Detailed Logistics Master...")
def process_mtr_files(uploaded_mtr_files):
    """Reads all uploaded CSV MTR files and concatenates them, keeping item-level detail."""
    
    all_mtr_data = []
    
    for file in uploaded_mtr_files:
        try:
            df_temp = pd.read_csv(file) 
            df_temp = df_temp.loc[:, ~df_temp.columns.str.contains('^Unnamed')] # Clean Unnamed columns
            
            all_mtr_data.append(df_temp)
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
            return pd.DataFrame()
        
    df_mtr_raw = pd.concat(all_mtr_data, ignore_index=True)
    
    # Rename for consistency
    df_mtr_raw.rename(columns={'Order Id': 'OrderID', 'Invoice Amount': 'MTR Invoice Amount'}, inplace=True)
    
    # Select only the columns required by the user (MTR Table Update requirement)
    required_mtr_cols = [
        'Invoice Number', 'Invoice Date', 'Transaction Type', 'OrderID', 
        'Quantity', 'Sku', 'Ship From City', 'Ship To City', 'Ship To State', 
        'MTR Invoice Amount'
    ]
    
    # Ensure all required columns exist before copying
    for col in required_mtr_cols:
        if col not in df_mtr_raw.columns:
            # If a column is missing (e.g., Sku), create it with empty values to prevent KeyError
            df_mtr_raw[col] = ''
    
    df_logistics_master = df_mtr_raw[required_mtr_cols].copy()
    
    # Cleaning
    df_logistics_master['OrderID'] = df_logistics_master['OrderID'].astype(str)
    df_logistics_master['MTR Invoice Amount'] = pd.to_numeric(df_logistics_master['MTR Invoice Amount'], errors='coerce').fillna(0)
    
    return df_logistics_master


@st.cache_data(show_spinner="Merging data and finalizing calculations...")
def create_final_reconciliation_df(df_financial_master, df_logistics_master):
    """Merges detailed MTR data with Payment Net Sale Value, Fees, and Tax/TCS."""
    
    # Perform Left Merge: MTR data is the primary base, Payment data is fetched
    df_final = pd.merge(
        df_logistics_master, 
        df_financial_master, 
        on='OrderID', 
        how='left'
    ).fillna(0)
    
    # Rename Net Payment column for clarity on screen
    df_final.rename(columns={'Net_Payment_Fetched': 'Net Payment'}, inplace=True)
    
    return df_final


# --- EXPORT FUNCTION (Single Sheet Only) ---
@st.cache_data
def convert_to_excel(df_reconciliation): # Removed df_payment_raw_breakdown
    """Creates a single-sheet Excel file for export."""
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_reconciliation.to_excel(writer, sheet_name='Reconciliation Summary', index=False)
    
    return output.getvalue()


# --- 2. File Upload Section ---

with st.sidebar:
    st.header("Upload Raw Data Files")

    payment_files = st.file_uploader(
        "1. Upload ALL Payment Reports (.txt)", 
        type=['txt'], 
        accept_multiple_files=True
    )

    mtr_files = st.file_uploader(
        "2. Upload ALL MTR Reports (.csv)", 
        type=['csv'], 
        accept_multiple_files=True
    )
    st.markdown("---")


# --- 3. Main Logic Execution ---

if payment_files and mtr_files:
    # Process files
    df_financial_master, df_payment_raw_breakdown = process_payment_files(payment_files)
    df_logistics_master = process_mtr_files(mtr_files)

    # Check for errors in file processing before continuing
    if df_financial_master.empty or df_logistics_master.empty:
        st.error("Data processing failed. Please check file formatting or look for error messages above.")
        st.stop()
    
    # Create Final Reconciliation DF
    df_reconciliation = create_final_reconciliation_df(df_financial_master, df_logistics_master)
    
    
    # --- Dashboard Display ---
    
    # KPI Cards (Key Metrics) - Use Total_Fees_KPI for the sum
    total_orders = df_reconciliation.shape[0]
    total_mtr_billed = df_reconciliation['MTR Invoice Amount'].sum()
    total_payment_fetched = df_reconciliation['Net Payment'].sum()
    total_fees = df_reconciliation['Total_Fees_KPI'].sum() # Updated to use the new total column
    total_tax = df_reconciliation['Total_Tax_TCS_TDS'].sum()

    st.subheader("Key Business Metrics (Based on Item Reconciliation)")
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
    col_kpi1.metric("Total Reconciled Items", f"{total_orders:,}")
    col_kpi2.metric("Total MTR Invoiced", f"INR {total_mtr_billed:,.2f}")
    col_kpi3.metric("Total Net Payment", f"INR {total_payment_fetched:,.2f}")
    col_kpi4.metric("Total Amazon Fees", f"INR {total_fees:,.2f}")
    col_kpi5.metric("Total Tax/TCS/TDS", f"INR {total_tax:,.2f}")
    
    st.markdown("---")

    # Order ID Selection is now for the filtered display of the summary
    st.header("1. Item-Level Reconciliation Summary (MTR Details + Classified Charges)")
    
    order_id_list = ['All Orders'] + sorted(df_reconciliation['OrderID'].unique().tolist())
    selected_order_id = st.selectbox("👉 Select Order ID to Filter Summary:", order_id_list)

    if selected_order_id != 'All Orders':
        df_display = df_reconciliation[df_reconciliation['OrderID'] == selected_order_id]
    else:
        df_display = df_reconciliation.sort_values(by='OrderID', ascending=True)
    
    # Display the primary reconciliation table
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- EXPORT SECTION ---
    st.header("2. Download Full Reconciliation Report")
    st.info("The Excel file will contain a single sheet with Item Details and all Classified Charges.")

    excel_data = convert_to_excel(df_reconciliation) # Updated function call
    
    st.download_button(
        label="Download Full Excel Report (Reconciliation Summary)",
        data=excel_data,
        file_name='amazon_reconciliation_summary.xlsx', # Renamed file for clarity
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    
else:
    st.info("Please upload your Payment (.txt) and MTR (.csv) files in the sidebar to start the reconciliation. The dashboard will appear automatically once files are uploaded.")
