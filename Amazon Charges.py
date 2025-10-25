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
    """Reads all uploaded TXT payment files, creates the financial summary (Net Sales) and raw breakdown."""
    
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
            df_temp = pd.read_csv(io.StringIO(file.getvalue().decode("latin-1")), 
                                  sep='\t', 
                                  skipinitialspace=True,
                                  header=1) # FIX: Use header=1 
            
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
    
    # Create Raw Payment Data for detailed breakdown view
    charge_breakdown_cols = ['OrderID', 'transaction-type', 'marketplace-name', 'amount-type', 'amount-description', 'amount', 'posted-date', 'settlement-id']
    df_charge_breakdown = df_payment_cleaned[charge_breakdown_cols]
    
    # --- PIVOTING: Calculate Total Net Amount (Revenue - Fees) per OrderID ---
    
    # Calculate Total Amount (Sum of all 'amount' entries for an order)
    # This automatically includes Principal, Fees, Tax, etc., and provides the Net Settlement amount from payment report
    df_financial_master = df_charge_breakdown.groupby('OrderID')['amount'].sum().reset_index(name='Net_Sale_Value_P')
    
    return df_financial_master, df_charge_breakdown # Financial master is now just Net_Sale_Value_P


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
    df_logistics_master = df_mtr_raw[required_mtr_cols].copy()
    
    # Cleaning
    df_logistics_master['OrderID'] = df_logistics_master['OrderID'].astype(str)
    df_logistics_master['MTR Invoice Amount'] = pd.to_numeric(df_logistics_master['MTR Invoice Amount'], errors='coerce').fillna(0)
    
    return df_logistics_master


@st.cache_data(show_spinner="Merging data and finalizing calculations...")
def create_final_reconciliation_df(df_financial_master, df_logistics_master):
    """Merges detailed MTR data with Payment Net Sale Value."""
    
    # Perform Left Merge: MTR data is the primary base, Payment data is fetched
    df_final = pd.merge(
        df_logistics_master, 
        df_financial_master, 
        on='OrderID', 
        how='left'
    ).fillna(0)
    
    # Rename Payment column as requested by user
    df_final.rename(columns={'Net_Sale_Value_P': 'Payment for OrderID'}, inplace=True)
    
    # Calculate difference
    df_final['MTR_vs_Payment_Difference'] = df_final['Payment for OrderID'] - df_final['MTR Invoice Amount']
    
    return df_final


# --- EXPORT FUNCTION ---
@st.cache_data
def convert_to_excel(df_reconciliation, df_payment_raw_breakdown):
    """Creates a multi-sheet Excel file for export."""
    
    df_breakdown_export = df_payment_raw_breakdown[[
        'OrderID', 'transaction-type', 'amount-type', 
        'amount-description', 'amount', 'posted-date', 
        'marketplace-name', 'settlement-id'
    ]].sort_values(by='OrderID')
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_reconciliation.to_excel(writer, sheet_name='Reconciliation Summary', index=False)
        df_breakdown_export.to_excel(writer, sheet_name='Payment Breakdown', index=False)
    
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
    
    # KPI Cards (Key Metrics)
    total_orders = df_reconciliation.shape[0]
    total_mtr_billed = df_reconciliation['MTR Invoice Amount'].sum()
    total_payment_fetched = df_reconciliation['Payment for OrderID'].sum()

    st.subheader("Key Business Metrics (Based on MTR Items)")
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi1.metric("Total Reconciled Items", f"{total_orders:,}")
    col_kpi2.metric("Total MTR Invoiced", f"INR {total_mtr_billed:,.2f}")
    col_kpi3.metric("Total Payment Fetched", f"INR {total_payment_fetched:,.2f}")
    
    st.markdown("---")

    # Order ID Selection is now for the filtered display of the summary
    st.header("1. Item-Level Reconciliation Summary")
    
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
    st.info("The Excel file will contain two sheets: 1. Reconciliation Summary (Item Details) and 2. Payment Breakdown (All charges).")

    excel_data = convert_to_excel(df_reconciliation, df_payment_raw_breakdown)
    
    st.download_button(
        label="Download Full Excel Report (Summary + Breakdown)",
        data=excel_data,
        file_name='full_amazon_reconciliation_report.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    
else:
    st.info("Please upload your Payment (.txt) and MTR (.csv) files in the sidebar to start the reconciliation. The dashboard will appear automatically once files are uploaded.")
