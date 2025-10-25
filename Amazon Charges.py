import streamlit as st
import pandas as pd
import numpy as np
import io

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Amazon Seller Reconciliation Dashboard")
st.title("💰 Amazon Seller Central Reconciliation Dashboard")
st.markdown("---")


# --- 1. Data Processing Functions ---

@st.cache_data(show_spinner="Processing Payment Files and Creating Financial Master...")
def process_payment_files(uploaded_payment_files):
    """Reads all uploaded TXT payment files, combines them, and creates two outputs:
    1. df_financial_master: A summary of Principal and Fees per OrderID (FIXED LOGIC).
    2. df_charge_breakdown: The raw data for detailed charge view (for export)."""
    
    all_payment_data = []
    
    # Loop through all uploaded TXT files
    for file in uploaded_payment_files:
        try:
            # Use sep='\t' for TXT files
            df_temp = pd.read_csv(io.StringIO(file.getvalue().decode("latin-1")), sep='\t', skipinitialspace=True)
            all_payment_data.append(df_temp)
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
            return pd.DataFrame(), pd.DataFrame() # Return empty on error
    
    df_payment_raw = pd.concat(all_payment_data, ignore_index=True)
    
    # Cleaning: Remove summary rows and convert amount to numeric
    df_payment_cleaned = df_payment_raw.dropna(subset=['order-id']).copy()
    df_payment_cleaned['amount'] = pd.to_numeric(df_payment_cleaned['amount'], errors='coerce').fillna(0)
    df_payment_cleaned.rename(columns={'order-id': 'OrderID'}, inplace=True)
    
    # Create Raw Payment Data for detailed breakdown view
    charge_breakdown_cols = ['OrderID', 'transaction-type', 'marketplace-name', 'amount-type', 'amount-description', 'amount', 'posted-date', 'settlement-id']
    df_charge_breakdown = df_payment_cleaned[charge_breakdown_cols]
    
    # --- PIVOTING: Create Financial Master Summary ---
    
    # 1. Total Principal (Revenue)
    principal_df = df_charge_breakdown[df_charge_breakdown['amount-description'] == 'Principal'].groupby('OrderID')['amount'].sum().reset_index(name='Total_Principal')
    
    # 2. Total Amazon Fees (Cost)
    fee_descriptions = [
        'FBA Pick & Pack Fee', 'FBA Weight Handling Fee', 'Commission', 
        'Fixed closing fee', 'Technology Fee', 'Refund commission',
        'FBA Pick & Pack Fee CGST', 'FBA Pick & Pack Fee SGST', 
        'FBA Weight Handling Fee CGST', 'FBA Weight Handling Fee SGST',
        'Commission IGST', 'Fixed closing fee IGST', 'Technology Fee IGST',
        'Refund commission IGST', 'Commission CGST', 'Commission SGST',
        'Fixed closing fee CGST', 'Fixed closing fee SGST'
    ]
    
    df_fees = df_charge_breakdown[df_charge_breakdown['amount-description'].isin(fee_descriptions)]
    fees_summary_df = df_fees.groupby('OrderID')['amount'].sum().reset_index(name='Total_Amazon_Fees')
    fees_summary_df['Total_Amazon_Fees'] = fees_summary_df['Total_Amazon_Fees'].abs()
    
    # --- FIX FOR VALUE ERROR ---
    # 1. Get a unique list of all OrderIDs involved (Robustness)
    all_order_ids = pd.concat([principal_df['OrderID'], fees_summary_df['OrderID']]).unique()
    df_financial_master = pd.DataFrame(all_order_ids, columns=['OrderID'])
    
    # 2. Left Merge Principal and Fees onto the Master OrderID list
    df_financial_master = pd.merge(df_financial_master, principal_df, on='OrderID', how='left').fillna({'Total_Principal': 0})
    df_financial_master = pd.merge(df_financial_master, fees_summary_df, on='OrderID', how='left').fillna({'Total_Amazon_Fees': 0})
    
    # Ensure columns are numeric
    df_financial_master['Total_Principal'] = pd.to_numeric(df_financial_master['Total_Principal'])
    df_financial_master['Total_Amazon_Fees'] = pd.to_numeric(df_financial_master['Total_Amazon_Fees'])

    return df_financial_master, df_charge_breakdown


@st.cache_data(show_spinner="Processing MTR Files and Creating Logistics Master...")
def process_mtr_files(uploaded_mtr_files):
    """Reads all uploaded CSV MTR files, combines them, and creates a summary of MTR billed amount per OrderID."""
    
    all_mtr_data = []
    
    for file in uploaded_mtr_files:
        try:
            df_temp = pd.read_csv(file) 
            all_mtr_data.append(df_temp)
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
            return pd.DataFrame()
        
    df_mtr_raw = pd.concat(all_mtr_data, ignore_index=True)
    df_mtr_raw.rename(columns={'Order Id': 'OrderID', 'Invoice Amount': 'MTR Invoice Amount'}, inplace=True)
    df_mtr_raw['MTR Invoice Amount'] = pd.to_numeric(df_mtr_raw['MTR Invoice Amount'], errors='coerce').fillna(0)
    
    df_logistics_master = df_mtr_raw.groupby('OrderID')['MTR Invoice Amount'].sum().reset_index(name='Total_MTR_Billed')
    
    return df_logistics_master


@st.cache_data(show_spinner="Merging data and finalizing calculations...")
def create_final_reconciliation_df(df_financial_master, df_logistics_master):
    """Merges financial and logistics master data and calculates final net settlement."""
    
    df_final = pd.merge(
        df_financial_master, 
        df_logistics_master, 
        on='OrderID', 
        how='outer'
    ).fillna(0)
    
    # Final Calculation: Principal - Amazon Fees - MTR Billed
    df_final['Net_Settlement'] = df_final['Total_Principal'] - df_final['Total_Amazon_Fees'] - df_final['Total_MTR_Billed']
    
    df_final['Status'] = np.select(
        [df_final['Net_Settlement'] > 0, df_final['Net_Settlement'] < 0],
        ['Profit', 'Loss'],
        default='Break Even'
    )

    return df_final


# --- EXPORT FUNCTION ---
@st.cache_data
def convert_to_excel(df_reconciliation, df_payment_raw_breakdown):
    """Creates a multi-sheet Excel file for export."""
    
    # Prepare Detailed Breakdown Data for Export
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
        st.error("Data processing failed for one or more files. Please check file content and try again.")
        st.stop()
    
    # Create Final Reconciliation DF
    df_reconciliation = create_final_reconciliation_df(df_financial_master, df_logistics_master)
    
    
    # --- Dashboard Display ---
    
    # KPI Cards (Key Metrics)
    total_orders = df_reconciliation.shape[0]
    total_principal = df_reconciliation['Total_Principal'].sum()
    total_fees = df_reconciliation['Total_Amazon_Fees'].sum()
    total_mtr = df_reconciliation['Total_MTR_Billed'].sum()
    net_settlement_value = df_reconciliation['Net_Settlement'].sum()

    st.subheader("Key Business Metrics")
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    col_kpi1.metric("Total Reconciled Orders", f"{total_orders:,}")
    col_kpi2.metric("Total Principal (Revenue)", f"INR {total_principal:,.2f}")
    col_kpi3.metric("Total Amazon Fees", f"INR {total_fees:,.2f}")
    col_kpi4.metric("Net Settlement Value", f"INR {net_settlement_value:,.2f}", 
                    delta=f"MTR Cost: -INR {total_mtr:,.2f}")
    
    st.markdown("---")

    # Order ID Selection is now for the filtered display of the summary
    st.header("1. Order-Level Reconciliation Summary")
    
    order_id_list = ['All Orders'] + sorted(df_reconciliation['OrderID'].unique().tolist())
    selected_order_id = st.selectbox("👉 Select Order ID to Filter Summary:", order_id_list)

    if selected_order_id != 'All Orders':
        df_display = df_reconciliation[df_reconciliation['OrderID'] == selected_order_id]
    else:
        df_display = df_reconciliation.sort_values(by='Net_Settlement', ascending=False)
    
    # Display the primary reconciliation table
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- EXPORT SECTION ---
    st.header("2. Download Full Reconciliation Report")
    st.info("The Excel file will contain two sheets: 1. Reconciliation Summary (Screen Data) and 2. Payment Breakdown (All charges per Order ID).")

    excel_data = convert_to_excel(df_reconciliation, df_payment_raw_breakdown)
    
    st.download_button(
        label="Download Full Excel Report (Summary + Breakdown)",
        data=excel_data,
        file_name='full_amazon_reconciliation_report.xlsx', # Change extension to xlsx
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    
else:
    st.info("Please upload your Payment (.txt) and MTR (.csv) files in the sidebar to start the reconciliation. The dashboard will appear automatically once files are uploaded.")
