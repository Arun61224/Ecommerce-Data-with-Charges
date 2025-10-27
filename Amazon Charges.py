import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile 

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Amazon Seller Reconciliation Dashboard")
st.title("💰 Amazon Seller Central Reconciliation Dashboard (Detailed)")
st.markdown("---")


# --- HELPER FUNCTIONS ---

@st.cache_data
def create_cost_sheet_template():
    """Generates a simple Excel template for Cost Sheet."""
    template_data = {
        'SKU': ['ExampleSKU-001', 'ExampleSKU-002'],
        'Product Cost': [150.50, 220.00]
    }
    df = pd.DataFrame(template_data)
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Cost_Sheet_Template', index=False)
    
    return output.getvalue()

@st.cache_data(show_spinner="Processing Cost Sheet...")
def process_cost_sheet(uploaded_file):
    """Reads the uploaded cost sheet and prepares it for merging."""
    try:
        df_cost = pd.read_excel(uploaded_file)
        df_cost.rename(columns={'SKU': 'Sku'}, inplace=True) 
        df_cost['Sku'] = df_cost['Sku'].astype(str)
        df_cost['Product Cost'] = pd.to_numeric(df_cost['Product Cost'], errors='coerce').fillna(0)
        
        df_cost_master = df_cost.groupby('Sku')['Product Cost'].mean().reset_index(name='Product Cost')

        return df_cost_master
    except Exception as e:
        st.error(f"Error reading Cost Sheet: Please ensure the file is an Excel file with 'SKU' and 'Product Cost' columns. Details: {e}")
        return pd.DataFrame()

@st.cache_data
def convert_to_excel(df):
    """Converts the final DataFrame into an Excel file bytes object."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Reconciliation_Summary', index=False)
    return output.getvalue()

# NEW GLOBAL HELPER FUNCTION (Moved from inside process_payment_files)
# This prevents caching conflict errors.
def calculate_fee_total(df, keyword, name):
    """Calculates the absolute total amount for specific fee/tax keywords."""
    df_fee = df[df['amount-description'].str.contains(keyword, case=False, na=False)]
    df_summary = df_fee.groupby('OrderID')['amount'].sum().reset_index(name=name)
    df_summary[name] = df_summary[name].abs()
    return df_summary

# FIX: Removed @st.cache_data and modified return value to hashable format
def process_payment_zip_file(uploaded_zip_file):
    """
    Reads a single uploaded ZIP file, extracts contents, and returns a list of 
    (file_content_string, file_name) for all Payment (.txt) files.
    """
    payment_data = []

    try:
        with zipfile.ZipFile(io.BytesIO(uploaded_zip_file.read()), 'r') as zf:
            for name in zf.namelist():
                # Ignore system files and directories
                if name.startswith('__MACOSX/') or name.endswith('/') or name.startswith('.'):
                    continue

                if name.lower().endswith('.txt'):
                    file_content_bytes = zf.read(name)
                    # Decode to string immediately
                    file_content_str = file_content_bytes.decode("latin-1")
                    payment_data.append((file_content_str, name))

    except zipfile.BadZipFile:
        st.error("Error: The uploaded file is not a valid ZIP file.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred during unzipping: {e}")
        return []

    return payment_data # List of (string, string) tuples

# --- 1. Data Processing Functions ---

@st.cache_data(show_spinner="Processing Payment Files and Creating Financial Master...")
def process_payment_files(uploaded_payment_data):
    """Reads all payment file contents (passed as strings), creates the financial summary."""
    
    all_payment_data = []
    
    cols = ['settlement-id', 'settlement-start-date', 'settlement-end-date', 'deposit-date', 
            'total-amount', 'currency', 'transaction-type', 'order-id', 'merchant-order-id', 
            'adjustment-id', 'shipment-id', 'marketplace-name', 'amount-type', 
            'amount-description', 'amount', 'fulfillment-id', 'posted-date', 
            'posted-date-time', 'order-item-code', 'merchant-order-item-id', 
            'merchant-adjustment-item-id', 'sku', 'quantity-purchased', 'promotion-id'] 

    # uploaded_payment_data is a list of (file_content_str, file_name)
    for file_content_str, file_name in uploaded_payment_data:
        try:
            # Use io.StringIO directly with the content string
            df_temp = pd.read_csv(io.StringIO(file_content_str), 
                                 sep='\t', 
                                 skipinitialspace=True,
                                 header=1)
            
            if len(df_temp.columns) > len(cols):
                df_temp = df_temp.iloc[:, :len(cols)]
            df_temp.columns = cols 
            
            all_payment_data.append(df_temp)
        except Exception as e:
            st.error(f"Error reading {file_name} (Payment TXT): The file structure is unexpected. Details: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    df_payment_raw = pd.concat(all_payment_data, ignore_index=True)
    df_payment_cleaned = df_payment_raw.dropna(subset=['order-id']).copy()
    
    df_payment_cleaned.rename(columns={'order-id': 'OrderID'}, inplace=True)
    df_payment_cleaned['OrderID'] = df_payment_cleaned['OrderID'].astype(str)
    df_payment_cleaned['amount'] = pd.to_numeric(df_payment_cleaned['amount'], errors='coerce').fillna(0)
    
    charge_breakdown_cols = ['OrderID', 'transaction-type', 'marketplace-name', 'amount-type', 'amount-description', 'amount', 'posted-date', 'settlement-id']
    df_charge_breakdown = df_payment_cleaned[charge_breakdown_cols]
    
    
    # --- PIVOTING: Calculate CLASSIFIED AMOUNTS per OrderID ---
    # Using the now global calculate_fee_total function
        
    df_financial_master = df_charge_breakdown.groupby('OrderID')['amount'].sum().reset_index(name='Net_Payment_Fetched')
    
    df_comm = calculate_fee_total(df_charge_breakdown, 'Commission', 'Total_Commission_Fee')
    df_fixed = calculate_fee_total(df_charge_breakdown, 'Fixed closing fee', 'Total_Fixed_Closing_Fee')
    df_pick = calculate_fee_total(df_charge_breakdown, 'Pick & Pack Fee', 'Total_FBA_Pick_Pack_Fee')
    df_weight = calculate_fee_total(df_charge_breakdown, 'Weight Handling Fee', 'Total_FBA_Weight_Handling_Fee')
    df_tech = calculate_fee_total(df_charge_breakdown, 'Technology Fee', 'Total_Technology_Fee')
    
    tax_descriptions = ['TCS', 'TDS', 'Tax']
    df_tax_summary = calculate_fee_total(df_charge_breakdown, '|'.join(tax_descriptions), 'Total_Tax_TCS_TDS')


    for df in [df_comm, df_fixed, df_pick, df_weight, df_tech, df_tax_summary]: 
        df_financial_master = pd.merge(df_financial_master, df, on='OrderID', how='left').fillna(0)
    
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
            df_temp = df_temp.loc[:, ~df_temp.columns.str.contains('^Unnamed')]
            
            all_mtr_data.append(df_temp)
        except Exception as e:
            st.error(f"Error reading {file.name} (MTR CSV): {e}")
            return pd.DataFrame()
        
    df_mtr_raw = pd.concat(all_mtr_data, ignore_index=True)
    
    df_mtr_raw.rename(columns={'Order Id': 'OrderID', 'Invoice Amount': 'MTR Invoice Amount'}, inplace=True)
    
    required_mtr_cols = [
        'Invoice Number', 'Invoice Date', 'Transaction Type', 'OrderID', 
        'Quantity', 'Sku', 'Ship From City', 'Ship To City', 'Ship To State', 
        'MTR Invoice Amount'
    ]
    
    for col in required_mtr_cols:
        if col not in df_mtr_raw.columns:
            df_mtr_raw[col] = ''
    
    df_logistics_master = df_mtr_raw[required_mtr_cols].copy()
    
    df_logistics_master['OrderID'] = df_logistics_master['OrderID'].astype(str)
    df_logistics_master['MTR Invoice Amount'] = pd.to_numeric(df_logistics_master['MTR Invoice Amount'], errors='coerce').fillna(0)
    df_logistics_master['Sku'] = df_logistics_master['Sku'].astype(str) 
    
    return df_logistics_master


@st.cache_data(show_spinner="Merging data and finalizing calculations...")
def create_final_reconciliation_df(df_financial_master, df_logistics_master, df_cost_master):
    """Merges detailed MTR data with Payment Net Sale Value, Fees, Tax/TCS, and Product Cost."""
    
    df_final = pd.merge(
        df_logistics_master, 
        df_financial_master, 
        on='OrderID', 
        how='left'
    ).fillna(0)
    
    df_final.rename(columns={'Net_Payment_Fetched': 'Net Payment'}, inplace=True)
    
    if not df_cost_master.empty:
        df_final = pd.merge(
            df_final,
            df_cost_master,
            on='Sku', 
            how='left'
        ).fillna({'Product Cost': 0})
        
        # Calculate Product Profit/Loss
        df_final['Product Profit/Loss'] = (
            df_final['Net Payment'] - 
            df_final['Total_Fees_KPI'] - 
            df_final['MTR Invoice Amount'] -
            (df_final['Product Cost'] * df_final['Quantity'])
        )
    else:
        df_final['Product Cost'] = 0.00
        df_final['Product Profit/Loss'] = 0.00 

    return df_final


# --- 2. File Upload Section ---

with st.sidebar:
    st.header("Upload Raw Data Files")
    
    st.subheader("Cost Sheet (Optional)")
    
    excel_template = create_cost_sheet_template()
    st.download_button(
        label="Download Cost Template 📥",
        data=excel_template,
        file_name='cost_sheet_template.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    
    cost_file = st.file_uploader(
        "1. Upload Product Cost Sheet (.xlsx)", 
        type=['xlsx'],
    )
    
    st.markdown("---")

    st.subheader("Amazon Reports (Mandatory)")
    
    # Payment Files via Single ZIP File Uploader
    payment_zip_file = st.file_uploader(
        "2. Upload ALL Payment Reports in a **Single Zipped Folder** (.zip)", 
        type=['zip'], 
    )
    
    # MTR Files via Multiple File Uploader
    mtr_files = st.file_uploader(
        "3. Upload ALL MTR Reports (.csv)", 
        type=['csv'], 
        accept_multiple_files=True
    )
    st.markdown("---")


# --- 3. Main Logic Execution ---

if payment_zip_file and mtr_files:
    # 1. Process the ZIP file for Payment reports
    with st.spinner("Unzipping Payment files..."): 
        payment_data_tuples = process_payment_zip_file(payment_zip_file)
    
    if not payment_data_tuples:
        st.error("ZIP file processed, but no Payment (.txt) files found inside. Please check the contents of your ZIP file.")
        st.stop()
        
    # 2. Process files - Pass the hashable list of tuples to the cached function
    df_financial_master, df_payment_raw_breakdown = process_payment_files(payment_data_tuples)
    df_logistics_master = process_mtr_files(mtr_files)

    # 3. Process Cost Sheet (only if uploaded)
    if cost_file:
        df_cost_master = process_cost_sheet(cost_file)
    else:
        df_cost_master = pd.DataFrame()

    # Check for errors in file processing before continuing
    if df_financial_master.empty or df_logistics_master.empty:
        st.error("Data processing failed. Please check file formatting or look for error messages above.")
        st.stop()
        
    # 4. Create Final Reconciliation DF
    df_reconciliation = create_final_reconciliation_df(df_financial_master, df_logistics_master, df_cost_master)
    
    
    # --- Dashboard Display ---
    
    # KPI Cards (Key Metrics)
    total_items = df_reconciliation.shape[0]
    total_mtr_billed = df_reconciliation['MTR Invoice Amount'].sum()
    total_payment_fetched = df_reconciliation['Net Payment'].sum()
    total_fees = df_reconciliation['Total_Fees_KPI'].sum()
    total_tax = df_reconciliation['Total_Tax_TCS_TDS'].sum()
    total_product_cost = df_reconciliation['Product Cost'].sum() 
    total_profit = df_reconciliation['Product Profit/Loss'].sum()

    st.subheader("Key Business Metrics (Based on Item Reconciliation)")
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5, col_kpi6 = st.columns(6)
    
    col_kpi1.metric("Total Items", f"{total_items:,}")
    col_kpi2.metric("Total Net Payment", f"INR {total_payment_fetched:,.2f}")
    col_kpi3.metric("Total MTR Invoiced", f"INR {total_mtr_billed:,.2f}")
    col_kpi4.metric("Total Amazon Fees", f"INR {total_fees:,.2f}")
    col_kpi5.metric("Total Product Cost", f"INR {total_product_cost:,.2f}")
    col_kpi6.metric("TOTAL PROFIT/LOSS", f"INR {total_profit:,.2f}", 
                     delta=f"Tax/TCS: INR {total_tax:,.2f}")
    
    st.markdown("---")

    # Order ID Selection 
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
    st.info("The Excel file will contain a single sheet with Item Details, Classified Charges, and Profit Calculation.")

    excel_data = convert_to_excel(df_reconciliation) 
    
    st.download_button(
        label="Download Full Excel Report (Reconciliation Summary)",
        data=excel_data,
        file_name='amazon_reconciliation_summary.xlsx', 
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    
else:
    st.info("Please upload your **Payment Reports (.zip)** and **MTR Reports (.csv)** in the sidebar to start the reconciliation. The dashboard will appear automatically once files are uploaded.")
