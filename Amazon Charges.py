import streamlit as st
import pandas as pd
import numpy as np # Numpy ko import karna zaroori hai
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

# Uncached function
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
        st.error(f"Error reading Cost Sheet ({uploaded_file.name}): Please ensure the file is an Excel file with 'SKU' and 'Product Cost' columns. Details: {e}")
        return pd.DataFrame()

@st.cache_data
def convert_to_excel(df):
    """Converts the final DataFrame into an Excel file bytes object."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Reconciliation_Summary', index=False)
    return output.getvalue()

def calculate_fee_total(df, keyword, name):
    """Calculates the absolute total amount for specific fee/tax keywords."""
    df_fee = df[df['amount-description'].str.contains(keyword, case=False, na=False)]
    df_summary = df_fee.groupby('OrderID')['amount'].sum().reset_index(name=name)
    df_summary[name] = df_summary[name].abs()
    return df_summary

# Uncached helper for ZIP processing
def process_payment_zip_file(uploaded_zip_file):
    """
    Reads a single uploaded ZIP file, extracts contents, and returns a list of 
    pseudo-file objects for all Payment (.txt) files.
    """
    payment_files = []

    try:
        with zipfile.ZipFile(uploaded_zip_file, 'r') as zf:
            for name in zf.namelist():
                if name.startswith('__MACOSX/') or name.endswith('/') or name.startswith('.'):
                    continue

                if name.lower().endswith('.txt'):
                    file_content_bytes = zf.read(name)
                    
                    pseudo_file = type('FileUploaderObject', (object,), {
                        'name': name,
                        'getvalue': lambda *args, b=file_content_bytes: b,
                        'read': lambda *args, b=file_content_bytes: b
                    })()
                    payment_files.append(pseudo_file)

    except zipfile.BadZipFile:
        st.error(f"Error: The uploaded file {uploaded_zip_file.name} is not a valid ZIP file.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred during unzipping {uploaded_zip_file.name}: {e}")
        return []

    return payment_files # List of pseudo-file objects

# --- 1. Data Processing Functions ---

# Uncached function
def process_payment_files(uploaded_payment_files):
    """Reads all payment file objects in chunks to save memory."""
    
    all_payment_data = [] 
    
    required_cols_lower = ['order-id', 'amount-description', 'amount']

    for file in uploaded_payment_files:
        try:
            chunk_iter = pd.read_csv(
                io.StringIO(file.getvalue().decode("latin-1")), 
                sep='\t', 
                skipinitialspace=True,
                header=0, # Assume header is on the FIRST line
                chunksize=100000 
            )

            first_chunk = True
            for chunk in chunk_iter:
                chunk.columns = [str(col).strip().lower() for col in chunk.columns]
                
                if first_chunk:
                    missing_cols = [col for col in required_cols_lower if col not in chunk.columns]
                    if missing_cols:
                        st.error(f"Error in {file.name}: The file is missing essential columns: {', '.join(missing_cols)}. Please check your file's header row for typos.")
                        return pd.DataFrame(), pd.DataFrame() 
                    first_chunk = False
                
                chunk.dropna(subset=['order-id'], inplace=True)
                
                chunk_small = chunk[required_cols_lower].copy()
                
                all_payment_data.append(chunk_small)
                del chunk
            
        except Exception as e:
            st.error(f"Error reading {file.name} (Payment TXT): The file structure is unexpected. Details: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    if not all_payment_data:
        st.error("No valid payment data was found in the TXT files.")
        return pd.DataFrame(), pd.DataFrame()
        
    df_charge_breakdown = pd.concat(all_payment_data, ignore_index=True)
    
    if df_charge_breakdown.empty:
        st.error("Payment files were read, but no valid 'order-id' entries were found.")
        return pd.DataFrame(), pd.DataFrame()
        
    df_charge_breakdown.rename(columns={'order-id': 'OrderID'}, inplace=True)
    df_charge_breakdown['OrderID'] = df_charge_breakdown['OrderID'].astype(str)
    df_charge_breakdown['amount'] = pd.to_numeric(df_charge_breakdown['amount'], errors='coerce').fillna(0)
    
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
        df_financial_master['Total_FBA_Weight_Handling_Fee'] +
        df_financial_master['Total_Technology_Fee']
    )
    
    return df_financial_master, df_charge_breakdown 

# Uncached function
def process_mtr_files(uploaded_mtr_files):
    """Reads all uploaded CSV MTR files (file objects) and concatenates them."""
    
    all_mtr_data = []
    
    required_mtr_cols = [
        'Invoice Number', 'Invoice Date', 'Transaction Type', 'Order Id', 
        'Quantity', 'Sku', 'Ship From City', 'Ship To City', 'Ship To State', 
        'Invoice Amount'
    ]
    
    for file in uploaded_mtr_files:
        try:
            chunk_iter = pd.read_csv(file, chunksize=100000)
            
            for chunk in chunk_iter:
                chunk = chunk.loc[:, ~chunk.columns.str.contains('^Unnamed')]
                
                cols_to_keep = [col for col in required_mtr_cols if col in chunk.columns]
                chunk_small = chunk[cols_to_keep].copy()

                all_mtr_data.append(chunk_small)
                del chunk

        except Exception as e:
            st.error(f"Error reading {file.name} (MTR CSV): {e}")
            return pd.DataFrame()

    if not all_mtr_data:
        st.error("No valid MTR data was found in the CSV files.")
        return pd.DataFrame()
        
    df_mtr_raw = pd.concat(all_mtr_data, ignore_index=True)
    
    df_mtr_raw.rename(columns={'Order Id': 'OrderID', 'Invoice Amount': 'MTR Invoice Amount'}, inplace=True)
    
    final_cols = [
        'Invoice Number', 'Invoice Date', 'Transaction Type', 'OrderID', 
        'Quantity', 'Sku', 'Ship From City', 'Ship To City', 'Ship To State', 
        'MTR Invoice Amount'
    ]
    
    for col in final_cols:
        if col not in df_mtr_raw.columns:
            df_mtr_raw[col] = ''
    
    df_logistics_master = df_mtr_raw[final_cols].copy()
    
    df_logistics_master['OrderID'] = df_logistics_master['OrderID'].astype(str)
    df_logistics_master['MTR Invoice Amount'] = pd.to_numeric(df_logistics_master['MTR Invoice Amount'], errors='coerce').fillna(0)
    df_logistics_master['Sku'] = df_logistics_master['Sku'].astype(str) 
    
    return df_logistics_master


# We KEEP caching here, as this is a CPU-intensive merge operation
@st.cache_data(show_spinner="Merging data and finalizing calculations...")
def create_final_reconciliation_df(df_financial_master, df_logistics_master, df_cost_master):
    """Merges detailed MTR data with Payment Net Sale Value, Fees, Tax/TCS, and Product Cost."""
    
    # 1. Merge MTR data with financial data
    df_final = pd.merge(
        df_logistics_master, 
        df_financial_master, 
        on='OrderID', 
        how='left'
    )

    # 2. Get the total MTR Invoice Amount for each order
    df_final['Total_MTR_per_Order'] = df_final.groupby('OrderID')['MTR Invoice Amount'].transform('sum')
    
    # 3. Get the number of items for each order (for cases where Total_MTR is 0)
    df_final['Item_Count_per_Order'] = df_final.groupby('OrderID')['OrderID'].transform('count')

    # 4. Determine the proportion for this specific item
    df_final['Proportion'] = np.where(
        df_final['Total_MTR_per_Order'] > 0, # Case 1: If Total MTR is not zero
        df_final['MTR Invoice Amount'] / df_final['Total_MTR_per_Order'], # Use MTR ratio
        1 / df_final['Item_Count_per_Order'] # Case 2: If Total MTR is zero, split equally
    )

    # 5. Get list of all financial columns that need to be allocated
    financial_cols_to_allocate = list(df_financial_master.columns.drop('OrderID'))
    
    # 6. Apply the proportion to all financial columns
    for col in financial_cols_to_allocate:
        df_final[col] = df_final[col] * df_final['Proportion']

    # Rename Net_Payment_Fetched to Net Payment *after* allocation
    df_final.rename(columns={'Net_Payment_Fetched': 'Net Payment'}, inplace=True)

    # 7. Merge Product Cost
    if not df_cost_master.empty:
        df_final = pd.merge(
            df_final,
            df_cost_master,
            on='Sku', 
            how='left'
        )
    else:
        df_final['Product Cost'] = 0.00
    
    df_final.fillna(0, inplace=True)

    # 8. Set Product Cost to 0 for refund/replacement/cancel types
    
    # --- CHANGE: ADDED 'cancel' TO THE LIST ---
    refund_types_lower = ['cancel refund', 'freereplacement', 'refund', 'cancel']
    
    if 'Transaction Type' in df_final.columns:
        standardized_transaction_type = df_final['Transaction Type'].astype(str).str.strip().str.lower()
        
        df_final['Product Cost'] = np.where(
            standardized_transaction_type.isin(refund_types_lower), 
            0, # Set cost to 0 if it's in the list
            df_final['Product Cost'] # Otherwise, keep original cost
        )
    # --- END OF CHANGE ---

    # 9. Calculate Profit/Loss
    df_final['Product Profit/Loss'] = (
        df_final['Net Payment'] - 
        (df_final['Product Cost'] * df_final['Quantity'])
    )

    # Clean up helper columns
    df_final.drop(columns=['Total_MTR_per_Order', 'Item_Count_per_Order', 'Proportion'], inplace=True, errors='ignore')

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
    
    payment_zip_files = st.file_uploader(
        "2. Upload ALL Payment Reports in **one or more Zipped Folders** (.zip)", 
        type=['zip'],
        accept_multiple_files=True 
    )
    
    mtr_files = st.file_uploader(
        "3. Upload ALL MTR Reports (.csv)", 
        type=['csv'], 
        accept_multiple_files=True
    )
    st.markdown("---")


# --- 3. Main Logic Execution ---

if payment_zip_files and mtr_files:
    
    # 1. Process Cost Sheet (if uploaded) - Uncached
    if cost_file:
        df_cost_master = process_cost_sheet(cost_file)
    else:
        df_cost_master = pd.DataFrame()
        
    # 2. Process Payment ZIP (uncached)
    all_payment_file_objects = [] 
    with st.spinner("Unzipping Payment files..."): 
        for zip_file in payment_zip_files: 
            payment_file_objects = process_payment_zip_file(zip_file)
            all_payment_file_objects.extend(payment_file_objects) 
    
    if not all_payment_file_objects: 
        st.error("ZIP file(s) processed, but no Payment (.txt) files found inside.")
        st.stop()
        
    # 3. Process files - Call the UNCACHED, memory-efficient functions
    with st.spinner("Processing Payment files... (This may take a while for large files)"):
        df_financial_master, df_payment_raw_breakdown = process_payment_files(all_payment_file_objects) 
    
    with st.spinner("Processing MTR files... (This may take a while for large files)"):
        df_logistics_master = process_mtr_files(mtr_files)

    if df_financial_master.empty or df_logistics_master.empty:
        st.error("Data processing failed. Please check file formatting or look for error messages above.")
        st.stop()
        
    # 4. Create Final Reconciliation DF - This step IS CACHED
    df_reconciliation = create_final_reconciliation_df(df_financial_master, df_logistics_master, df_cost_master)
    
    
    # --- Dashboard Display ---
    
    total_items = df_reconciliation.shape[0]
    total_mtr_billed = df_reconciliation['MTR Invoice Amount'].sum()
    total_payment_fetched = df_reconciliation['Net Payment'].sum()
    total_fees = df_reconciliation['Total_Fees_KPI'].sum()
    total_tax = df_reconciliation['Total_Tax_TCS_TDS'].sum()
    total_product_cost = (df_reconciliation['Product Cost'] * df_reconciliation['Quantity']).sum() 
    total_profit = df_reconciliation['Product Profit/Loss'].sum()

    st.subheader("Key Business Metrics (Based on Item Reconciliation)")
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5, col_kpi6 = st.columns(6)
    
    col_kpi1.metric("Total Items", f"{total_items:,}")
    col_kpi2.metric("Total Net Payment", f"INR {total_payment_fetched:,.2f}")
    col_kpi3.metric("Total MTR Invoiced", f"INR {total_mtr_billed:,.2f}")
    col_kpi4.metric("Total Amazon Fees", f"INR {total_fees:.2f}")
    col_kpi5.metric("Total Product Cost", f"INR {total_product_cost:,.2f}")
    col_kpi6.metric("TOTAL PROFIT/LOSS", f"INR {total_profit:,.2f}", 
                     delta=f"Tax/TCS: INR {total_tax:.2f}")
    
    st.markdown("---")

    st.header("1. Item-Level Reconciliation Summary (MTR Details + Classified Charges)")
    
    order_id_list = ['All Orders'] + sorted(df_reconciliation['OrderID'].unique().tolist())
    selected_order_id = st.selectbox("👉 Select Order ID to Filter Summary:", order_id_list)

    if selected_order_id != 'All Orders':
        df_display = df_reconciliation[df_reconciliation['OrderID'] == selected_order_id]
    else:
        df_display = df_reconciliation.sort_values(by='OrderID', ascending=True)
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.markdown("---")

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
