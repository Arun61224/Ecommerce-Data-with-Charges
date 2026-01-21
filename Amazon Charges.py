import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
import gc # Garbage collection to free up memory

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Amazon Seller Reconciliation Dashboard")
st.title("💰 Amazon Seller Central Reconciliation Dashboard (Stable)")
st.markdown("---")

# --- HELPER FUNCTIONS (CACHED FOR STABILITY) ---

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
def process_cost_sheet_bytes(file_bytes, file_name):
    """Processes cost sheet from bytes to avoid file pointer issues."""
    required_cols = ['SKU', 'Product Cost']
    try:
        filename = file_name.lower()
        if filename.endswith(('.xlsx', '.xls')):
            df_cost = pd.read_excel(io.BytesIO(file_bytes))
        else:
            try:
                df_cost = pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8')
            except UnicodeDecodeError:
                df_cost = pd.read_csv(io.BytesIO(file_bytes), encoding='latin-1')
        
        df_cost.columns = [str(col).strip() for col in df_cost.columns]
        
        # Check for missing columns
        missing_cols = [col for col in required_cols if col not in df_cost.columns]
        if missing_cols:
             return None, f"Missing columns: {', '.join(missing_cols)}"
        
        df_cost.rename(columns={'SKU': 'Sku'}, inplace=True)
        df_cost['Sku'] = df_cost['Sku'].astype(str)
        df_cost['Product Cost'] = pd.to_numeric(df_cost['Product Cost'], errors='coerce').fillna(0)
        
        # Aggregate duplicates
        df_cost_master = df_cost.groupby('Sku')['Product Cost'].mean().reset_index(name='Product Cost')
        return df_cost_master, None

    except Exception as e:
        return None, str(e)

def calculate_fee_total(df, keyword, name):
    """Calculates fee totals (helper for payment processing)."""
    if 'amount-description' not in df.columns:
        return pd.DataFrame({'OrderID': pd.Series(dtype='str'), name: pd.Series(dtype='float')})

    df_filtered = df.dropna(subset=['amount-description'])
    df_fee = df_filtered[df_filtered['amount-description'].astype(str).str.contains(keyword, case=False, na=False)]

    if df_fee.empty:
         return pd.DataFrame({'OrderID': pd.Series(dtype='str'), name: pd.Series(dtype='float')})

    df_summary = df_fee.groupby('OrderID')['amount'].sum().reset_index(name=name)
    df_summary[name] = df_summary[name].abs()
    return df_summary

@st.cache_data(show_spinner="Processing Payment Files (This stays cached)...")
def process_payment_zips_bytes(zip_file_bytes_list):
    """Reads multiple ZIP bytes, extracts TXTs, and processes them."""
    all_payment_data = []
    required_cols_lower = ['order-id', 'amount-description', 'amount']
    
    for zip_bytes in zip_file_bytes_list:
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
                for name in zf.namelist():
                    if name.startswith('__MACOSX/') or name.endswith('/') or name.startswith('.'):
                        continue
                    if name.lower().endswith('.txt'):
                        with zf.open(name) as f:
                            content = f.read()
                            # Decoding logic
                            decoded_content = None
                            try:
                                decoded_content = content.decode("utf-8")
                            except UnicodeDecodeError:
                                try:
                                    decoded_content = content.decode("latin-1")
                                except:
                                    continue # Skip if cant decode
                            
                            if decoded_content:
                                try:
                                    chunk_iter = pd.read_csv(
                                        io.StringIO(decoded_content),
                                        sep='\t', skipinitialspace=True, header=0, chunksize=50000
                                    )
                                    for chunk in chunk_iter:
                                        chunk.columns = [str(col).strip().lower() for col in chunk.columns]
                                        if 'order-id' in chunk.columns and 'amount' in chunk.columns:
                                            # Filter required columns immediately to save memory
                                            cols_to_keep = [c for c in required_cols_lower if c in chunk.columns]
                                            chunk_small = chunk[cols_to_keep].copy()
                                            chunk_small.dropna(subset=['order-id'], inplace=True)
                                            all_payment_data.append(chunk_small)
                                except:
                                    continue
        except Exception:
            continue

    if not all_payment_data:
        return None, None

    try:
        df_charge_breakdown = pd.concat(all_payment_data, ignore_index=True)
        # Free memory
        del all_payment_data
        gc.collect()
    except:
        return None, None

    df_charge_breakdown.rename(columns={'order-id': 'OrderID'}, inplace=True)
    df_charge_breakdown['OrderID'] = df_charge_breakdown['OrderID'].astype(str)
    df_charge_breakdown['amount'] = pd.to_numeric(df_charge_breakdown['amount'], errors='coerce').fillna(0)

    # Master Financial grouping
    df_financial_master = df_charge_breakdown.groupby('OrderID')['amount'].sum().reset_index(name='Net_Payment_Fetched')

    # Fee Calculations
    df_comm = calculate_fee_total(df_charge_breakdown, 'Commission', 'Total_Commission_Fee')
    df_fixed = calculate_fee_total(df_charge_breakdown, 'Fixed closing fee', 'Total_Fixed_Closing_Fee')
    df_pick = calculate_fee_total(df_charge_breakdown, 'Pick & Pack Fee', 'Total_FBA_Pick_Pack_Fee')
    df_weight = calculate_fee_total(df_charge_breakdown, 'Weight Handling Fee', 'Total_FBA_Weight_Handling_Fee')
    df_tech = calculate_fee_total(df_charge_breakdown, 'Technology Fee', 'Total_Technology_Fee')
    df_tax_summary = calculate_fee_total(df_charge_breakdown, 'TCS|TDS|Tax', 'Total_Tax_TCS_TDS')

    for df_fee in [df_comm, df_fixed, df_pick, df_weight, df_tech, df_tax_summary]:
         if not df_fee.empty:
             df_financial_master = pd.merge(df_financial_master, df_fee, on='OrderID', how='left')
    
    df_financial_master.fillna(0, inplace=True)
    
    # Calculate KPI Fees
    fee_cols = ['Total_Commission_Fee', 'Total_Fixed_Closing_Fee', 'Total_FBA_Weight_Handling_Fee', 'Total_Technology_Fee']
    present_cols = [c for c in fee_cols if c in df_financial_master.columns]
    df_financial_master['Total_Fees_KPI'] = df_financial_master[present_cols].sum(axis=1)

    return df_financial_master, df_charge_breakdown

@st.cache_data(show_spinner="Processing MTR Files (This stays cached)...")
def process_mtr_bytes(mtr_file_bytes_list):
    """Reads multiple MTR CSV bytes."""
    all_mtr_data = []
    required_mtr_cols = ['Invoice Number', 'Invoice Date', 'Transaction Type', 'Order Id', 
                         'Quantity', 'Sku', 'Ship From City', 'Ship To City', 'Ship To State', 'Invoice Amount']

    for file_bytes in mtr_file_bytes_list:
        try:
            # Read entire file or chunks. Using chunks for safety.
            chunk_iter = pd.read_csv(io.BytesIO(file_bytes), chunksize=50000)
            for chunk in chunk_iter:
                try:
                    chunk = chunk.loc[:, ~chunk.columns.str.contains('^Unnamed')]
                    chunk.columns = [str(col).strip() for col in chunk.columns]
                    
                    cols_to_keep = [col for col in required_mtr_cols if col in chunk.columns]
                    if cols_to_keep:
                        all_mtr_data.append(chunk[cols_to_keep].copy())
                except:
                    continue
        except:
            continue

    if not all_mtr_data:
        return None

    try:
        df_mtr_raw = pd.concat(all_mtr_data, ignore_index=True)
        del all_mtr_data
        gc.collect()
    except:
        return None

    df_mtr_raw.rename(columns={'Order Id': 'OrderID', 'Invoice Amount': 'MTR Invoice Amount'}, inplace=True)
    
    # Ensure columns exist
    for col in required_mtr_cols:
        target_col = 'OrderID' if col == 'Order Id' else ('MTR Invoice Amount' if col == 'Invoice Amount' else col)
        if target_col not in df_mtr_raw.columns:
            df_mtr_raw[target_col] = ''

    df_logistics_master = df_mtr_raw.copy()
    df_logistics_master['OrderID'] = df_logistics_master['OrderID'].astype(str)
    df_logistics_master['MTR Invoice Amount'] = pd.to_numeric(df_logistics_master['MTR Invoice Amount'], errors='coerce').fillna(0)
    df_logistics_master['Sku'] = df_logistics_master['Sku'].astype(str)
    df_logistics_master['Quantity'] = pd.to_numeric(df_logistics_master['Quantity'], errors='coerce').fillna(1).astype(int)

    return df_logistics_master

@st.cache_data(show_spinner="Merging Data...")
def create_final_reconciliation_df(df_financial_master, df_logistics_master, df_cost_master):
    """Merges all datasets."""
    
    # Pre-check
    if df_logistics_master is None or df_financial_master is None:
        return pd.DataFrame()

    try:
        df_final = pd.merge(df_logistics_master, df_financial_master, on='OrderID', how='left')
    except:
        return pd.DataFrame()

    # Numeric conversions
    numeric_cols_needed = ['MTR Invoice Amount', 'Net_Payment_Fetched', 'Quantity']
    for col in numeric_cols_needed:
         if col not in df_final.columns: df_final[col] = 0
         df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0)

    # Proportion Logic
    df_final['Total_MTR_per_Order'] = df_final.groupby('OrderID')['MTR Invoice Amount'].transform('sum')
    df_final['Item_Count_per_Order'] = df_final.groupby('OrderID')['OrderID'].transform('count')
    
    df_final['Proportion'] = np.where(
        (df_final['Total_MTR_per_Order'] != 0),
        df_final['MTR Invoice Amount'] / df_final['Total_MTR_per_Order'],
        np.where(df_final['Item_Count_per_Order'] > 0, 1 / df_final['Item_Count_per_Order'], 0)
    )

    # Apply proportion to financial columns
    financial_cols_present = [col for col in df_financial_master.columns if col != 'OrderID' and col in df_final.columns]
    for col in financial_cols_present:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0) * df_final['Proportion']

    # Rename Net Payment
    if 'Net_Payment_Fetched' in df_final.columns:
        df_final.rename(columns={'Net_Payment_Fetched': 'Net Payment'}, inplace=True)
    elif 'Net Payment' not in df_final.columns:
         df_final['Net Payment'] = 0.0

    # Merge Cost
    if df_cost_master is not None and not df_cost_master.empty:
        df_final = pd.merge(df_final, df_cost_master, on='Sku', how='left')
    
    if 'Product Cost' not in df_final.columns:
        df_final['Product Cost'] = 0.0
    
    df_final['Product Cost'] = pd.to_numeric(df_final['Product Cost'], errors='coerce').fillna(0)

    # Transaction Type Logic (Refunds/Cancels)
    if 'Transaction Type' in df_final.columns:
        ttype = df_final['Transaction Type'].astype(str).str.strip().str.lower()
        
        conditions = [
            ttype.isin(['refund', 'freereplacement']),
            ttype.str.contains('cancel')
        ]
        choices = [
            -0.2 * df_final['Product Cost'], # Refund Cost
            -0.2 * df_final['Product Cost']  # Cancel Cost
        ]
        
        df_final['Product Cost'] = np.select(conditions, choices, default=df_final['Product Cost'])

    # Final Calc
    df_final['Product Profit/Loss'] = (
        df_final['Net Payment'] -
        (df_final['Product Cost'] * df_final['Quantity'])
    )

    # Cleanup
    df_final.drop(columns=['Total_MTR_per_Order', 'Item_Count_per_Order', 'Proportion'], inplace=True, errors='ignore')
    
    # Ensure all Fee columns exist for display
    expected_cols = ['Total_Commission_Fee', 'Total_Fixed_Closing_Fee', 'Total_FBA_Pick_Pack_Fee', 
                     'Total_FBA_Weight_Handling_Fee', 'Total_Technology_Fee', 'Total_Tax_TCS_TDS', 'Total_Fees_KPI']
    for col in expected_cols:
        if col not in df_final.columns: df_final[col] = 0.0

    return df_final

@st.cache_data
def convert_to_excel_bytes(df):
    """Converts dataframe to Excel bytes efficiently."""
    output = io.BytesIO()
    # Create a copy for formatting to not affect the original DF logic
    df_excel = df.copy()
    
    # Rounding for Excel
    numeric_cols = df_excel.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'Quantity':
            df_excel[col] = df_excel[col].round(2)

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_excel.to_excel(writer, sheet_name='Reconciliation_Summary', index=False)
    return output.getvalue()

# --- SIDEBAR UI ---

with st.sidebar:
    st.header("Upload Raw Data Files")

    st.subheader("Cost Sheet (Mandatory)")
    excel_template = create_cost_sheet_template()
    st.download_button(
        label="Download Cost Template 📥",
        data=excel_template,
        file_name='cost_sheet_template.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    cost_file = st.file_uploader("1. Product Cost Sheet (.xlsx/.csv)", type=['xlsx', 'csv'])

    st.markdown("---")
    st.subheader("Amazon Reports (Mandatory)")
    payment_zip_files = st.file_uploader("2. Payment Reports (.zip)", type=['zip'], accept_multiple_files=True)
    mtr_files = st.file_uploader("3. MTR Reports (.csv)", type=['csv'], accept_multiple_files=True)
    st.markdown("---")

# --- MAIN PAGE INPUTS ---

st.subheader("4. Other Monthly Expenses (Mandatory Inputs)")
col_exp1, col_exp2 = st.columns(2)
col_exp3, col_exp4 = st.columns(2)

with col_exp1:
    storage_fee = st.number_input("Monthly Storage Fee (INR)", min_value=0.0, value=0.0, step=100.0)
with col_exp2:
    ads_spends = st.number_input("Monthly Advertising Spends (INR)", min_value=0.0, value=0.0, step=100.0)
with col_exp3:
    total_salary = st.number_input("Total Salary (INR)", min_value=0.0, value=0.0, step=1000.0)
with col_exp4:
    miscellaneous_expenses = st.number_input("Miscellaneous Expenses (INR)", min_value=0.0, value=0.0, step=100.0)

st.markdown("---")

# --- MAIN LOGIC ---

if payment_zip_files and mtr_files:
    
    # 1. READ FILES INTO BYTES (Crucial for Streamlit Caching/Rerun stability)
    cost_bytes = cost_file.getvalue() if cost_file else None
    cost_name = cost_file.name if cost_file else ""
    
    payment_bytes_list = [f.getvalue() for f in payment_zip_files]
    mtr_bytes_list = [f.getvalue() for f in mtr_files]

    # 2. PROCESS COST
    df_cost_master = pd.DataFrame()
    if cost_bytes:
        df_cost_master, error_msg = process_cost_sheet_bytes(cost_bytes, cost_name)
        if error_msg:
            st.error(f"Cost Sheet Error: {error_msg}")
            st.stop()
    
    # 3. PROCESS PAYMENTS
    df_financial_master, _ = process_payment_zips_bytes(payment_bytes_list)
    if df_financial_master is None:
        st.error("Could not process Payment ZIPs. Please check if they contain valid .txt files.")
        st.stop()

    # 4. PROCESS MTR
    df_logistics_master = process_mtr_bytes(mtr_bytes_list)
    if df_logistics_master is None:
        st.error("Could not process MTR CSVs. Please check file format.")
        st.stop()

    # 5. MERGE
    df_reconciliation = create_final_reconciliation_df(df_financial_master, df_logistics_master, df_cost_master)

    if df_reconciliation.empty:
         st.error("Reconciliation failed. No matching Order IDs found between Payment and MTR files.")
         st.stop()

    # 6. GENERATE EXCEL (Cached)
    excel_data = convert_to_excel_bytes(df_reconciliation)

    # --- DISPLAY DASHBOARD ---
    
    # KPIs
    total_items = len(df_reconciliation)
    total_payment = df_reconciliation['Net Payment'].sum()
    total_mtr = df_reconciliation['MTR Invoice Amount'].sum()
    total_fees = df_reconciliation['Total_Fees_KPI'].sum()
    
    total_prod_cost = (df_reconciliation['Product Cost'] * df_reconciliation['Quantity']).sum()
    
    gross_profit = df_reconciliation['Product Profit/Loss'].sum()
    total_expenses = storage_fee + ads_spends + total_salary + miscellaneous_expenses
    net_profit = gross_profit - total_expenses

    st.subheader("Key Business Metrics")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    kpi1.metric("Net Payment", f"₹ {total_payment:,.0f}")
    kpi2.metric("MTR Invoiced", f"₹ {total_mtr:,.0f}")
    kpi3.metric("Total Fees", f"₹ {total_fees:,.0f}")
    kpi4.metric("Product Cost", f"₹ {total_prod_cost:,.0f}")
    kpi5.metric("NET PROFIT", f"₹ {net_profit:,.0f}", delta=f"Exp: ₹ {total_expenses:,.0f}", delta_color="normal")

    st.markdown("---")

    # DATA TABLE
    st.header("1. Item-Level Reconciliation Summary")
    
    # Filter
    if 'OrderID' in df_reconciliation.columns:
        search_order = st.text_input("🔍 Search Order ID (Optional)", "")
        if search_order:
            df_display = df_reconciliation[df_reconciliation['OrderID'].str.contains(search_order, case=False)]
        else:
            df_display = df_reconciliation.head(200) # Show only first 200 rows to prevent browser crash
            st.caption("Showing top 200 rows for performance. Download Excel for full data.")
    else:
        df_display = df_reconciliation.head(200)

    # Display configuration
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Net Payment": st.column_config.NumberColumn(format="₹ %.2f"),
            "MTR Invoice Amount": st.column_config.NumberColumn(format="₹ %.2f"),
            "Product Profit/Loss": st.column_config.NumberColumn(format="₹ %.2f"),
        }
    )

    st.markdown("---")

    # DOWNLOAD
    st.header("2. Download Full Report")
    
    st.download_button(
        label="📥 Download Full Excel Report",
        data=excel_data,
        file_name='Amazon_Reconciliation_Final.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        key='download_final_report'
    )

else:
    # Default State (No files)
    total_expenses = storage_fee + ads_spends + total_salary + miscellaneous_expenses
    st.subheader("Expenses Summary (No Files Uploaded)")
    st.metric("Total Expenses Input", f"₹ {total_expenses:,.2f}")
    st.info("👈 Please upload Cost Sheet, Payment Zips, and MTR CSVs in the sidebar.")

