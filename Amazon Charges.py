import streamlit as st
import pandas as pd
import numpy as np # Numpy को import करना ज़रूरी है
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
    # Create a copy before formatting for Excel
    df_excel = df.copy()
    # Apply formatting for Excel export
    numeric_cols = [
        'MTR Invoice Amount', 'Net Payment', 'Total_Commission_Fee', 
        'Total_Fixed_Closing_Fee', 'Total_FBA_Pick_Pack_Fee', 
        'Total_FBA_Weight_Handling_Fee', 'Total_Technology_Fee', 
        'Total_Fees_KPI', 'Total_Tax_TCS_TDS', 'Product Cost', 
        'Product Profit/Loss'
    ]
    for col in numeric_cols:
        if col in df_excel.columns:
            # Round the data itself for export
            df_excel[col] = pd.to_numeric(df_excel[col], errors='coerce').fillna(0).round(2)
            
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_excel.to_excel(writer, sheet_name='Reconciliation_Summary', index=False)
    return output.getvalue()

def calculate_fee_total(df, keyword, name):
    """Calculates the absolute total amount for specific fee/tax keywords."""
    # Ensure 'amount-description' exists and handle potential NaN values
    if 'amount-description' not in df.columns:
        return pd.DataFrame({'OrderID': [], name: []}) # Return empty DataFrame if column missing
    
    # Fill NaN values in 'amount-description' before using .str.contains
    df_filtered = df.dropna(subset=['amount-description'])
    df_fee = df_filtered[df_filtered['amount-description'].astype(str).str.contains(keyword, case=False, na=False)]
    
    if df_fee.empty:
        # Return a DataFrame with the correct structure but no data if no match
         return pd.DataFrame({'OrderID': df['OrderID'].unique(), name: [0.0] * len(df['OrderID'].unique())})


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
                
                # Check if 'order-id' exists before dropping NaNs
                if 'order-id' in chunk.columns:
                    chunk.dropna(subset=['order-id'], inplace=True)
                else:
                    # If 'order-id' is missing entirely, this chunk is invalid
                    st.warning(f"Skipping part of {file.name} due to missing 'order-id' column.")
                    continue # Skip this chunk

                
                # Ensure all required columns are present before selecting
                if all(col in chunk.columns for col in required_cols_lower):
                     chunk_small = chunk[required_cols_lower].copy()
                     all_payment_data.append(chunk_small)
                else:
                    st.warning(f"Skipping part of {file.name} due to missing required columns after header check (this shouldn't normally happen).")

                del chunk
            
        except Exception as e:
            st.error(f"Error reading {file.name} (Payment TXT): The file structure is unexpected. Details: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    if not all_payment_data:
        st.error("No valid payment data was found in the TXT files or required columns were missing.")
        return pd.DataFrame(), pd.DataFrame()
        
    df_charge_breakdown = pd.concat(all_payment_data, ignore_index=True)
    
    if df_charge_breakdown.empty:
        st.error("Payment files were read, but no valid 'order-id' entries were found after processing.")
        return pd.DataFrame(), pd.DataFrame()
        
    df_charge_breakdown.rename(columns={'order-id': 'OrderID'}, inplace=True)
    df_charge_breakdown['OrderID'] = df_charge_breakdown['OrderID'].astype(str)
    df_charge_breakdown['amount'] = pd.to_numeric(df_charge_breakdown['amount'], errors='coerce').fillna(0)
    
    df_financial_master = df_charge_breakdown.groupby('OrderID')['amount'].sum().reset_index(name='Net_Payment_Fetched')
    
    # --- Fee Calculation ---
    # Merge results safely, ensuring OrderID is preserved even if a fee type is missing
    orders_df = pd.DataFrame({'OrderID': df_charge_breakdown['OrderID'].unique()})
    
    df_comm = calculate_fee_total(df_charge_breakdown, 'Commission', 'Total_Commission_Fee')
    df_fixed = calculate_fee_total(df_charge_breakdown, 'Fixed closing fee', 'Total_Fixed_Closing_Fee')
    df_pick = calculate_fee_total(df_charge_breakdown, 'Pick & Pack Fee', 'Total_FBA_Pick_Pack_Fee')
    df_weight = calculate_fee_total(df_charge_breakdown, 'Weight Handling Fee', 'Total_FBA_Weight_Handling_Fee')
    df_tech = calculate_fee_total(df_charge_breakdown, 'Technology Fee', 'Total_Technology_Fee')
    
    tax_descriptions = ['TCS', 'TDS', 'Tax']
    df_tax_summary = calculate_fee_total(df_charge_breakdown, '|'.join(tax_descriptions), 'Total_Tax_TCS_TDS')

    # Start with the base financial master and merge others onto it
    # Use left merge to keep all orders from df_financial_master
    for df_fee in [df_comm, df_fixed, df_pick, df_weight, df_tech, df_tax_summary]:
        # Ensure the fee DataFrame has OrderID before merging
        if 'OrderID' in df_fee.columns:
            df_financial_master = pd.merge(df_financial_master, df_fee, on='OrderID', how='left')
        else:
             # If a fee calculation failed or returned empty, add the column with zeros
             fee_col_name = df_fee.columns[1] # Assuming second column is the fee amount
             if fee_col_name not in df_financial_master.columns:
                 df_financial_master[fee_col_name] = 0.0


    # Fill NaNs that might have resulted from merges
    df_financial_master.fillna(0, inplace=True)

    
    # Calculate Total_Fees_KPI using only columns confirmed to exist
    fee_kpi_cols = [
        'Total_Commission_Fee', 'Total_Fixed_Closing_Fee',
        'Total_FBA_Weight_Handling_Fee', 'Total_Technology_Fee'
    ]
    # Sum only the columns that are actually present in the DataFrame
    df_financial_master['Total_Fees_KPI'] = df_financial_master[
        [col for col in fee_kpi_cols if col in df_financial_master.columns]
    ].sum(axis=1)

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
                # Process chunk immediately
                try:
                    chunk = chunk.loc[:, ~chunk.columns.str.contains('^Unnamed')]
                    cols_to_keep = [col for col in required_mtr_cols if col in chunk.columns]
                    if cols_to_keep: # Only process if required columns are found
                         chunk_small = chunk[cols_to_keep].copy()
                         all_mtr_data.append(chunk_small)
                except Exception as chunk_err:
                     st.warning(f"Skipping a section in {file.name} due to processing error: {chunk_err}")
                finally:
                     del chunk # Ensure chunk is deleted

        except Exception as e:
            st.error(f"Error reading {file.name} (MTR CSV): {e}")
            # Continue processing other files if possible, or return empty
            # Depending on desired behavior, you might want to `return pd.DataFrame()` here
            
    if not all_mtr_data:
        st.error("No valid MTR data could be processed from the CSV files.")
        return pd.DataFrame()
        
    try:
        df_mtr_raw = pd.concat(all_mtr_data, ignore_index=True)
    except Exception as concat_err:
         st.error(f"Error combining MTR data chunks: {concat_err}")
         return pd.DataFrame()

    
    df_mtr_raw.rename(columns={'Order Id': 'OrderID', 'Invoice Amount': 'MTR Invoice Amount'}, inplace=True)
    
    final_cols = [
        'Invoice Number', 'Invoice Date', 'Transaction Type', 'OrderID', 
        'Quantity', 'Sku', 'Ship From City', 'Ship To City', 'Ship To State', 
        'MTR Invoice Amount'
    ]
    
    # Add missing columns safely
    for col in final_cols:
        if col not in df_mtr_raw.columns:
            df_mtr_raw[col] = '' # Initialize with empty string or appropriate default
            
    # Ensure columns exist before selecting
    final_cols_present = [col for col in final_cols if col in df_mtr_raw.columns]
    df_logistics_master = df_mtr_raw[final_cols_present].copy()
    
    # Add any remaining missing columns (shouldn't happen if loop above worked)
    for col in final_cols:
         if col not in df_logistics_master.columns:
              df_logistics_master[col] = ''

    # Convert types carefully
    df_logistics_master['OrderID'] = df_logistics_master['OrderID'].astype(str)
    df_logistics_master['MTR Invoice Amount'] = pd.to_numeric(df_logistics_master['MTR Invoice Amount'], errors='coerce').fillna(0)
    df_logistics_master['Sku'] = df_logistics_master['Sku'].astype(str) 
    # Ensure Quantity is numeric, default to 1 if conversion fails or is missing
    df_logistics_master['Quantity'] = pd.to_numeric(df_logistics_master['Quantity'], errors='coerce').fillna(1).astype(int)


    
    return df_logistics_master


# We KEEP caching here, as this is a CPU-intensive merge operation
@st.cache_data(show_spinner="Merging data and finalizing calculations...")
def create_final_reconciliation_df(df_financial_master, df_logistics_master, df_cost_master):
    """Merges detailed MTR data with Payment Net Sale Value, Fees, Tax/TCS, and Product Cost."""
    
    # Ensure required columns exist before merging
    if df_logistics_master.empty:
         st.error("MTR data is empty, cannot proceed with merge.")
         return pd.DataFrame()
    if df_financial_master.empty:
         st.error("Financial data is empty, cannot proceed with merge.")
         return pd.DataFrame()
         
    if 'OrderID' not in df_logistics_master.columns or 'OrderID' not in df_financial_master.columns:
         st.error("Critical 'OrderID' column missing in MTR or Financial data.")
         return pd.DataFrame()


    # 1. Merge MTR data with financial data
    try:
        df_final = pd.merge(
            df_logistics_master, 
            df_financial_master, 
            on='OrderID', 
            how='left'
        )
    except Exception as merge_err:
        st.error(f"Error merging MTR and Financial data: {merge_err}")
        return pd.DataFrame()


    # Handle cases where MTR Invoice Amount might be missing or non-numeric before calculations
    df_final['MTR Invoice Amount'] = pd.to_numeric(df_final['MTR Invoice Amount'], errors='coerce').fillna(0)


    # 2. Get the total MTR Invoice Amount for each order
    df_final['Total_MTR_per_Order'] = df_final.groupby('OrderID')['MTR Invoice Amount'].transform('sum')
    
    # 3. Get the number of items for each order (for cases where Total_MTR is 0)
    df_final['Item_Count_per_Order'] = df_final.groupby('OrderID')['OrderID'].transform('count')

    # 4. Determine the proportion for this specific item
    # Ensure division by zero doesn't occur if Total_MTR_per_Order is 0
    # Also handle Item_Count_per_Order being 0 (though unlikely)
    df_final['Proportion'] = np.where(
        (df_final['Total_MTR_per_Order'] > 0),
        df_final['MTR Invoice Amount'] / df_final['Total_MTR_per_Order'],
        np.where(df_final['Item_Count_per_Order'] > 0, 1 / df_final['Item_Count_per_Order'], 0) # Assign 0 proportion if count is 0
    )


    # 5. Get list of all financial columns that need to be allocated
    # Ensure these columns actually exist after the merge before allocating
    financial_cols_present = [col for col in df_financial_master.columns if col != 'OrderID' and col in df_final.columns]

    
    # 6. Apply the proportion to all existing financial columns
    for col in financial_cols_present:
        # Ensure the column is numeric before multiplying
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0) * df_final['Proportion']


    # Rename Net_Payment_Fetched to Net Payment *after* allocation
    if 'Net_Payment_Fetched' in df_final.columns:
        df_final.rename(columns={'Net_Payment_Fetched': 'Net Payment'}, inplace=True)
    elif 'Net Payment' not in df_final.columns: # If rename didn't happen and 'Net Payment' doesn't exist
         df_final['Net Payment'] = 0.0 # Add it with default 0


    # 7. Merge Product Cost
    if not df_cost_master.empty and 'Sku' in df_final.columns and 'Sku' in df_cost_master.columns:
        try:
            df_final = pd.merge(
                df_final,
                df_cost_master,
                on='Sku', 
                how='left'
            )
        except Exception as cost_merge_err:
             st.warning(f"Could not merge cost data: {cost_merge_err}. Proceeding without it.")
             df_final['Product Cost'] = 0.00 # Ensure column exists

    else:
        # Ensure 'Product Cost' column exists even if merge doesn't happen
        if 'Product Cost' not in df_final.columns:
             df_final['Product Cost'] = 0.00
             
    # Fill ALL remaining NaNs with 0 AFTER all merges
    df_final.fillna(0, inplace=True)


    # 8. Set Product Cost to 0 for refund/replacement/cancel types
    refund_types_lower = ['cancel refund', 'freereplacement', 'refund', 'cancel']
    
    if 'Transaction Type' in df_final.columns:
        standardized_transaction_type = df_final['Transaction Type'].astype(str).str.strip().str.lower()
        
        # Ensure 'Product Cost' is numeric before applying np.where
        df_final['Product Cost'] = pd.to_numeric(df_final['Product Cost'], errors='coerce').fillna(0)

        
        df_final['Product Cost'] = np.where(
            standardized_transaction_type.isin(refund_types_lower), 
            0, 
            df_final['Product Cost'] 
        )

    # 9. Calculate Profit/Loss
    # Ensure relevant columns are numeric before calculation
    df_final['Net Payment'] = pd.to_numeric(df_final['Net Payment'], errors='coerce').fillna(0)
    df_final['Product Cost'] = pd.to_numeric(df_final['Product Cost'], errors='coerce').fillna(0)
    df_final['Quantity'] = pd.to_numeric(df_final['Quantity'], errors='coerce').fillna(1).astype(int) # Default quantity to 1 if missing/invalid


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
    
    df_cost_master = pd.DataFrame() # Initialize empty
    df_financial_master = pd.DataFrame()
    df_logistics_master = pd.DataFrame()
    df_reconciliation = pd.DataFrame()

    # 1. Process Cost Sheet (if uploaded) - Uncached
    if cost_file:
         with st.spinner("Processing Cost Sheet..."):
              df_cost_master = process_cost_sheet(cost_file)
        
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

    # Check if processing failed before merging
    if df_financial_master.empty or df_logistics_master.empty:
        st.error("Data processing failed. Cannot proceed to merge. Check file formatting or error messages above.")
        st.stop()
        
    # 4. Create Final Reconciliation DF - This step IS CACHED
    # Ensure DataFrames are not None before passing
    if df_financial_master is not None and df_logistics_master is not None:
         df_reconciliation = create_final_reconciliation_df(df_financial_master, df_logistics_master, df_cost_master if df_cost_master is not None else pd.DataFrame())
    else:
         st.error("Failed to process input files correctly. Cannot create final report.")
         st.stop()

    # Final check if reconciliation DataFrame is valid
    if df_reconciliation.empty:
         st.error("Failed to create the final reconciliation report. Please review the processing steps and input files.")
         st.stop()

    
    # --- Dashboard Display ---
    try:
        # Ensure columns exist before summing/calculating metrics
        total_items = df_reconciliation.shape[0]
        total_mtr_billed = df_reconciliation['MTR Invoice Amount'].sum() if 'MTR Invoice Amount' in df_reconciliation.columns else 0
        total_payment_fetched = df_reconciliation['Net Payment'].sum() if 'Net Payment' in df_reconciliation.columns else 0
        total_fees = df_reconciliation['Total_Fees_KPI'].sum() if 'Total_Fees_KPI' in df_reconciliation.columns else 0
        total_tax = df_reconciliation['Total_Tax_TCS_TDS'].sum() if 'Total_Tax_TCS_TDS' in df_reconciliation.columns else 0
        
        # Calculate total product cost safely
        if 'Product Cost' in df_reconciliation.columns and 'Quantity' in df_reconciliation.columns:
            total_product_cost = (df_reconciliation['Product Cost'] * df_reconciliation['Quantity']).sum()
        else:
            total_product_cost = 0
            
        total_profit = df_reconciliation['Product Profit/Loss'].sum() if 'Product Profit/Loss' in df_reconciliation.columns else 0

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
        
        # Ensure OrderID exists for sorting/filtering
        if 'OrderID' in df_reconciliation.columns:
            order_id_list = ['All Orders'] + sorted(df_reconciliation['OrderID'].unique().tolist())
            selected_order_id = st.selectbox("👉 Select Order ID to Filter Summary:", order_id_list)

            if selected_order_id != 'All Orders':
                df_display = df_reconciliation[df_reconciliation['OrderID'] == selected_order_id].copy() # Use copy to avoid SettingWithCopyWarning
            else:
                df_display = df_reconciliation.sort_values(by='OrderID', ascending=True).copy()
        else:
             st.warning("OrderID column not found in final data. Displaying all data.")
             df_display = df_reconciliation.copy()


        
        # --- FIX: Apply number formatting robustly ---
        format_dict = {}
        numeric_cols_to_format = [
            'MTR Invoice Amount', 'Net Payment', 'Total_Commission_Fee', 
            'Total_Fixed_Closing_Fee', 'Total_FBA_Pick_Pack_Fee', 
            'Total_FBA_Weight_Handling_Fee', 'Total_Technology_Fee', 
            'Total_Fees_KPI', 'Total_Tax_TCS_TDS', 'Product Cost', 
            'Product Profit/Loss'
        ]
        
        # Create format dict only for columns that exist in df_display
        for col in numeric_cols_to_format:
            if col in df_display.columns:
                format_dict[col] = '{:.2f}' # Format as string with 2 decimal places

        # Apply formatting using st.dataframe's built-in format capabilities
        st.dataframe(
            df_display, 
            column_config={
                 col: st.column_config.NumberColumn(format="%.2f") 
                 for col in format_dict.keys() # Apply NumberColumn format
             },
            use_container_width=True, 
            hide_index=True
        )
        # --- END OF FIX ---

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
    
    except Exception as display_err:
        st.error(f"An error occurred while displaying the dashboard: {display_err}")


    
else:
    st.info("Please upload your **Payment Reports (.zip)** and **MTR Reports (.csv)** in the sidebar to start the reconciliation. The dashboard will appear automatically once files are uploaded.")
