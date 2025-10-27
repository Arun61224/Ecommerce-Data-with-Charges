# ... (Keep all your existing imports and helper functions like create_cost_sheet_template, process_cost_sheet, process_payment_files, process_mtr_files, create_final_reconciliation_df) ...

import zipfile
import io

# --- NEW: ZIP PROCESSING FUNCTION ---

@st.cache_data(show_spinner="Unzipping and sorting files...")
def process_zip_file(uploaded_zip_file):
    """
    Reads a single uploaded ZIP file, extracts contents in memory,
    and sorts them into lists of Payment (.txt) and MTR (.csv) files.
    """
    payment_files = []
    mtr_files = []

    try:
        # Use a BytesIO object to handle the file in memory
        with zipfile.ZipFile(io.BytesIO(uploaded_zip_file.read()), 'r') as zf:
            for name in zf.namelist():
                # Ignore system files, directories, and files in macOS-generated folders
                if name.startswith('__MACOSX/'):
                    continue
                if name.endswith('/') or name.startswith('.'):
                    continue

                # Read the file's content
                file_content = zf.read(name)
                
                # Create a pseudo-file object that your existing functions expect
                # The io.BytesIO is wrapped in a class-like structure to mimic the st.file_uploader object's properties
                pseudo_file = type('FileUploaderObject', (object,), {
                    'name': name,
                    'getvalue': lambda: file_content,
                    'read': lambda: file_content # Sometimes Streamlit file object uses .read()
                })()

                if name.lower().endswith('.txt'):
                    payment_files.append(pseudo_file)
                elif name.lower().endswith('.csv'):
                    mtr_files.append(pseudo_file)

    except zipfile.BadZipFile:
        st.error("Error: The uploaded file is not a valid ZIP file.")
        return [], []
    except Exception as e:
        st.error(f"An unexpected error occurred during unzipping: {e}")
        return [], []

    return payment_files, mtr_files


# --- 2. File Upload Section --- (MODIFIED)

with st.sidebar:
    st.header("Upload Raw Data Files")
    
    # ... (Cost Sheet section remains the same) ...

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
    
    # MODIFIED: Single ZIP File Uploader
    zip_file = st.file_uploader(
        "2. Upload ALL Payment (.txt) and MTR (.csv) Reports in a **Single Zipped Folder** (.zip)", 
        type=['zip'], 
    )
    st.markdown("---")


# --- 3. Main Logic Execution (MODIFIED) ---

if zip_file:
    # Process the ZIP file first to get the individual files
    payment_files, mtr_files = process_zip_file(zip_file)
    
    if not payment_files or not mtr_files:
        st.error("The uploaded ZIP file must contain at least one Payment (.txt) file and one MTR (.csv) file.")
        st.stop()
        
    # Process files
    df_financial_master, df_payment_raw_breakdown = process_payment_files(payment_files)
    df_logistics_master = process_mtr_files(mtr_files)

    # ... (Rest of the main logic remains the same) ...
    # Process Cost Sheet (only if uploaded)
    if cost_file:
        df_cost_master = process_cost_sheet(cost_file)
    else:
        df_cost_master = pd.DataFrame()

    # Check for errors in file processing before continuing
    if df_financial_master.empty or df_logistics_master.empty:
        st.error("Data processing failed. Please check file formatting or look for error messages above.")
        st.stop()
        
    # Create Final Reconciliation DF
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
    st.info("The Excel file will contain a single sheet with Item Details, Classified Charges, and Profit Calculation.")

    # Need to define convert_to_excel if it's not present (it was missing from your provided code, so adding a simple one)
    @st.cache_data
    def convert_to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Reconciliation_Summary', index=False)
        return output.getvalue()


    excel_data = convert_to_excel(df_reconciliation) 
    
    st.download_button(
        label="Download Full Excel Report (Reconciliation Summary)",
        data=excel_data,
        file_name='amazon_reconciliation_summary.xlsx', 
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    
else:
    st.info("Please upload your Payment (.txt) and MTR (.csv) files within a **Compressed ZIP Folder** in the sidebar to start the reconciliation. The dashboard will appear automatically once the ZIP file is uploaded.")
