# --- START OF FILE app_revised.py ---

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Bank Transaction Analyzer")

# --- Categorization Logic ---
CATEGORY_KEYWORDS = {
    "Income": ["ndia"],
    "Groceries": ["haigh", "fresco", "hearthfire", "kombu","woolworths", "iga", "fuller fresh", "e.a. fuller","coles", "dorrigo deli"],
    "Fuel": ["bp", "caltex", "shell", "ampol", "fullers fuel"],
    "School Fees": ["st hilda's", "school fee", "edstart", "st hildas", "edutest"],
    "Clothing": ["clothing", "clothes", "fashion", "shoes", "apparel"],
    "Dogs and Chickens": ["lyka", "petbarn", "norco", "vet"],
    "Insurance": ["nrma", "aia", "insurance","nib"],
    "Meals & Dining": ["matilda", "coastal harvest", "maxsum", "burger", "thai", "indian", "black bear", "5 church street", "cafe", "restaurant", "mcdonalds", "kfc"],
    "Pharmacy": ["bellingen pharmacy", "chemist", "pharmacy", "pharm"],
    "Union Fees": ["union fee", "asu", "cpsu"],
    "Rent/Mortgage": ["real estate", "rent", "mortgage"],
    "Utilities": ["energy", "water", "gas", "telstra", "optus", "vodafone", "agl", "origin"],
    "Subscriptions": ["netflix", "spotify", "stan", "disney","apple", "primevideo"],
    "Shopping": ["kmart", "big w", "target", "amazon", "ebay"],
    "Transport": ["uber", "didi", "taxi", "opal", "public transport"],
    "Health:": ["outpost hair", "doctor", "dentist", "physio", "hospital", "medical", "bellingen healing", "medicare", "mcare benefits"], # Note: typo in original Health: -> Health
    "Ada": ["bun bun bao", "westpac cho ada", "savin ada", "sweet bellingen", "yy rock", "sens fushion"],
    "Home Maintenance": ["bunnings", "hardware", "home depot", "handyman", "gardening"],
    "Books": ["alternatives"],
    "Donations": ["childfund"],
    "Misc": ["misc"],
}
# Correcting potential typo in Health category key
if "Health:" in CATEGORY_KEYWORDS:
     CATEGORY_KEYWORDS["Health"] = CATEGORY_KEYWORDS.pop("Health:")


UNCATEGORIZED_LABEL = "Uncategorized"
ALL_CATEGORY_OPTIONS = [UNCATEGORIZED_LABEL] + sorted(list(CATEGORY_KEYWORDS.keys()))

def categorize_transaction(narrative, keywords_dict):
    if not isinstance(narrative, str):
        return UNCATEGORIZED_LABEL
    narrative_lower = narrative.lower()
    for category, keywords in keywords_dict.items():
        keywords_lower = [k.lower() for k in keywords]
        for keyword in keywords_lower:
            if keyword in narrative_lower:
                return category
    return UNCATEGORIZED_LABEL

# --- Callback function to update category in Session State ---
def update_category(row_index, selectbox_key):
    new_category = st.session_state[selectbox_key]
    if 'df_processed' in st.session_state and st.session_state.df_processed is not None and row_index in st.session_state.df_processed.index:
        st.session_state.df_processed.loc[row_index, 'Categories'] = new_category
    # Streamlit automatically reruns on widget change

# --- Calculation Functions (Keep as they were, ensure they handle None df) ---
def calculate_fortnightly_expenses(df):
    required_cols = ['Date', 'Debit Amount', 'Categories']
    # Check if df is None or required cols are missing
    if df is None or not all(col in df.columns for col in required_cols):
        # Avoid showing error if df is None initially, only if cols are missing from existing df
        if df is not None:
            missing_cols = [col for col in required_cols if col not in df.columns]
            st.error(f"Calculation Error: Missing columns {', '.join(missing_cols)}")
        return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0

    df_calc = df.copy()
    try:
        df_calc['Date'] = pd.to_datetime(df_calc['Date'])
        # Ensure 'Debit Amount' exists before converting
        if 'Debit Amount' in df_calc.columns:
            df_calc['Debit Amount'] = pd.to_numeric(df_calc['Debit Amount'], errors='coerce').fillna(0)
        else:
             st.error("Calculation Error: 'Debit Amount' column not found.")
             return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0
        df_calc['Categories'] = df_calc['Categories'].astype(str)
    except Exception as e:
        st.error(f"Error converting columns for calculation: {e}")
        return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0

    school_fees_category = "School Fees"
    df_expenses = df_calc[
        (df_calc['Debit Amount'] > 0) & (df_calc['Categories'] != school_fees_category)
    ].copy()

    if df_expenses.empty:
        st.info(f"No non-school fee expense transactions found for fortnightly calculation.")
        return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0

    # Ensure Date is the index before resampling
    if not isinstance(df_expenses.index, pd.DatetimeIndex):
        try:
            df_expenses.sort_values('Date', inplace=True)
            df_expenses.set_index('Date', inplace=True)
        except KeyError:
            st.error("Calculation Error: 'Date' column not found for indexing.")
            return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0
        except Exception as e:
            st.error(f"Error setting Date index for calculation: {e}")
            return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0

    fortnightly_sum = df_expenses['Debit Amount'].resample('14D', label='right', closed='right').sum()
    valid_fortnights = fortnightly_sum[fortnightly_sum > 0]
    average_expense = valid_fortnights.mean() if not valid_fortnights.empty else 0.0
    return fortnightly_sum.reset_index(), average_expense


# --- Plotting Functions (Keep as they were, ensure they handle None df) ---
def plot_expenses_timeseries(fortnightly_data):
    if fortnightly_data is None or fortnightly_data.empty: return None
    fig = px.line(fortnightly_data, x='Date', y='Debit Amount',
                  title="Fortnightly General Expenses Over Time (Excl. School Fees)",
                  markers=True, labels={'Debit Amount': 'Total Expenses ($)'})
    fig.update_layout(xaxis_title="Fortnight Period Ending", yaxis_title="Total Expenses ($)")
    return fig

def plot_category_pie(df):
    if df is None or 'Debit Amount' not in df.columns or 'Categories' not in df.columns: return None
    df_plot = df[df['Debit Amount'] > 0].copy()
    if df_plot.empty: return None
    expenses_by_cat = df_plot.groupby('Categories')['Debit Amount'].sum().reset_index()
    expenses_by_cat = expenses_by_cat[expenses_by_cat['Debit Amount'] > 0]
    if expenses_by_cat.empty: return None
    fig = px.pie(expenses_by_cat, names='Categories', values='Debit Amount', title="Expense Distribution by Category", hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label', sort=False)
    return fig

def plot_category_bar(df):
    if df is None or 'Debit Amount' not in df.columns or 'Categories' not in df.columns: return None
    df_plot = df[df['Debit Amount'] > 0].copy()
    if df_plot.empty: return None
    expenses_by_cat = df_plot.groupby('Categories')['Debit Amount'].sum().reset_index()
    expenses_by_cat = expenses_by_cat[expenses_by_cat['Debit Amount'] > 0]
    if expenses_by_cat.empty: return None
    expenses_by_cat = expenses_by_cat.sort_values('Debit Amount', ascending=False)
    fig = px.bar(expenses_by_cat, x='Categories', y='Debit Amount', title="Total Expenses per Category", labels={'Debit Amount': 'Total Expenses ($)'})
    fig.update_layout(xaxis_title="Category", yaxis_title="Total Expenses ($)")
    return fig

# --- Helper Function for Excel Download (Keep as is) ---
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if isinstance(df, pd.DataFrame):
            df.to_excel(writer, index=False, sheet_name='Categorized Transactions')
        else:
            pd.DataFrame().to_excel(writer, index=False, sheet_name='No Data')
    processed_data = output.getvalue()
    return processed_data

# --- Initialize Session State ---
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'processed_file_name' not in st.session_state:
    st.session_state.processed_file_name = None # Track the name of the file processed

# --- Streamlit App UI ---
st.title("💰 Bank Transaction Analysis Tool")
st.markdown("Upload your bank statement (Excel format) to categorize transactions and visualize spending.")

# --- File Upload ---
# Use a simple key, no need for the complex counter key here
uploaded_file = st.file_uploader(
    "Choose an Excel file (.xlsx)",
    type="xlsx",
    key="file_uploader"
)

# --- Processing Logic ---
# This block runs ONLY when a new file is uploaded
process_new_file = False
if uploaded_file is not None:
    # Check if the uploaded file name is different from the last one processed
    if uploaded_file.name != st.session_state.get('processed_file_name', None):
        process_new_file = True
        st.info(f"New file detected: '{uploaded_file.name}'. Processing...") # User feedback
    # Optional: Allow reprocessing if df is None but file is present (e.g., after error)
    # elif st.session_state.df_processed is None:
    #     process_new_file = True
    #     st.info(f"Reprocessing file: '{uploaded_file.name}'...")


if process_new_file:
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')

        # --- Data Preprocessing and Validation ---
        required_columns = ['Date', 'Narrative', 'Debit Amount', 'Credit Amount', 'Balance']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            st.error(f"Error: Missing required columns: {', '.join(missing)}. Please ensure your Excel file has these columns.")
            # Reset state as processing failed
            st.session_state.df_processed = None
            st.session_state.processed_file_name = None
        else:
            # --- Basic Data Cleaning ---
            # Convert Date early
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except Exception as e:
                st.warning(f"Could not convert 'Date' column to datetime: {e}. Dates might display incorrectly.")

            # Ensure numeric columns are numeric, fill NaNs
            for col in ['Debit Amount', 'Credit Amount', 'Balance']:
                 if col in df.columns:
                     df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # --- Automatic Categorization ---
            df_processed = df.copy()
            df_processed['Narrative'] = df_processed['Narrative'].fillna('').astype(str)

            if 'Categories' not in df_processed.columns:
                 df_processed['Categories'] = df_processed['Narrative'].apply(lambda x: categorize_transaction(x, CATEGORY_KEYWORDS))
            else:
                # Handle existing 'Categories' column more carefully
                df_processed['Categories'] = df_processed['Categories'].fillna(UNCATEGORIZED_LABEL).astype(str)
                mask_needs_categorizing = (df_processed['Categories'].str.strip() == '') | \
                                          (df_processed['Categories'].str.lower() == 'uncategorized') | \
                                          (df_processed['Categories'] == UNCATEGORIZED_LABEL)
                df_processed.loc[mask_needs_categorizing, 'Categories'] = df_processed.loc[mask_needs_categorizing, 'Narrative'].apply(lambda x: categorize_transaction(x, CATEGORY_KEYWORDS))

            # Store the successfully processed dataframe and filename in session state
            st.session_state.df_processed = df_processed
            st.session_state.processed_file_name = uploaded_file.name
            st.success(f"File '{uploaded_file.name}' processed successfully!")
            # *** Rerun essential after successful processing to update the display ***
            st.rerun()

    except Exception as e:
        st.error(f"An unexpected error occurred during file processing: {e}")
        st.exception(e)
        # Reset state on error
        st.session_state.df_processed = None
        st.session_state.processed_file_name = None


# --- Display and Interaction Area ---
# This block runs EVERY time, using data from session state if available
# It will run after the st.rerun() call upon successful processing

if st.session_state.df_processed is not None:
    st.write("---")
    st.header("📊 Transaction Data")
    st.write(f"Displaying data from: **{st.session_state.processed_file_name}**") # Show which file is active
    st.dataframe(st.session_state.df_processed) # Display current state

    # --- Manual Categorization for Uncategorized Transactions ---
    st.write("---")
    st.header("✏️ Manual Categorization")
    # Operate directly on the session state DataFrame for filtering, but use .copy() if modifying subsets
    df_display = st.session_state.df_processed
    uncategorized_mask = df_display['Categories'] == UNCATEGORIZED_LABEL
    uncategorized_count = uncategorized_mask.sum()

    if uncategorized_count > 0:
        st.warning(f"Found {uncategorized_count} transactions needing manual categorization.")
        st.markdown("**Assign a category using the dropdowns below:**")

        # Create columns for layout outside the loop for better performance
        col_date, col_narrative, col_amount, col_category = st.columns([1, 4, 1, 2])

        # Header row
        with col_date: st.write("**Date**")
        with col_narrative: st.write("**Narrative**")
        with col_amount: st.write("**Amount ($)**")
        with col_category: st.write("**Select Category**")
        st.divider() # Header divider

        # Iterate through the *indices* of uncategorized rows in the session state df
        for idx in df_display[uncategorized_mask].index:
            row = df_display.loc[idx] # Get the row data
            selectbox_key = f"category_select_{idx}"

            with col_date:
                # Check if Date is valid timestamp before formatting
                date_val = row.get('Date')
                if pd.notnull(date_val) and isinstance(date_val, pd.Timestamp):
                    st.write(date_val.strftime('%Y-%m-%d'))
                else:
                    st.write(str(date_val)) # Display as string if not expected format
            with col_narrative:
                st.write(row.get('Narrative', 'N/A'))
            with col_amount:
                amount = row.get('Debit Amount', 0)
                st.write(f"{amount:,.2f}" if pd.notnull(amount) else "N/A")
            with col_category:
                st.selectbox(
                    label=f"Category for row {idx}", # Unique hidden label
                    options=ALL_CATEGORY_OPTIONS,
                    # Find current index, default to Uncategorized if something weird happened
                    index=ALL_CATEGORY_OPTIONS.index(row.get('Categories', UNCATEGORIZED_LABEL)),
                    key=selectbox_key,
                    label_visibility="collapsed",
                    on_change=update_category,
                    args=(idx, selectbox_key)
                )
            # Consider removing the divider inside the loop if there are many items,
            # or use a lighter visual like st.container() with border=True
            # st.divider()

    else:
        st.success("✅ All transactions are currently categorized!")

    # --- Download Button (Uses potentially updated session state df) ---
    st.download_button(
        label="📥 Download Categorized Data as Excel",
        data=to_excel(st.session_state.df_processed), # Download the current state
        file_name=f'categorized_{st.session_state.processed_file_name}', # Use original filename
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    # --- Expense Calculations (Uses potentially updated session state df) ---
    st.write("---")
    st.header("💹 Fortnightly General Expense Analysis (Excl. School Fees)")
    # Pass the current state DataFrame
    fortnightly_expenses_df, avg_fortnightly_expense = calculate_fortnightly_expenses(st.session_state.df_processed)

    if fortnightly_expenses_df is not None and not fortnightly_expenses_df.empty:
        st.metric(label="Average Fortnightly General Expense", value=f"${avg_fortnightly_expense:,.2f}")
        with st.expander("View Fortnightly General Expense Data"):
            st.dataframe(fortnightly_expenses_df.style.format({"Date": "{:%Y-%m-%d}", "Debit Amount": "${:,.2f}"}))
    # No need for extra warnings here, calculate_fortnightly_expenses handles info/errors

    # --- Visualizations (Uses potentially updated session state df) ---
    st.write("---")
    st.header("📈 Visualizations")
    current_df_for_plots = st.session_state.df_processed # Use current state

    col3, col4 = st.columns(2) # Use different names if col1/col2 used above
    with col3:
        st.subheader("Expense Breakdown")
        fig_pie = plot_category_pie(current_df_for_plots)
        if fig_pie: st.plotly_chart(fig_pie, use_container_width=True)
        else: st.info("No expense data for pie chart.")

        fig_bar = plot_category_bar(current_df_for_plots)
        if fig_bar: st.plotly_chart(fig_bar, use_container_width=True)
        else: st.info("No expense data for bar chart.")

    with col4:
        st.subheader("Expenses Over Time")
        # Use the calculated fortnightly data
        fig_timeseries = plot_expenses_timeseries(fortnightly_expenses_df)
        if fig_timeseries: st.plotly_chart(fig_timeseries, use_container_width=True)
        else: st.info("No fortnightly expense data to plot.")

# --- Initial state message ---
elif not uploaded_file: # Only show if no file is uploaded AT ALL
    st.info("Awaiting Excel file upload...")
# If uploaded_file is not None, but df_processed is None, it means processing failed
# and the error message should already be displayed from the processing block.

# --- Sidebar ---
st.sidebar.title("About")
st.sidebar.info(
    """
    Analyze bank transactions:
    1. **Upload** Excel file.
    2. **Auto-categorized**.
    3. **Manually categorize** remaining items.
    4. View **summaries & charts**.
    5. **Download** results.
    """
)
st.sidebar.title("Categories & Keywords")
# Format keywords for display
display_keywords = {cat: ', '.join(kw) for cat, kw in CATEGORY_KEYWORDS.items()}
st.sidebar.expander("View Keywords (Code Level)").json(display_keywords)
st.sidebar.markdown("*(Edit `CATEGORY_KEYWORDS` in the script to permanently add keywords)*")

# --- END OF FILE app_revised.py ---