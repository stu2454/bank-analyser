# --- START OF FILE app.py ---

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Bank Transaction Analyzer")

# --- Categorization Logic ---
# Define keywords for each category (lowercase for case-insensitive matching)
# More specific keywords should come first in the list for a category if overlap exists
CATEGORY_KEYWORDS = {
    "Income": ["ndia", "iag"],
    "Groceries": ["prime qual", "peaches", "haigh", "fresco", "hearthfire", "kombu","woolworths", "iga", "fuller fresh", "e.a. fuller","coles", "dorrigo deli"],
    "Fuel/Car": ["bp", "caltex", "shell", "ampol", "fullers fuel", "reddy express", "ballina", "matade"],
    "School Fees": ["mspgold", "sports", "st hilda's", "school fee", "edstart", "st hildas", "edutest"],
    "Clothing": ["kmart", "myer", "blue illusion", "lorna jane", "bras", "slipsilk", "vivid", "clothing", "clothes", "fashion", "shoes", "apparel"], # Keep this added category
    "Dogs and Chickens": ["lyka", "petbarn", "norco", "vet"],
    "Insurance": ["nrma", "aia", "insurance","nib", "heather"],
    "Meals & Dining": ["gelato", "eric and deb", "hyde", "oporto", "sens fusion", "fushion", "bun bun bao", "eats","matilda", "coastal harvest", "maxsum", "burger", "thai", "indian", "black bear", "5 church street", "cafe", "restaurant", "fiume"],
    "Pharmacy": ["bellingen pharmacy", "chemist", "pharmacy", "pharm"],
    "Union Fees": ["union fee", "asu", "cpsu"],
    "Rent/Mortgage": ["real estate", "rent", "mortgage"],
    "Utilities": ["energy", "water", "gas", "telstra", "optus", "vodafone", "agl", "origin"],
    "Subscriptions": ["netflix", "spotify", "stan", "disney", "apple", "primevideo", "new york times", "chatgpt", "openai"],
    "Health/Personal Care": ["mecca", "bradley", "outpost hair", "doctor", "dentist", "physio", "hospital", "medical", "bellingen healing", "medicare", "mcare benefits"], # Corrected key
    "Ada": [ "westpac cho ada", "savin ada", "sweet bellingen", "yy rock"],
    "Home Maintenance": ["outdoor", "cleaner", "bunnings", "hardware", "handyman", "gardening"],
    "Books": ["alternatives", "book", "books", "bookstore", "library"],
    "Donations": ["childfund"],
    "Lotteries": ["lotto", "lottery", "lotteries"],
    "Misc": ["misc"],
    # Add more categories and keywords as needed (use lowercase)
}
# Ensure no legacy typo keys exist
if "Health:" in CATEGORY_KEYWORDS:
     CATEGORY_KEYWORDS["Health"] = CATEGORY_KEYWORDS.pop("Health:")


UNCATEGORIZED_LABEL = "Uncategorized"
# Create the list of category options for the dropdown dynamically
# Needs to be updated if CATEGORY_KEYWORDS changes during runtime (but doesn't here)
ALL_CATEGORY_OPTIONS = [UNCATEGORIZED_LABEL] + sorted(list(CATEGORY_KEYWORDS.keys()))

def categorize_transaction(narrative, keywords_dict):
    """
    Categorizes a transaction based on keywords in the narrative.
    """
    if not isinstance(narrative, str):
        return UNCATEGORIZED_LABEL
    narrative_lower = narrative.lower()
    for category, keywords in keywords_dict.items():
        # Ensure keywords are lowercase for comparison during categorization itself
        keywords_lower = [k.lower() for k in keywords]
        for keyword in keywords_lower:
            if keyword in narrative_lower:
                return category
    return UNCATEGORIZED_LABEL

# --- Callback function to update category in Session State ---
# NOTE: This callback is NO LONGER USED with st.data_editor, but can be kept for reference or future use
def update_category(row_index, selectbox_key):
    """Callback to update the category for a specific row in session state."""
    # This function is kept for context but is not called by the data_editor implementation
    new_category = st.session_state[selectbox_key]
    if 'df_processed' in st.session_state and isinstance(st.session_state.df_processed, pd.DataFrame):
        if row_index in st.session_state.df_processed.index:
            st.session_state.df_processed.loc[row_index, 'Categories'] = new_category

# --- Calculation Functions ---
# (Keep calculate_fortnightly_expenses as provided)
def calculate_fortnightly_expenses(df, excluded_category="School Fees"):
    required_cols = ['Date', 'Debit Amount', 'Categories']
    if df is None or not isinstance(df, pd.DataFrame):
         return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        st.error(f"Calculation Error: Missing columns {', '.join(missing_cols)}")
        return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0
    df_calc = df.copy()
    error_occurred = False
    try:
        df_calc['Date'] = pd.to_datetime(df_calc['Date'], errors='coerce')
        df_calc['Debit Amount'] = pd.to_numeric(df_calc['Debit Amount'], errors='coerce').fillna(0)
        df_calc['Categories'] = df_calc['Categories'].astype(str)
        if df_calc['Date'].isnull().any():
            df_calc = df_calc.dropna(subset=['Date'])
            if df_calc.empty:
                 return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0
    except Exception as e:
        st.error(f"Error preparing columns for calculation: {e}")
        error_occurred = True
    if error_occurred:
         return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0
    df_expenses = df_calc[
        (df_calc['Debit Amount'] > 0) & (df_calc['Categories'] != excluded_category)
    ].copy()
    if df_expenses.empty:
        return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0
    try:
        if not isinstance(df_expenses.index, pd.DatetimeIndex):
            df_expenses.sort_values('Date', inplace=True)
            df_expenses.set_index('Date', inplace=True)
        fortnightly_sum = df_expenses['Debit Amount'].resample('14D', label='right', closed='right').sum()
        valid_fortnights = fortnightly_sum[fortnightly_sum > 0]
        average_expense = valid_fortnights.mean() if not valid_fortnights.empty else 0.0
    except Exception as e:
        st.error(f"Error during resampling or averaging: {e}")
        return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0
    return fortnightly_sum.reset_index(), average_expense

# --- Plotting Functions ---
# (Keep plotting functions as provided)
def plot_expenses_timeseries(fortnightly_data):
    if fortnightly_data is None or not isinstance(fortnightly_data, pd.DataFrame) or fortnightly_data.empty: return None
    if 'Date' not in fortnightly_data.columns or 'Debit Amount' not in fortnightly_data.columns: return None
    try:
        fig = px.line(fortnightly_data, x='Date', y='Debit Amount', title="Fortnightly General Expenses Over Time (Excl. School Fees)", markers=True, labels={'Debit Amount': 'Total Expenses ($)'})
        fig.update_layout(xaxis_title="Fortnight Period Ending", yaxis_title="Total Expenses ($)")
        return fig
    except Exception as e: st.error(f"Error creating timeseries plot: {e}"); return None

def plot_category_pie(df):
    if df is None or not isinstance(df, pd.DataFrame): return None
    if 'Debit Amount' not in df.columns or 'Categories' not in df.columns: return None
    df_plot = df[df['Debit Amount'] > 0].copy()
    df_plot['Debit Amount'] = pd.to_numeric(df_plot['Debit Amount'], errors='coerce').fillna(0)
    df_plot = df_plot[df_plot['Debit Amount'] > 0]
    if df_plot.empty: return None
    try:
        expenses_by_cat = df_plot.groupby('Categories')['Debit Amount'].sum().reset_index()
        expenses_by_cat = expenses_by_cat[expenses_by_cat['Debit Amount'] > 0]
        if expenses_by_cat.empty: return None
        fig = px.pie(expenses_by_cat, names='Categories', values='Debit Amount', title="Expense Distribution by Category", hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label', sort=False)
        return fig
    except Exception as e: st.error(f"Error creating pie chart: {e}"); return None

def plot_category_bar(df):
    if df is None or not isinstance(df, pd.DataFrame): return None
    if 'Debit Amount' not in df.columns or 'Categories' not in df.columns: return None
    df_plot = df[df['Debit Amount'] > 0].copy()
    df_plot['Debit Amount'] = pd.to_numeric(df_plot['Debit Amount'], errors='coerce').fillna(0)
    df_plot = df_plot[df_plot['Debit Amount'] > 0]
    if df_plot.empty: return None
    try:
        expenses_by_cat = df_plot.groupby('Categories')['Debit Amount'].sum().reset_index()
        expenses_by_cat = expenses_by_cat[expenses_by_cat['Debit Amount'] > 0]
        if expenses_by_cat.empty: return None
        expenses_by_cat = expenses_by_cat.sort_values('Debit Amount', ascending=False)
        fig = px.bar(expenses_by_cat, x='Categories', y='Debit Amount', title="Total Expenses per Category", labels={'Debit Amount': 'Total Expenses ($)'})
        fig.update_layout(xaxis_title="Category", yaxis_title="Total Expenses ($)")
        return fig
    except Exception as e: st.error(f"Error creating bar chart: {e}"); return None

# --- Helper Function for Excel Download ---
# (Keep to_excel as provided)
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if isinstance(df, pd.DataFrame):
            df_copy = df.copy()
            for col in ['Debit Amount', 'Credit Amount', 'Balance']:
                 if col in df_copy.columns:
                     try: df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                     except Exception: pass
            df_copy.to_excel(writer, index=False, sheet_name='Categorized Transactions')
        else: pd.DataFrame().to_excel(writer, index=False, sheet_name='No Data')
    processed_data = output.getvalue()
    return processed_data

# --- Initialize Session State ---
# (Keep session state initialization as provided)
if 'df_processed' not in st.session_state: st.session_state.df_processed = None
if 'processed_file_name' not in st.session_state: st.session_state.processed_file_name = None
if 'what_if_adjustments' not in st.session_state: st.session_state.what_if_adjustments = {}

# --- Streamlit App UI ---
st.title("💰 Bank Transaction Analysis Tool")
st.markdown("Upload your bank statement (Excel format) to categorize transactions, visualize spending, and explore 'What-If' scenarios.")

# --- File Upload ---
# (Keep file upload as provided)
uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type="xlsx", key="file_uploader")

# --- Processing Logic ---
# (Keep processing logic as provided)
process_new_file = False
if uploaded_file is not None:
    if uploaded_file.name != st.session_state.get('processed_file_name', None):
        process_new_file = True
        st.info(f"New file detected: '{uploaded_file.name}'. Processing...")

if process_new_file:
    st.session_state.df_processed = None
    st.session_state.processed_file_name = None
    st.session_state.what_if_adjustments = {}
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        required_columns = ['Date', 'Narrative', 'Debit Amount', 'Credit Amount', 'Balance']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            st.error(f"Error: Missing required columns: {', '.join(missing)}.")
        else:
            try:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                initial_rows = len(df)
                df.dropna(subset=['Date'], inplace=True)
                if len(df) < initial_rows: st.warning(f"Dropped {initial_rows - len(df)} rows due to invalid date entries.")
            except Exception as e: st.error(f"Critical error converting 'Date' column: {e}"); df = None
            if df is not None:
                for col in ['Debit Amount', 'Credit Amount', 'Balance']:
                     if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                df_processed = df.copy()
                df_processed['Narrative'] = df_processed['Narrative'].fillna('').astype(str)
                if 'Categories' not in df_processed.columns:
                     df_processed['Categories'] = df_processed['Narrative'].apply(lambda x: categorize_transaction(x, CATEGORY_KEYWORDS))
                else:
                    st.info("Existing 'Categories' column found. Applying auto-categorization to empty or 'Uncategorized' rows.")
                    df_processed['Categories'] = df_processed['Categories'].fillna(UNCATEGORIZED_LABEL).astype(str)
                    mask_needs_categorizing = (df_processed['Categories'].str.strip() == '') | (df_processed['Categories'].str.lower() == 'uncategorized') | (df_processed['Categories'] == UNCATEGORIZED_LABEL)
                    df_processed.loc[mask_needs_categorizing, 'Categories'] = df_processed.loc[mask_needs_categorizing, 'Narrative'].apply(lambda x: categorize_transaction(x, CATEGORY_KEYWORDS))
                st.session_state.df_processed = df_processed
                st.session_state.processed_file_name = uploaded_file.name
                st.success(f"File '{uploaded_file.name}' processed successfully!")
                st.rerun()
    except Exception as e:
        st.error(f"An unexpected error occurred during file processing: {e}")
        st.exception(e)
        st.session_state.df_processed = None
        st.session_state.processed_file_name = None
        st.session_state.what_if_adjustments = {}

# --- Display and Interaction Area ---
if isinstance(st.session_state.get('df_processed'), pd.DataFrame):
    current_df = st.session_state.df_processed

    st.write("---")
    st.header("📊 Transaction Data")
    st.write(f"Displaying data from: **{st.session_state.processed_file_name}**")
    # Display the full dataframe (read-only view here)
    st.dataframe(current_df)

    # --- !!! MODIFIED: Manual Categorization Section using st.data_editor !!! ---
    st.write("---")
    st.header("✏️ Manual Categorization")

    # Filter the DataFrame to get only uncategorized rows
    uncategorized_mask = current_df['Categories'] == UNCATEGORIZED_LABEL
    # Create a DataFrame containing only the columns needed for the editor
    # and only the rows that are uncategorized. IMPORTANT: Use .copy()
    df_to_edit = current_df.loc[uncategorized_mask, ['Date', 'Narrative', 'Debit Amount', 'Categories']].copy()

    if not df_to_edit.empty:
        st.warning(f"Found {len(df_to_edit)} transactions needing manual categorization.")
        st.markdown("**Select the correct category from the dropdown in the 'Categories' column below:**")

        # Define column configurations for the data editor
        column_config = {
            "Date": st.column_config.DateColumn(
                "Date",
                format="YYYY-MM-DD", # Format for display
                disabled=True      # Make column read-only
            ),
            "Narrative": st.column_config.TextColumn(
                "Narrative",
                # width="large", # Optional: Adjust width if needed
                disabled=True   # Make column read-only
            ),
            "Debit Amount": st.column_config.NumberColumn(
                "Amount ($)",          # Display name
                format="$%.2f",        # Format as currency
                disabled=True          # Make column read-only
            ),
            "Categories": st.column_config.SelectboxColumn(
                "Category",               # Display name (Changed from 'Categories' for clarity)
                options=ALL_CATEGORY_OPTIONS, # List of choices
                required=True,             # Must select a value
                # Default value is automatically handled by the editor based on current data
                # disabled=False (default) # This column IS editable
            )
            # Note: We only included columns needed for the editor in df_to_edit
            # so no need to hide 'Credit Amount', 'Balance' etc. here
        }

        # Display the data editor
        edited_df = st.data_editor(
            df_to_edit,
            column_config=column_config,
            hide_index=True,             # Hide the DataFrame index
            use_container_width=True,    # Use full available width
            key="category_editor"        # Unique key for the editor widget
        )

        # --- Update the main DataFrame in session state ---
        # Check if any changes were made in the editor compared to the initial slice
        # Use .equals() for value comparison. Need to ensure indices match or reset index for comparison
        # Resetting index is safer if index isn't guaranteed to be identical
        if not edited_df.reset_index(drop=True).equals(df_to_edit.reset_index(drop=True)):
             st.info("Applying category changes...")
             # Update the 'Categories' column in the main DataFrame (current_df/st.session_state.df_processed)
             # Use the index from edited_df (which matches the original slice) to update the correct rows
             st.session_state.df_processed.loc[edited_df.index, 'Categories'] = edited_df['Categories']
             # Rerun the script to reflect changes immediately in plots, calculations, etc.
             st.rerun()

    else:
        st.success("✅ All transactions are currently categorized!")
    # --- !!! END OF MODIFIED SECTION !!! ---


    # --- Download Button ---
    # (Keep as provided)
    st.download_button(
        label="📥 Download Categorized Data as Excel",
        data=to_excel(current_df), # Pass the potentially updated DataFrame
        file_name=f'categorized_{st.session_state.processed_file_name}', # Include original filename
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        key='download_button'
    )

    # --- Fortnightly Expense Calculation (Based on Current Data) ---
    # (Keep as provided)
    st.write("---")
    st.header("💹 Fortnightly General Expense Analysis (Excl. School Fees)")
    fortnightly_expenses_df, avg_fortnightly_expense = calculate_fortnightly_expenses(current_df)
    if fortnightly_expenses_df is not None and not fortnightly_expenses_df.empty:
        st.metric(label="Current Average Fortnightly General Expense", value=f"${avg_fortnightly_expense:,.2f}")
        with st.expander("View Current Fortnightly General Expense Data"):
            st.dataframe(fortnightly_expenses_df.style.format({"Date": "{:%Y-%m-%d}", "Debit Amount": "${:,.2f}"}))
    else:
        st.info("Fortnightly expense calculation did not yield results (check data or filters).")


    # --- What-If Scenario Analysis Section ---
    # (Keep as provided)
    st.write("---")
    st.header("🔮 What-If Scenario Analysis")
    st.markdown("Adjust spending percentages below to see the potential impact on your average fortnightly expenses (excluding School Fees).")
    try:
        expense_categories = sorted([cat for cat in current_df['Categories'].unique() if cat not in ["Income", "School Fees", UNCATEGORIZED_LABEL] and pd.notna(cat)])
    except Exception as e: st.error(f"Error identifying expense categories: {e}"); expense_categories = []
    if not expense_categories:
        st.info("No adjustable expense categories found in the data (after excluding Income, School Fees, Uncategorized).")
    else:
        st.markdown("**Set Percentage Reduction (%) for Categories:**")
        what_if_adjustments = st.session_state.what_if_adjustments
        cols = st.columns(3)
        col_idx = 0
        for category in expense_categories:
            slider_key = f"what_if_{category}"
            current_value = what_if_adjustments.get(slider_key, 0)
            with cols[col_idx % len(cols)]:
                 what_if_adjustments[slider_key] = st.slider(f"Reduce {category} by:", min_value=0, max_value=100, value=current_value, step=5, key=slider_key, help=f"Set the percentage to reduce spending for '{category}'.")
            col_idx += 1
        df_hypothetical = current_df.copy()
        df_hypothetical['Debit Amount'] = pd.to_numeric(df_hypothetical['Debit Amount'], errors='coerce').fillna(0)
        for category, percentage in what_if_adjustments.items():
             cat_name = category.replace("what_if_", "")
             if percentage > 0 and cat_name in df_hypothetical['Categories'].values:
                 mask = df_hypothetical['Categories'] == cat_name
                 df_hypothetical.loc[mask, 'Debit Amount'] *= (1 - percentage / 100.0)
        _, avg_hypothetical_expense = calculate_fortnightly_expenses(df_hypothetical)
        st.write("---")
        col1_whatif, col2_whatif, col3_whatif = st.columns(3)
        with col1_whatif: st.metric(label="Current Avg Fortnightly", value=f"${avg_fortnightly_expense:,.2f}")
        with col2_whatif: st.metric(label="Hypothetical Avg Fortnightly", value=f"${avg_hypothetical_expense:,.2f}")
        with col3_whatif:
            savings = avg_fortnightly_expense - avg_hypothetical_expense
            st.metric(label="Potential Fortnightly Savings", value=f"${savings:,.2f}", delta=f"{savings:.2f}", delta_color="normal")
        target_savings = 500
        target_fortnightly = avg_fortnightly_expense - target_savings
        if avg_hypothetical_expense <= target_fortnightly: st.success(f"🎉 Goal Achieved! Hypothetical average (${avg_hypothetical_expense:,.2f}) meets or beats the target savings of ${target_savings:,.2f}/fortnight.")
        else: st.warning(f"⚠️ Keep Adjusting: Hypothetical average (${avg_hypothetical_expense:,.2f}) is above the target savings goal (needs to be ~${target_fortnightly:,.2f} or less).")


    # --- Visualizations Section (Based on Current Data) ---
    # (Keep as provided)
    st.write("---")
    st.header("📈 Visualizations (Based on Current Data)")
    st.markdown("These charts reflect the *current* state of your categorized data.")
    current_df_for_plots = current_df
    col_viz1, col_viz2 = st.columns(2)
    with col_viz1:
        st.subheader("Expense Breakdown")
        fig_pie = plot_category_pie(current_df_for_plots)
        if fig_pie: st.plotly_chart(fig_pie, use_container_width=True)
        else: st.info("No expense data available for pie chart.")
        fig_bar = plot_category_bar(current_df_for_plots)
        if fig_bar: st.plotly_chart(fig_bar, use_container_width=True)
        else: st.info("No expense data available for bar chart.")
    with col_viz2:
        st.subheader("Expenses Over Time")
        fig_timeseries = plot_expenses_timeseries(fortnightly_expenses_df)
        if fig_timeseries: st.plotly_chart(fig_timeseries, use_container_width=True)
        else: st.info("No fortnightly expense data available to plot over time.")


# --- Initial State Message ---
# (Keep as provided)
elif not uploaded_file and st.session_state.get('df_processed') is None:
    st.info("Awaiting Excel file upload...")

# --- Sidebar Information ---
# (Update About text slightly to mention editable table)
st.sidebar.title("About")
st.sidebar.info(
    """
    This app helps analyze bank transactions:
    1.  **Upload** your Excel statement.
    2.  Transactions are **auto-categorized**.
    3.  **Manually correct** categories using the editable table below.
    4.  View **fortnightly expense** summaries.
    5.  Use **'What-If' sliders** to explore budget changes.
    6.  Explore interactive **charts**.
    7.  **Download** the categorized data.
    """
)
st.sidebar.title("Categories & Keywords")
# (Keep as provided)
display_keywords = {cat: ', '.join(kw) for cat, kw in CATEGORY_KEYWORDS.items()}
with st.sidebar.expander("View Keywords (Code Level)"):
    st.json(display_keywords) # Use json for nice formatting
st.sidebar.markdown("*(To permanently add/modify keywords, edit the `CATEGORY_KEYWORDS` dictionary in the `app.py` script and restart the app.)*")

# --- END OF FILE app.py ---