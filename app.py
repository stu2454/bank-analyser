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
    "Health": ["outpost hair", "doctor", "dentist", "physio", "hospital", "medical", "bellingen healing", "medicare", "mcare benefits"], # Corrected key
    "Ada": ["bun bun bao", "westpac cho ada", "savin ada", "sweet bellingen", "yy rock", "sens fushion"],
    "Home Maintenance": ["bunnings", "hardware", "home depot", "handyman", "gardening"],
    "Books": ["alternatives"],
    "Donations": ["childfund"],
    "Misc": ["misc"],
    # Add more categories and keywords as needed (use lowercase)
}
# Ensure no legacy typo keys exist
if "Health:" in CATEGORY_KEYWORDS:
     CATEGORY_KEYWORDS["Health"] = CATEGORY_KEYWORDS.pop("Health:")


UNCATEGORIZED_LABEL = "Uncategorized"
# Create the list of category options for the dropdown dynamically
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
def update_category(row_index, selectbox_key):
    """Callback to update the category for a specific row in session state."""
    new_category = st.session_state[selectbox_key]
    # Check if df_processed exists and is a DataFrame before modifying
    if 'df_processed' in st.session_state and isinstance(st.session_state.df_processed, pd.DataFrame):
        if row_index in st.session_state.df_processed.index:
            st.session_state.df_processed.loc[row_index, 'Categories'] = new_category
    # Streamlit automatically reruns on widget change

# --- Calculation Functions ---
def calculate_fortnightly_expenses(df, excluded_category="School Fees"):
    """
    Calculates total expenses for each 14-day period, excluding a specified category.
    Handles potential errors and missing columns gracefully.

    Args:
        df (pd.DataFrame): The dataframe with transaction data. Must contain 'Date', 'Debit Amount', 'Categories'.
        excluded_category (str): The category name to exclude from expense calculation.

    Returns:
        tuple: (pd.DataFrame with fortnightly sums, float average fortnightly expense)
               Returns (empty DataFrame, 0.0) on error or if no relevant expenses.
    """
    required_cols = ['Date', 'Debit Amount', 'Categories']
    if df is None or not isinstance(df, pd.DataFrame):
         # Return default empty state if df is None or not a DataFrame
         return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0

    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        # Display error only if columns are missing, not if df was None initially
        st.error(f"Calculation Error: Missing columns {', '.join(missing_cols)}")
        return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0

    df_calc = df.copy()
    error_occurred = False
    try:
        # Convert Date, coercing errors to NaT (Not a Time)
        df_calc['Date'] = pd.to_datetime(df_calc['Date'], errors='coerce')
        # Convert Debit Amount, coercing errors to NaN, then filling with 0
        df_calc['Debit Amount'] = pd.to_numeric(df_calc['Debit Amount'], errors='coerce').fillna(0)
        # Ensure Categories is string type for reliable comparison
        df_calc['Categories'] = df_calc['Categories'].astype(str)

        # Check for and handle NaT dates after conversion
        if df_calc['Date'].isnull().any():
            # Optionally notify the user
            # st.warning("Some dates could not be parsed and were ignored in calculations.")
            df_calc = df_calc.dropna(subset=['Date'])
            if df_calc.empty:
                 return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0

    except Exception as e:
        st.error(f"Error preparing columns for calculation: {e}")
        error_occurred = True # Flag error

    if error_occurred:
         return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0

    # --- Filter based on criteria ---
    df_expenses = df_calc[
        (df_calc['Debit Amount'] > 0) & (df_calc['Categories'] != excluded_category)
    ].copy()

    if df_expenses.empty:
        # Use st.info for expected cases like no expenses found
        # st.info(f"No expense transactions found (excluding '{excluded_category}') for fortnightly calculation.")
        return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0 # Return empty DataFrame and 0 average

    # --- Resampling and Averaging ---
    try:
        # Ensure Date is the index before resampling
        if not isinstance(df_expenses.index, pd.DatetimeIndex):
            df_expenses.sort_values('Date', inplace=True)
            df_expenses.set_index('Date', inplace=True)

        # Resample into 14-day periods, sum expenses in each period
        # label='right', closed='right' means the label is the end date of the period
        fortnightly_sum = df_expenses['Debit Amount'].resample('14D', label='right', closed='right').sum()

        # Calculate average, ignoring periods with zero expense unless they are the only periods
        valid_fortnights = fortnightly_sum[fortnightly_sum > 0]
        average_expense = valid_fortnights.mean() if not valid_fortnights.empty else 0.0

    except Exception as e:
        st.error(f"Error during resampling or averaging: {e}")
        return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0 # Return empty on error

    return fortnightly_sum.reset_index(), average_expense # Return DataFrame for plotting and the average

# --- Plotting Functions ---
def plot_expenses_timeseries(fortnightly_data):
    """Plots fortnightly expenses over time."""
    if fortnightly_data is None or not isinstance(fortnightly_data, pd.DataFrame) or fortnightly_data.empty:
        return None
    if 'Date' not in fortnightly_data.columns or 'Debit Amount' not in fortnightly_data.columns:
        st.warning("Timeseries plot error: Missing 'Date' or 'Debit Amount' in fortnightly data.")
        return None
    try:
        fig = px.line(fortnightly_data, x='Date', y='Debit Amount',
                      title="Fortnightly General Expenses Over Time (Excl. School Fees)",
                      markers=True, labels={'Debit Amount': 'Total Expenses ($)'})
        fig.update_layout(xaxis_title="Fortnight Period Ending", yaxis_title="Total Expenses ($)")
        return fig
    except Exception as e:
        st.error(f"Error creating timeseries plot: {e}")
        return None

def plot_category_pie(df):
    """Plots a pie chart of expense distribution by category."""
    if df is None or not isinstance(df, pd.DataFrame): return None
    if 'Debit Amount' not in df.columns or 'Categories' not in df.columns:
        # st.info("Pie chart: Required columns ('Debit Amount', 'Categories') not found.")
        return None

    # Make sure calculations happen on a copy and handle non-numeric Debit Amounts
    df_plot = df[df['Debit Amount'] > 0].copy()
    df_plot['Debit Amount'] = pd.to_numeric(df_plot['Debit Amount'], errors='coerce').fillna(0)
    df_plot = df_plot[df_plot['Debit Amount'] > 0] # Filter again after coercion

    if df_plot.empty: return None

    try:
        expenses_by_cat = df_plot.groupby('Categories')['Debit Amount'].sum().reset_index()
        # Exclude categories with zero sum after grouping (shouldn't happen with above filter, but safe)
        expenses_by_cat = expenses_by_cat[expenses_by_cat['Debit Amount'] > 0]
        if expenses_by_cat.empty: return None

        fig = px.pie(expenses_by_cat, names='Categories', values='Debit Amount',
                     title="Expense Distribution by Category", hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label', sort=False) # Prevent auto-sorting if needed
        return fig
    except Exception as e:
        st.error(f"Error creating pie chart: {e}")
        return None


def plot_category_bar(df):
    """Plots a bar chart of total expenses per category."""
    if df is None or not isinstance(df, pd.DataFrame): return None
    if 'Debit Amount' not in df.columns or 'Categories' not in df.columns:
        # st.info("Bar chart: Required columns ('Debit Amount', 'Categories') not found.")
        return None

    # Make sure calculations happen on a copy and handle non-numeric Debit Amounts
    df_plot = df[df['Debit Amount'] > 0].copy()
    df_plot['Debit Amount'] = pd.to_numeric(df_plot['Debit Amount'], errors='coerce').fillna(0)
    df_plot = df_plot[df_plot['Debit Amount'] > 0] # Filter again after coercion

    if df_plot.empty: return None

    try:
        expenses_by_cat = df_plot.groupby('Categories')['Debit Amount'].sum().reset_index()
        expenses_by_cat = expenses_by_cat[expenses_by_cat['Debit Amount'] > 0]
        if expenses_by_cat.empty: return None

        expenses_by_cat = expenses_by_cat.sort_values('Debit Amount', ascending=False)
        fig = px.bar(expenses_by_cat, x='Categories', y='Debit Amount',
                     title="Total Expenses per Category", labels={'Debit Amount': 'Total Expenses ($)'})
        fig.update_layout(xaxis_title="Category", yaxis_title="Total Expenses ($)")
        return fig
    except Exception as e:
        st.error(f"Error creating bar chart: {e}")
        return None

# --- Helper Function for Excel Download ---
def to_excel(df):
    """Converts DataFrame to Excel format in memory (BytesIO)."""
    output = BytesIO()
    # Use ExcelWriter context manager for better handling
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if isinstance(df, pd.DataFrame):
            # Create a copy to avoid modifying the original DataFrame in session state
            df_copy = df.copy()
            # Attempt to convert potential numeric columns stored as objects back to numeric for Excel
            for col in ['Debit Amount', 'Credit Amount', 'Balance']:
                 if col in df_copy.columns:
                     try:
                        # Use errors='ignore' if conversion isn't critical and might fail
                        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                     except Exception:
                         pass # Keep original data type if robust conversion fails
            df_copy.to_excel(writer, index=False, sheet_name='Categorized Transactions')
        else:
            # Create an empty DataFrame if input is not valid
            pd.DataFrame().to_excel(writer, index=False, sheet_name='No Data')
    processed_data = output.getvalue()
    return processed_data

# --- Initialize Session State ---
# Use keys consistently for session state variables
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None # Holds the main DataFrame after initial processing/categorization
if 'processed_file_name' not in st.session_state:
    st.session_state.processed_file_name = None # Tracks the name of the successfully processed file
if 'what_if_adjustments' not in st.session_state:
    st.session_state.what_if_adjustments = {} # Stores the percentage reduction values from sliders

# --- Streamlit App UI ---
st.title("💰 Bank Transaction Analysis Tool")
st.markdown("Upload your bank statement (Excel format) to categorize transactions, visualize spending, and explore 'What-If' scenarios.")

# --- File Upload ---
# Simple key for the uploader is usually sufficient with session state logic
uploaded_file = st.file_uploader(
    "Choose an Excel file (.xlsx)",
    type="xlsx",
    key="file_uploader"
)

# --- Processing Logic ---
# Determine if a new file needs processing
process_new_file = False
if uploaded_file is not None:
    # Check if the file name is different from the one already processed
    if uploaded_file.name != st.session_state.get('processed_file_name', None):
        process_new_file = True
        st.info(f"New file detected: '{uploaded_file.name}'. Processing...") # Provide feedback


if process_new_file:
    # Reset state variables before processing the new file
    st.session_state.df_processed = None
    st.session_state.processed_file_name = None
    st.session_state.what_if_adjustments = {} # Clear previous adjustments

    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')

        # --- Data Validation ---
        required_columns = ['Date', 'Narrative', 'Debit Amount', 'Credit Amount', 'Balance']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            st.error(f"Error: Missing required columns: {', '.join(missing)}. Please ensure your Excel file has these columns.")
            # Keep state reset, do not proceed
        else:
            # --- Basic Data Cleaning and Type Conversion ---
            try:
                # Convert Date, coerce errors, drop rows where date is invalid (NaT)
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                initial_rows = len(df)
                df.dropna(subset=['Date'], inplace=True)
                if len(df) < initial_rows:
                     st.warning(f"Dropped {initial_rows - len(df)} rows due to invalid date entries.")
            except Exception as e:
                st.error(f"Critical error converting 'Date' column: {e}")
                # Stop processing if date conversion fails fundamentally
                df = None # Prevent further processing

            if df is not None: # Proceed only if Date conversion was successful
                # Convert numeric columns, coerce errors to NaN, fill with 0
                for col in ['Debit Amount', 'Credit Amount', 'Balance']:
                     if col in df.columns:
                         df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

                # --- Automatic Categorization ---
                df_processed = df.copy()
                # Ensure Narrative is string, fill NaNs if necessary
                df_processed['Narrative'] = df_processed['Narrative'].fillna('').astype(str)

                # Apply categorization logic
                if 'Categories' not in df_processed.columns:
                     df_processed['Categories'] = df_processed['Narrative'].apply(lambda x: categorize_transaction(x, CATEGORY_KEYWORDS))
                else:
                    # Handle existing 'Categories' column: fill NaNs, ensure string, recategorize empty/'Uncategorized'
                    st.info("Existing 'Categories' column found. Applying auto-categorization to empty or 'Uncategorized' rows.")
                    df_processed['Categories'] = df_processed['Categories'].fillna(UNCATEGORIZED_LABEL).astype(str)
                    mask_needs_categorizing = (df_processed['Categories'].str.strip() == '') | \
                                              (df_processed['Categories'].str.lower() == 'uncategorized') | \
                                              (df_processed['Categories'] == UNCATEGORIZED_LABEL) # Explicit check
                    df_processed.loc[mask_needs_categorizing, 'Categories'] = df_processed.loc[mask_needs_categorizing, 'Narrative'].apply(lambda x: categorize_transaction(x, CATEGORY_KEYWORDS))

                # --- Store results in Session State ---
                st.session_state.df_processed = df_processed
                st.session_state.processed_file_name = uploaded_file.name
                st.success(f"File '{uploaded_file.name}' processed successfully!")
                # Rerun is essential after successful processing to update the entire display based on the new state
                st.rerun()

    except Exception as e:
        st.error(f"An unexpected error occurred during file processing: {e}")
        st.exception(e) # Provides traceback for debugging
        # Ensure state is reset on any processing error
        st.session_state.df_processed = None
        st.session_state.processed_file_name = None
        st.session_state.what_if_adjustments = {}


# --- Display and Interaction Area ---
# This block runs on every script rerun IF df_processed is available in session state
if isinstance(st.session_state.get('df_processed'), pd.DataFrame):
    # Make a local reference for easier access within this block
    current_df = st.session_state.df_processed

    st.write("---")
    st.header("📊 Transaction Data")
    st.write(f"Displaying data from: **{st.session_state.processed_file_name}**")
    # Display the current state of the DataFrame (could have manual updates)
    st.dataframe(current_df)

    # --- Manual Categorization Section ---
    st.write("---")
    st.header("✏️ Manual Categorization")

    # Find uncategorized rows directly from the session state DataFrame
    uncategorized_mask = current_df['Categories'] == UNCATEGORIZED_LABEL
    uncategorized_count = uncategorized_mask.sum()

    if uncategorized_count > 0:
        st.warning(f"Found {uncategorized_count} transactions needing manual categorization.")
        st.markdown("**Assign a category using the dropdowns below:**")

        # Prepare columns for layout outside the loop
        col_date, col_narrative, col_amount, col_category = st.columns([1, 4, 1, 2])

        # Header row for this section
        with col_date: st.write("**Date**")
        with col_narrative: st.write("**Narrative**")
        with col_amount: st.write("**Amount ($)**")
        with col_category: st.write("**Select Category**")
        st.divider() # Visual separator after header

        # Iterate through the *indices* of the uncategorized rows
        for idx in current_df[uncategorized_mask].index:
            row = current_df.loc[idx] # Get the row data for this index
            # Generate a unique key for the selectbox widget for this specific row
            selectbox_key = f"category_select_{idx}"

            # Display row data and selectbox in columns
            with col_date:
                date_val = row.get('Date')
                # Check if Date is a valid timestamp before formatting
                if pd.notnull(date_val) and isinstance(date_val, pd.Timestamp):
                    st.write(date_val.strftime('%Y-%m-%d'))
                else:
                    st.write(str(date_val)) # Display as string if not expected format
            with col_narrative:
                st.write(row.get('Narrative', 'N/A')) # Use .get for safety
            with col_amount:
                amount = row.get('Debit Amount', 0)
                # Format as currency, handle potential NaN/None
                st.write(f"{amount:,.2f}" if pd.notnull(amount) else "N/A")
            with col_category:
                # Find the index of the current category in the options list
                current_category = row.get('Categories', UNCATEGORIZED_LABEL)
                try:
                    default_index = ALL_CATEGORY_OPTIONS.index(current_category)
                except ValueError:
                    default_index = ALL_CATEGORY_OPTIONS.index(UNCATEGORIZED_LABEL) # Fallback

                st.selectbox(
                    label=f"Category for row {idx}", # Hidden label, but unique
                    options=ALL_CATEGORY_OPTIONS,
                    index=default_index,
                    key=selectbox_key, # Crucial: Unique key for state management
                    label_visibility="collapsed", # Hide the label text
                    on_change=update_category, # Callback function
                    args=(idx, selectbox_key) # Pass index and key to callback
                )
            # Optional: Add a divider between editable rows if needed
            # st.divider()

    else:
        st.success("✅ All transactions are currently categorized!")

    # --- Download Button ---
    st.download_button(
        label="📥 Download Categorized Data as Excel",
        data=to_excel(current_df), # Pass the potentially updated DataFrame
        file_name=f'categorized_{st.session_state.processed_file_name}', # Include original filename
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        key='download_button'
    )

    # --- Fortnightly Expense Calculation (Based on Current Data) ---
    st.write("---")
    st.header("💹 Fortnightly General Expense Analysis (Excl. School Fees)")
    # Calculate based on the *current* state of the DataFrame in session state
    fortnightly_expenses_df, avg_fortnightly_expense = calculate_fortnightly_expenses(current_df)

    # Display the calculated average (if successful)
    if fortnightly_expenses_df is not None and not fortnightly_expenses_df.empty:
        st.metric(label="Current Average Fortnightly General Expense", value=f"${avg_fortnightly_expense:,.2f}")
        # Expander to show the detailed fortnightly data
        with st.expander("View Current Fortnightly General Expense Data"):
            st.dataframe(fortnightly_expenses_df.style.format({"Date": "{:%Y-%m-%d}", "Debit Amount": "${:,.2f}"}))
    else:
        # Info/error messages are handled within calculate_fortnightly_expenses if needed
        st.info("Fortnightly expense calculation did not yield results (check data or filters).")


    # --- What-If Scenario Analysis Section ---
    st.write("---")
    st.header("🔮 What-If Scenario Analysis")
    st.markdown("Adjust spending percentages below to see the potential impact on your average fortnightly expenses (excluding School Fees).")

    # Identify relevant expense categories from the current data
    # Exclude Income, the specified excluded category (School Fees), and Uncategorized
    try:
        expense_categories = sorted([
            cat for cat in current_df['Categories'].unique()
            if cat not in ["Income", "School Fees", UNCATEGORIZED_LABEL] and pd.notna(cat)
        ])
    except Exception as e:
        st.error(f"Error identifying expense categories: {e}")
        expense_categories = []


    if not expense_categories:
        st.info("No adjustable expense categories found in the data (after excluding Income, School Fees, Uncategorized).")
    else:
        st.markdown("**Set Percentage Reduction (%) for Categories:**")

        # Use session state dictionary to store slider values
        what_if_adjustments = st.session_state.what_if_adjustments

        # Use columns for better layout of sliders
        cols = st.columns(3) # Adjust number based on preference
        col_idx = 0
        for category in expense_categories:
            slider_key = f"what_if_{category}"
            # Get current value from session state, default to 0 if not set
            current_value = what_if_adjustments.get(slider_key, 0)

            # Place slider in the next available column
            with cols[col_idx % len(cols)]:
                 # Update the session state directly via the slider's value
                 what_if_adjustments[slider_key] = st.slider(
                     f"Reduce {category} by:",
                     min_value=0,
                     max_value=100,
                     value=current_value,
                     step=5,
                     key=slider_key, # Unique key for the slider widget
                     help=f"Set the percentage to reduce spending for '{category}'."
                 )
            col_idx += 1

        # --- Apply Adjustments and Recalculate ---
        # IMPORTANT: Always work on a copy for hypothetical calculations
        df_hypothetical = current_df.copy()

        # Ensure Debit Amount is numeric before modification
        df_hypothetical['Debit Amount'] = pd.to_numeric(df_hypothetical['Debit Amount'], errors='coerce').fillna(0)

        # Apply the reductions based on slider values (from session state)
        for category, percentage in what_if_adjustments.items():
             # Extract category name from the key 'what_if_CategoryName'
             cat_name = category.replace("what_if_", "")
             if percentage > 0 and cat_name in df_hypothetical['Categories'].values:
                 mask = df_hypothetical['Categories'] == cat_name
                 # Apply reduction factor
                 df_hypothetical.loc[mask, 'Debit Amount'] *= (1 - percentage / 100.0)

        # Recalculate the average using the modified (hypothetical) DataFrame
        _, avg_hypothetical_expense = calculate_fortnightly_expenses(df_hypothetical)

        # --- Display What-If Results ---
        st.write("---") # Separator
        col1_whatif, col2_whatif, col3_whatif = st.columns(3)
        with col1_whatif:
            st.metric(
                label="Current Avg Fortnightly",
                value=f"${avg_fortnightly_expense:,.2f}" # Original average
            )
        with col2_whatif:
             st.metric(
                label="Hypothetical Avg Fortnightly",
                value=f"${avg_hypothetical_expense:,.2f}" # Average after reductions
            )
        with col3_whatif:
            savings = avg_fortnightly_expense - avg_hypothetical_expense
            # Delta shows the difference; color indicates direction (green=savings)
            st.metric(
                label="Potential Fortnightly Savings",
                value=f"${savings:,.2f}",
                delta=f"{savings:.2f}", # Use delta to show the change amount
                delta_color="normal" # "normal" shows green for positive delta (savings)
            )

        # Optional: Add target feedback
        target_savings = 500
        target_fortnightly = avg_fortnightly_expense - target_savings
        if avg_hypothetical_expense <= target_fortnightly:
             st.success(f"🎉 Goal Achieved! Hypothetical average (${avg_hypothetical_expense:,.2f}) meets or beats the target savings of ${target_savings:,.2f}/fortnight.")
        else:
             st.warning(f"⚠️ Keep Adjusting: Hypothetical average (${avg_hypothetical_expense:,.2f}) is above the target savings goal (needs to be ~${target_fortnightly:,.2f} or less).")


    # --- Visualizations Section (Based on Current Data) ---
    st.write("---")
    st.header("📈 Visualizations (Based on Current Data)")
    st.markdown("These charts reflect the *current* state of your categorized data.")

    # Use the actual current DataFrame for plots, not the hypothetical one
    current_df_for_plots = current_df

    # Use columns for plot layout
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
        # Use the original fortnightly calculation results for the timeseries plot
        fig_timeseries = plot_expenses_timeseries(fortnightly_expenses_df)
        if fig_timeseries: st.plotly_chart(fig_timeseries, use_container_width=True)
        else: st.info("No fortnightly expense data available to plot over time.")


# --- Initial State Message ---
# Show message only if no file has been uploaded *at all* in this session run
elif not uploaded_file and st.session_state.get('df_processed') is None:
    st.info("Awaiting Excel file upload...")
# If uploaded_file exists but df_processed is None, it implies an error occurred during processing,
# and the error message should already be displayed above.

# --- Sidebar Information ---
st.sidebar.title("About")
st.sidebar.info(
    """
    This app helps analyze bank transactions:
    1.  **Upload** your Excel statement.
    2.  Transactions are **auto-categorized**.
    3.  **Manually correct** categories if needed.
    4.  View **fortnightly expense** summaries.
    5.  Use **'What-If' sliders** to explore budget changes.
    6.  Explore interactive **charts**.
    7.  **Download** the categorized data.
    """
)
st.sidebar.title("Categories & Keywords")
# Format keywords for display (readability)
display_keywords = {cat: ', '.join(kw) for cat, kw in CATEGORY_KEYWORDS.items()}
with st.sidebar.expander("View Keywords (Code Level)"):
    st.json(display_keywords) # Use json for nice formatting
st.sidebar.markdown("*(To permanently add/modify keywords, edit the `CATEGORY_KEYWORDS` dictionary in the `app.py` script and restart the app.)*")

# --- END OF FILE app.py ---