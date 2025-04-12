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
    "Groceries": ["haigh", "fresco", "hearthfire", "kombu","woolworths", "iga", "fuller fresh", "e.a. fuller","coles", "dorrigo deli"], # Added coles as common
    "Fuel": ["bp", "caltex", "shell", "ampol", "fullers fuel"], # Added common fuel brands
    "School Fees": ["st hilda's", "school fee", "edstart", "st hildas", "edutest"],
    "Dogs and Chickens": ["lyka", "petbarn", "norco", "vet"],
    "Insurance": ["nrma", "aia", "insurance","nib"],
    "Meals & Dining": ["matilda", "coastal harvest", "maxsum", "burger", "thai", "indian", "black bear", "5 church street", "cafe", "restaurant", "mcdonalds", "kfc"], # Added common ones
    "Pharmacy": ["bellingen pharmacy", "chemist", "pharmacy", "pharm"], # Added common pharmacies
    "Union Fees": ["union fee", "asu", "cpsu"], # Added common unions
    "Rent/Mortgage": ["real estate", "rent", "mortgage"], # Added common housing
    "Utilities": ["energy", "water", "gas", "telstra", "optus", "vodafone", "agl", "origin"], # Added common utilities
    "Subscriptions": ["netflix", "spotify", "stan", "disney","apple", "primevideo"], # Added common subscriptions
    "Shopping": ["kmart", "big w", "target", "amazon", "ebay"], # Added common retailers
    "Transport": ["uber", "didi", "taxi", "opal", "public transport"], # Added transport
    "Health:": ["outpost hair", "doctor", "dentist", "physio", "hospital", "medical", "bellingen healing", "medicare", "mcare benefits"], # Added health
    "Ada": ["bun bun bao", "westpac cho ada", "savin ada", "sweet bellingen", "yy rock", "sens fushion"], # Added Ada
    "Home Maintenance": ["bunnings", "hardware", "home depot", "handyman", "gardening"], # Added home maintenance
    "Books": ["alternatives"],
    "Donations": ["childfund"],# Add more categories and keywords as needed
}

UNCATEGORIZED_LABEL = "Uncategorized"

def categorize_transaction(narrative, keywords_dict):
    """
    Categorizes a transaction based on keywords in the narrative.

    Args:
        narrative (str): The transaction narrative.
        keywords_dict (dict): Dictionary of categories and their keywords.

    Returns:
        str: The determined category or UNCATEGORIZED_LABEL.
    """
    if not isinstance(narrative, str):
        return UNCATEGORIZED_LABEL # Handle non-string narratives

    narrative_lower = narrative.lower()
    for category, keywords in keywords_dict.items():
        for keyword in keywords:
            if keyword in narrative_lower:
                return category
    return UNCATEGORIZED_LABEL

# --- Calculation Functions ---
def calculate_fortnightly_expenses(df):
    """
    Calculates total expenses for each 14-day period, EXCLUDING 'School Fees'.

    Args:
        df (pd.DataFrame): The processed dataframe with 'Date', 'Debit Amount', and 'Categories'.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with fortnightly sums indexed by period end date.
                    Returns an empty DataFrame if no relevant expenses exist.
            - float: Average fortnightly expense (excluding School Fees).
                     Returns 0.0 if no relevant expenses exist.
    """
    required_cols = ['Date', 'Debit Amount', 'Categories']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        st.error(f"Required columns for expense calculation not found: {', '.join(missing_cols)}")
        # Return structure consistent with success case but empty
        return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0

    # Ensure correct data types
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Debit Amount'] = pd.to_numeric(df['Debit Amount'], errors='coerce').fillna(0)
        # Ensure Categories is string type for reliable comparison
        df['Categories'] = df['Categories'].astype(str)
    except Exception as e:
        st.error(f"Error converting columns for calculation: {e}")
        return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0

    # Define the category to exclude
    school_fees_category = "School Fees" # Match exact category name from CATEGORY_KEYWORDS

    # --- MODIFICATION START ---
    # Filter for expenses (Debit > 0) AND exclude the 'School Fees' category
    df_expenses = df[
        (df['Debit Amount'] > 0) & (df['Categories'] != school_fees_category)
    ].copy()
    # --- MODIFICATION END ---

    if df_expenses.empty:
        st.info(f"No expense transactions found after excluding '{school_fees_category}'.")
        return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0 # Return empty DataFrame and 0 average

    # Proceed with resampling and averaging
    df_expenses.sort_values('Date', inplace=True)
    df_expenses.set_index('Date', inplace=True)

    # Resample into 14-day periods, starting from the first transaction date
    fortnightly_sum = df_expenses['Debit Amount'].resample('14D', label='right', closed='right').sum()

    # Calculate average, ignoring periods with zero expense unless they are the only periods
    valid_fortnights = fortnightly_sum[fortnightly_sum > 0]
    average_expense = valid_fortnights.mean() if not valid_fortnights.empty else 0.0

    return fortnightly_sum.reset_index(), average_expense # Return DataFrame for plotting

# --- Plotting Functions ---
def plot_expenses_timeseries(fortnightly_data):
    """Plots fortnightly expenses over time."""
    if fortnightly_data is None or fortnightly_data.empty:
        return None
    fig = px.line(fortnightly_data, x='Date', y='Debit Amount',
                  title="Fortnightly Expenses Over Time",
                  markers=True, labels={'Debit Amount': 'Total Expenses ($)'})
    fig.update_layout(xaxis_title="Fortnight Period Ending", yaxis_title="Total Expenses ($)")
    return fig

def plot_category_pie(df):
    """Plots a pie chart of expense distribution by category."""
    expenses_by_cat = df[df['Debit Amount'] > 0].groupby('Categories')['Debit Amount'].sum().reset_index()
    expenses_by_cat = expenses_by_cat[expenses_by_cat['Debit Amount'] > 0] # Exclude zero expense categories
    if expenses_by_cat.empty:
        return None
    fig = px.pie(expenses_by_cat, names='Categories', values='Debit Amount',
                 title="Expense Distribution by Category", hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_category_bar(df):
    """Plots a bar chart of total expenses per category."""
    expenses_by_cat = df[df['Debit Amount'] > 0].groupby('Categories')['Debit Amount'].sum().reset_index()
    expenses_by_cat = expenses_by_cat.sort_values('Debit Amount', ascending=False)
    expenses_by_cat = expenses_by_cat[expenses_by_cat['Debit Amount'] > 0] # Exclude zero expense categories
    if expenses_by_cat.empty:
        return None
    fig = px.bar(expenses_by_cat, x='Categories', y='Debit Amount',
                 title="Total Expenses per Category", labels={'Debit Amount': 'Total Expenses ($)'})
    fig.update_layout(xaxis_title="Category", yaxis_title="Total Expenses ($)")
    return fig

# --- Helper Function for Excel Download ---
def to_excel(df):
    output = BytesIO()
    # Use ExcelWriter context manager for compatibility
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Categorized Transactions')
    processed_data = output.getvalue()
    return processed_data

# --- Streamlit App UI ---
st.title("💰 Bank Transaction Analysis Tool")
st.markdown("Upload your bank statement (Excel format) to categorize transactions and visualize spending.")

# --- File Upload ---
uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type="xlsx")

if uploaded_file is not None:
    try:
        # Read the excel file
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.success("File Uploaded Successfully!")

        # --- Data Preprocessing and Validation ---
        required_columns = ['Date', 'Narrative', 'Debit Amount', 'Credit Amount', 'Balance']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Error: Missing required columns. Ensure your file has: {', '.join(required_columns)}")
        else:
            st.write("### Original Data Preview (First 5 Rows):")
            st.dataframe(df.head())

            # --- Automatic Categorization ---
            st.write("---")
            st.header("📊 Transaction Categorization")
            df_processed = df.copy()

            # Ensure Narrative is string, fill NaNs if necessary
            df_processed['Narrative'] = df_processed['Narrative'].fillna('').astype(str)

            # Check if 'Categories' column exists, otherwise create it
            if 'Categories' not in df_processed.columns:
                 df_processed['Categories'] = df_processed['Narrative'].apply(lambda x: categorize_transaction(x, CATEGORY_KEYWORDS))
            else:
                # Option: Allow re-categorization or only categorize empty ones
                st.info("Existing 'Categories' column found. Applying automatic categorization to uncategorized rows only.")
                # Ensure existing categories are strings, fill NaNs with our label
                df_processed['Categories'] = df_processed['Categories'].fillna(UNCATEGORIZED_LABEL).astype(str)
                mask_uncategorized = (df_processed['Categories'].str.strip() == '') | (df_processed['Categories'].str.lower() == 'uncategorized')
                df_processed.loc[mask_uncategorized, 'Categories'] = df_processed.loc[mask_uncategorized, 'Narrative'].apply(lambda x: categorize_transaction(x, CATEGORY_KEYWORDS))


            # --- Display Categorized Data ---
            st.write("#### All Transactions (Categorized):")
            st.dataframe(df_processed)

            # --- Flag Uncategorized Transactions ---
            uncategorized_df = df_processed[df_processed['Categories'] == UNCATEGORIZED_LABEL]
            if not uncategorized_df.empty:
                st.warning(f"Found {len(uncategorized_df)} transactions that could not be automatically categorized.")
                with st.expander("⚠️ View Uncategorized Transactions"):
                    st.dataframe(uncategorized_df)
                    st.markdown(f"**Suggestion:** Review these transactions. You can manually update the 'Categories' column in your original Excel file or download the processed data below, edit it, and re-upload. Add keywords for these narratives to the `CATEGORY_KEYWORDS` dictionary in the script for future automation.")
            else:
                st.success("✅ All transactions were automatically categorized!")

            # --- Download Processed Data ---
            st.download_button(
                label="📥 Download Categorized Data as Excel",
                data=to_excel(df_processed),
                file_name='categorized_transactions.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )


            # --- Expense Calculations ---
            st.write("---")
            st.header(" fortnightly Expense Analysis")
            fortnightly_expenses_df, avg_fortnightly_expense = calculate_fortnightly_expenses(df_processed.copy()) # Pass a copy

            if fortnightly_expenses_df is not None:
                st.metric(label="Average Fortnightly Expense", value=f"${avg_fortnightly_expense:,.2f}")
                with st.expander("View Fortnightly Expense Data"):
                    st.dataframe(fortnightly_expenses_df.style.format({"Debit Amount": "${:,.2f}"})) # Format currency
            else:
                st.warning("Could not calculate fortnightly expenses.")


            # --- Visualizations ---
            st.write("---")
            st.header("📈 Visualizations")

            # Create columns for side-by-side plots
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Expense Breakdown")
                fig_pie = plot_category_pie(df_processed)
                if fig_pie:
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No expense data to display in pie chart.")

                fig_bar = plot_category_bar(df_processed)
                if fig_bar:
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                     st.info("No expense data to display in bar chart.")


            with col2:
                st.subheader("Expenses Over Time")
                fig_timeseries = plot_expenses_timeseries(fortnightly_expenses_df)
                if fig_timeseries:
                    st.plotly_chart(fig_timeseries, use_container_width=True)
                else:
                    st.info("No fortnightly expense data to plot over time.")

    except FileNotFoundError:
         st.error("Error: Could not find the uploaded file.")
    except ValueError as ve:
         st.error(f"Error processing data: {ve}. Please check data types in your columns (especially Date and amounts).")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.exception(e) # Provides more detailed traceback for debugging
else:
    st.info("Awaiting Excel file upload...")

# Add some instructions or notes at the bottom
st.sidebar.title("About")
st.sidebar.info(
    """
    This app helps you analyze bank transactions from an Excel file.
    1.  **Upload** your Excel file (must contain 'Date', 'Narrative', 'Debit Amount' columns).
    2.  Transactions are **automatically categorized** based on keywords in the 'Narrative'.
    3.  **Uncategorized** transactions are flagged for review.
    4.  View **fortnightly expense** summaries and averages.
    5.  Explore interactive **charts** showing spending patterns.
    """
)
st.sidebar.title("Categories & Keywords")
st.sidebar.expander("View/Edit Keywords (Code Level)").json(CATEGORY_KEYWORDS)
st.sidebar.markdown("*(To permanently add keywords, edit the `CATEGORY_KEYWORDS` dictionary in the `app.py` script)*")