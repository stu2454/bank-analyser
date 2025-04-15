# --- START OF FILE app.py ---

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import plotly.graph_objects as go

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Bank Transaction Analyzer")

# --- Categorization Logic ---
# (Keep CATEGORY_KEYWORDS, UNCATEGORIZED_LABEL, ALL_CATEGORY_OPTIONS, categorize_transaction as provided)
CATEGORY_KEYWORDS = {
    "Income": ["ndia", "iag", "medicare benefit", "iag dividend", "heather smith christmas shopping"],
    "Groceries": ["peaches patisser", "prime qual", "peaches", "haigh", "smp*fresco market", "hearthfire", "kombu", "woolworths", "iga", "fuller fresh", "e.a. fuller", "coles", "dorrigo deli"],
    "Fuel/Car": ["bp ", "caltex", "shell", "ampol", "fullers fuel", "reddy express", "ballina", "matade"],
    "School Fees": ["edstart"],
    "School other": ["school other","mspgoldcoastnor", "sports", "st hilda's", "school fee", "st hildas", "edutest"],
    "Clothing": ["just group", "kmart", "myer", "blue illusion", "lorna jane", "bras", "slipsilk", "vivid", "clothing", "clothes", "fashion", "shoes", "apparel", "adore beauty"],
    "Dogs and Chickens": ["lyka", "petbarn", "norco", "vet"],
    "Insurance": ["nrma", "aia", "insurance", "nib", "service nsw sydney"],
    "Meals & Dining": ["boost coffs", "eats help.ub er.com", "gelato", "eric and deb", "hyde", "oporto", "sens fusion", "fushion", "bun bun bao", "eats", "matilda's pantry", "coastal harvest", "maxsum", "burger", "thai", "indian", "black bear", "5 church street", "cafe", "restaurant", "fiume", "bellingen kitchen", "brooki bakehouse"],
    "Pharmacy": ["bellingen pharmacy", "chemist", "pharmacy", "pharm"],
    "Union Fees": ["union fee", "asu", "cpsu"],
    "Rates": ["bellingen rates", "bellingen feb rates", "bpay bellingen rates", "rates"],
    "Utilities": ["energy", "water", "gas", "telstra", "optus", "vodafone", "agl", "origin", "energyaust", "bpay energyaust"],
    "Subscriptions": ["stan.com.au", "netflix", "spotify", "stan.com.au", "disney", "apple.com/bill", "primevideo", "new york times", "chatgpt", "openai"],
    "Health/Personal Care": ["mecca", "bradley", "outpost hair", "doctor", "dentist", "physio", "hospital", "medical", "bellingen healing", "medicare", "mcare benefits"],
    "Ada": ["westpac cho ada", "savin ada", "sweet bellingen", "yy rock"],
    "Home Maintenance": ["outdoor", "cleaner", "bunnings", "hardware", "handyman", "gardening", "officeworks"],
    "Books": ["alternatives boo ksbellingen", "book", "books", "bookstore", "library", "box of books"],
    "Donations": ["childfund"],
    "Lotteries": ["lotto", "lottery", "lotteries"],
    "Misc": ["misc"],
    "Entertainment": ["birch carroll", "cinema", "evt-cinema"],
    "Furniture/Homewares": ["pillow talk", "spotlight"]
}
if "Health:" in CATEGORY_KEYWORDS: CATEGORY_KEYWORDS["Health/Personal Care"] = CATEGORY_KEYWORDS.pop("Health:") # Fix potential key mismatch if needed
UNCATEGORIZED_LABEL = "Uncategorized"
ALL_CATEGORY_OPTIONS = [UNCATEGORIZED_LABEL] + sorted(list(CATEGORY_KEYWORDS.keys()))

# Add this function to normalize keywords
def normalize_keywords(keywords_dict):
    """Convert all keywords to lowercase to ensure case-insensitive matching."""
    normalized = {}
    for category, keywords in keywords_dict.items():
        normalized[category] = [kw.lower() for kw in keywords]
    return normalized

# Normalize the keywords dictionary
CATEGORY_KEYWORDS = normalize_keywords(CATEGORY_KEYWORDS)

def categorize_transaction(narrative, keywords_dict):
    if not isinstance(narrative, str): return UNCATEGORIZED_LABEL
    narrative_lower = narrative.lower()
    
    # Clean the narrative by removing special characters and extra spaces
    cleaned_narrative = ' '.join(narrative_lower.split())
    cleaned_narrative = cleaned_narrative.replace('\\', ' ').replace('/', ' ').replace('*', ' ')
    
    # Remove common location words that might appear after the business name
    location_words = ['sydney', 'australia', 'aus', 'nsw', 'coffs harbour', 'coffs', 'harbour', 'bellingen']
    cleaned_narrative = ' '.join(word for word in cleaned_narrative.split() if word not in location_words)
    
    # For URLs, also try matching without the domain extension
    url_cleaned = cleaned_narrative.replace('.com', '').replace('.au', '').replace('.net', '')
    
    # Split the narrative into words for more precise matching
    narrative_words = set(cleaned_narrative.split())
    url_words = set(url_cleaned.split())
    
    # First try exact word matches
    for category, keywords in keywords_dict.items():
        for keyword in keywords:
            # Clean the keyword in the same way
            clean_keyword = keyword.lower().replace('*', ' ').replace('/', ' ').replace('\\', ' ')
            clean_keyword = ' '.join(word for word in clean_keyword.split() if word not in location_words)
            
            # For single-word keywords, require exact word match
            if ' ' not in clean_keyword:
                # Try both original and URL-cleaned versions
                if clean_keyword in narrative_words or clean_keyword in url_words:
                    return category
            # For multi-word keywords, allow partial match
            elif ' ' in clean_keyword:
                # Try both original and URL-cleaned versions
                if clean_keyword in cleaned_narrative or clean_keyword in url_cleaned:
                    return category
    
    return UNCATEGORIZED_LABEL

# Add this function to help debug categorization
def debug_categorization(narrative, keywords_dict):
    if not isinstance(narrative, str): return "Not a string"
    narrative_lower = narrative.lower()
    matches = []
    
    for category, keywords in keywords_dict.items():
        for keyword in keywords:
            if keyword.lower() in narrative_lower:
                matches.append(f"Matched '{keyword}' in category '{category}'")
    
    return "\n".join(matches) if matches else "No matches found"

# Add this debug function
def analyze_transaction(narrative, keywords_dict):
    """Analyze a transaction to see which keywords match and why."""
    if not isinstance(narrative, str): return "Not a string"
    narrative_lower = narrative.lower()
    analysis = []
    
    # Check each category
    for category, keywords in keywords_dict.items():
        matches = []
        for keyword in keywords:
            if keyword in narrative_lower:
                matches.append(keyword)
        if matches:
            analysis.append(f"Category '{category}' matches: {', '.join(matches)}")
    
    return "\n".join(analysis) if analysis else "No matches found"

# --- Calculation Functions ---
def calculate_fortnightly_expenses(df, excluded_category="School Fees"):
    """Calculates total expenses (Debit Amount > 0) fortnightly, excluding a category."""
    required_cols = ['Date', 'Debit Amount', 'Categories']
    if df is None or not isinstance(df, pd.DataFrame): return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        st.error(f"Expense Calc Error: Missing columns {', '.join(missing_cols)}")
        return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0

    df_calc = df.copy()
    try:
        df_calc['Date'] = pd.to_datetime(df_calc['Date'], errors='coerce')
        df_calc['Debit Amount'] = pd.to_numeric(df_calc['Debit Amount'], errors='coerce').fillna(0)
        df_calc['Categories'] = df_calc['Categories'].astype(str)
        df_calc.dropna(subset=['Date'], inplace=True)
        if df_calc.empty: return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0
    except Exception as e:
        st.error(f"Error preparing expense columns: {e}")
        return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0

    df_expenses = df_calc[
        (df_calc['Debit Amount'] > 0) & (df_calc['Categories'] != excluded_category)
    ].copy()
    if df_expenses.empty: return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0

    try:
        if not isinstance(df_expenses.index, pd.DatetimeIndex):
            df_expenses.sort_values('Date', inplace=True)
            df_expenses.set_index('Date', inplace=True)
        # Consistent resampling parameters
        fortnightly_sum = df_expenses['Debit Amount'].resample('14D', label='right', closed='right').sum()
        valid_fortnights = fortnightly_sum[fortnightly_sum > 0]
        average_expense = valid_fortnights.mean() if not valid_fortnights.empty else 0.0
    except Exception as e:
        st.error(f"Error during expense resampling: {e}")
        return pd.DataFrame({'Date': [], 'Debit Amount': []}), 0.0
    return fortnightly_sum.reset_index(), average_expense

# --- NEW FUNCTION: Calculate Fortnightly Income ---
def calculate_fortnightly_income(df):
    """Calculates total income (Credit Amount > 0) fortnightly."""
    required_cols = ['Date', 'Credit Amount']
    if df is None or not isinstance(df, pd.DataFrame): return pd.DataFrame({'Date': [], 'Credit Amount': []})
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        st.error(f"Income Calc Error: Missing columns {', '.join(missing_cols)}")
        return pd.DataFrame({'Date': [], 'Credit Amount': []})

    df_calc = df.copy()
    try:
        df_calc['Date'] = pd.to_datetime(df_calc['Date'], errors='coerce')
        df_calc['Credit Amount'] = pd.to_numeric(df_calc['Credit Amount'], errors='coerce').fillna(0)
        df_calc.dropna(subset=['Date'], inplace=True)
        if df_calc.empty: return pd.DataFrame({'Date': [], 'Credit Amount': []})
    except Exception as e:
        st.error(f"Error preparing income columns: {e}")
        return pd.DataFrame({'Date': [], 'Credit Amount': []})

    df_income = df_calc[df_calc['Credit Amount'] > 0].copy()
    if df_income.empty: return pd.DataFrame({'Date': [], 'Credit Amount': []})

    try:
        if not isinstance(df_income.index, pd.DatetimeIndex):
            df_income.sort_values('Date', inplace=True)
            df_income.set_index('Date', inplace=True)
        # Consistent resampling parameters
        fortnightly_sum = df_income['Credit Amount'].resample('14D', label='right', closed='right').sum()
    except Exception as e:
        st.error(f"Error during income resampling: {e}")
        return pd.DataFrame({'Date': [], 'Credit Amount': []})

    return fortnightly_sum.reset_index()


# --- Plotting Functions ---
# (Keep plot_expenses_timeseries, plot_category_pie, plot_category_bar as provided)
def plot_expenses_timeseries(fortnightly_data):
    if fortnightly_data is None or not isinstance(fortnightly_data, pd.DataFrame) or fortnightly_data.empty: return None
    if 'Date' not in fortnightly_data.columns or 'Debit Amount' not in fortnightly_data.columns: return None
    try:
        fig = px.line(fortnightly_data, x='Date', y='Debit Amount', title="Fortnightly General Expenses Over Time (Excl. School Fees)", markers=True, labels={'Debit Amount': 'Total Expenses ($)'})
        fig.update_layout(xaxis_title="Fortnight Period Ending", yaxis_title="Total Expenses ($)")
        return fig
    except Exception as e: st.error(f"Error creating timeseries plot: {e}"); return None

def plot_category_pie(df, selected_date=None):
    """Plots expense distribution by category for a specific fortnight."""
    if df is None or not isinstance(df, pd.DataFrame): return None
    if 'Debit Amount' not in df.columns or 'Categories' not in df.columns: return None
    
    # Filter data for the selected fortnight if a date is provided
    if selected_date:
        # Convert selected_date to datetime if it's not already
        selected_date = pd.to_datetime(selected_date)
        # Get the start and end of the fortnight
        fortnight_start = selected_date - pd.Timedelta(days=13)
        fortnight_end = selected_date
        df_plot = df[(df['Date'] >= fortnight_start) & (df['Date'] <= fortnight_end)].copy()
    else:
        df_plot = df.copy()
    
    df_plot = df_plot[df_plot['Debit Amount'] > 0].copy()
    df_plot['Debit Amount'] = pd.to_numeric(df_plot['Debit Amount'], errors='coerce').fillna(0)
    df_plot = df_plot[df_plot['Debit Amount'] > 0]
    if df_plot.empty: return None
    try:
        expenses_by_cat = df_plot.groupby('Categories')['Debit Amount'].sum().reset_index()
        expenses_by_cat = expenses_by_cat[expenses_by_cat['Debit Amount'] > 0]
        if expenses_by_cat.empty: return None
        
        # Create the title with proper string formatting
        if selected_date:
            date_range = f" ({fortnight_start.strftime('%d %b %Y')} - {fortnight_end.strftime('%d %b %Y')})"
        else:
            date_range = ""
        title = f"Expense Distribution by Category{date_range}"
        
        # Create a consistent color map for all categories
        all_categories = sorted(df['Categories'].unique())
        color_map = {cat: px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)] 
                    for i, cat in enumerate(all_categories)}
        
        # Create the pie chart with consistent colors
        fig = px.pie(expenses_by_cat, 
                    names='Categories', 
                    values='Debit Amount', 
                    title=title, 
                    hole=0.3,
                    color='Categories',
                    color_discrete_map=color_map)
        
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


# --- NEW PLOTTING FUNCTION: Income vs Expense ---
def plot_income_vs_expense(df_combined, pay_adjustment=0, exact_amount=None):
    """Plots fortnightly income vs expenses using a combined bar chart."""
    if df_combined is None or not isinstance(df_combined, pd.DataFrame) or df_combined.empty:
        return None
    if not all(col in df_combined.columns for col in ['Date', 'Income', 'Expenses']):
        st.warning("Income/Expense plot error: Missing required columns ('Date', 'Income', 'Expenses').")
        return None

    try:
        import plotly.graph_objects as go
        
        # Create a copy of the DataFrame to avoid modifying the original
        df_plot = df_combined.copy()
        
        # Apply pay adjustment if specified
        if pay_adjustment != 0:
            # Calculate your current salary (total income - wife's salary)
            wife_salary = 2874.77  # Your wife's constant salary
            your_current_salary = df_plot['Income'].iloc[0] - wife_salary
            # Apply percentage adjustment to your salary only
            your_adjusted_salary = your_current_salary * (1 + pay_adjustment/100)
            # Set new total income (your adjusted salary + wife's salary)
            df_plot['Income'] = your_adjusted_salary + wife_salary
        elif exact_amount is not None:
            # Set new total income (your new salary + wife's salary)
            wife_salary = 2874.77  # Your wife's constant salary
            df_plot['Income'] = exact_amount + wife_salary
        
        # Create the figure
        fig = go.Figure()
        
        # Add income bar
        fig.add_trace(go.Bar(
            x=df_plot['Date'],
            y=df_plot['Income'],
            name='Income',
            marker_color='green',
            offsetgroup=0
        ))
        
        # Add expenses bar
        fig.add_trace(go.Bar(
            x=df_plot['Date'],
            y=df_plot['Expenses'],
            name='Expenses',
            marker_color='red',
            offsetgroup=1
        ))
        
        # Add school fees bar (stacked on expenses)
        fig.add_trace(go.Bar(
            x=df_plot['Date'],
            y=[1908] * len(df_plot),
            name='School Fees ($1908)',
            marker_color='rgba(255, 0, 0, 0.1)',
            marker_line_color='red',
            marker_line_width=1,
            offsetgroup=1,
            base=df_plot['Expenses']
        ))

        # Update the layout
        title = "Fortnightly Income vs Expenses"
        if pay_adjustment != 0:
            title += f" (Your salary adjusted by {pay_adjustment}%)"
        elif exact_amount is not None:
            total_income = exact_amount + 2874.77
            title += f" (Your salary: ${exact_amount:,.2f}, Total: ${total_income:,.2f})"
            
        fig.update_layout(
            title=title,
            xaxis_title="Fortnight Period Ending",
            yaxis_title="Amount ($)",
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1
        )

        return fig
    except Exception as e:
        st.error(f"Error creating Income vs Expense plot: {e}")
        return None


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
if 'what_if_pay_adjustment' not in st.session_state:
    st.session_state.what_if_pay_adjustment = 0

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
                
                # Display analysis in an expander at the top
                with st.expander("🔍 Transaction Analysis (Click to expand)", expanded=True):
                    st.write("### Problematic Transactions Analysis")
                    problematic_transactions = [
                        "WITHDRAWAL MOBILE 6776510 BPAY Bellingen Rates",
                        "WITHDRAWAL MOBILE 2519635 BPAY EnergyAust"
                    ]
                    
                    for transaction in problematic_transactions:
                        st.write("---")
                        st.write(f"#### Transaction: `{transaction}`")
                        
                        # Show analysis
                        analysis = analyze_transaction(transaction, CATEGORY_KEYWORDS)
                        st.write("**Matching Analysis:**")
                        st.write(analysis)
                        
                        # Show actual categorization
                        category = categorize_transaction(transaction, CATEGORY_KEYWORDS)
                        st.write(f"**Final Category:** {category}")
                        
                        # Show relevant keywords
                        st.write("**Relevant Keywords:**")
                        for cat, keywords in CATEGORY_KEYWORDS.items():
                            if any(kw in transaction.lower() for kw in keywords):
                                st.write(f"- {cat}: {', '.join(kw for kw in keywords if kw in transaction.lower())}")
                
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

    # --- Manual Categorization Section using st.data_editor ---
    # (Keep data_editor section as modified previously)
    st.write("---")
    st.header("✏️ Manual Categorization")
    uncategorized_mask = current_df['Categories'] == UNCATEGORIZED_LABEL
    df_to_edit = current_df.loc[uncategorized_mask, ['Date', 'Narrative', 'Debit Amount', 'Categories']].copy()
    if not df_to_edit.empty:
        st.warning(f"Found {len(df_to_edit)} transactions needing manual categorization.")
        st.markdown("**Select the correct category from the dropdown in the 'Categories' column below:**")
        column_config = {
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", disabled=True),
            "Narrative": st.column_config.TextColumn("Narrative", disabled=True),
            "Debit Amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f", disabled=True),
            "Categories": st.column_config.SelectboxColumn("Category", options=ALL_CATEGORY_OPTIONS, required=True)
        }
        edited_df = st.data_editor(df_to_edit, column_config=column_config, hide_index=True, use_container_width=True, key="category_editor")
        if not edited_df.reset_index(drop=True).equals(df_to_edit.reset_index(drop=True)):
             st.info("Applying category changes...")
             st.session_state.df_processed.loc[edited_df.index, 'Categories'] = edited_df['Categories']
             st.rerun()
    else:
        st.success("✅ All transactions are currently categorized!")


    # --- Download Button ---
    # (Keep as provided)
    st.download_button(
        label="📥 Download Categorized Data as Excel",
        data=to_excel(current_df),
        file_name=f'categorized_{st.session_state.processed_file_name}',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        key='download_button'
    )

    # --- Fortnightly Analysis Section ---
    # Calculate BOTH income and expenses here
    st.write("---")
    st.header("💹 Fortnightly Analysis")
    fortnightly_expenses_df, avg_fortnightly_expense = calculate_fortnightly_expenses(current_df)
    fortnightly_income_df = calculate_fortnightly_income(current_df) # Calculate income

    # --- Display Average Expense Metric ---
    if fortnightly_expenses_df is not None and not fortnightly_expenses_df.empty:
        st.metric(label="Current Average Fortnightly General Expense (Excl. School Fees)", value=f"${avg_fortnightly_expense:,.2f}")
        # Remove the expander for the expense data table here if it feels redundant with the new plot
        # with st.expander("View Current Fortnightly General Expense Data"):
        #    st.dataframe(fortnightly_expenses_df.style.format({"Date": "{:%Y-%m-%d}", "Debit Amount": "${:,.2f}"}))
    else:
        st.info("Fortnightly expense calculation did not yield results.")

    # --- Visualizations Section ---
    st.write("---")
    st.header("📈 Visualizations")
    
    # Calculate combined income and expenses data here
    combined_fortnightly_df = None
    if fortnightly_income_df is not None and fortnightly_expenses_df is not None:
        # Rename columns for clarity before merging
        fortnightly_income_df = fortnightly_income_df.rename(columns={'Credit Amount': 'Income'})
        fortnightly_expenses_df_plot = fortnightly_expenses_df.rename(columns={'Debit Amount': 'Expenses'})

        # Merge income and expenses on Date
        combined_fortnightly_df = pd.merge(
            fortnightly_income_df[['Date', 'Income']],
            fortnightly_expenses_df_plot[['Date', 'Expenses']],
            on='Date',
            how='outer'
        ).fillna(0).sort_values(by='Date')

        # Add Narrative column with a placeholder value for plotting
        combined_fortnightly_df['Narrative'] = 'Fortnightly Summary'
    
    # Add What-If Pay Adjustment Section
    st.subheader("What-If Pay Adjustment")
    
    # Create two columns for the adjustment options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Percentage Adjustment**")
        pay_adjustment = st.slider(
            "Adjust Take-Home Pay (%)",
            min_value=-50,
            max_value=100,
            value=0,
            step=5,
            help="Adjust your take-home pay by percentage to see the impact on the income vs expenses chart"
        )
    
    with col2:
        st.markdown("**Exact Amount**")
        exact_amount = st.number_input(
            "Set Your New Fortnightly Salary ($)",
            min_value=0.0,
            max_value=10000.0,
            value=0.0,
            step=100.0,
            help="Set your new fortnightly salary amount. Your wife's salary ($2,874.77) will be added to this amount."
        )
    
    # Move Income vs Expense plot here
    st.subheader("Fortnightly Income vs Expense")
    fig_income_expense = plot_income_vs_expense(combined_fortnightly_df, pay_adjustment, exact_amount if exact_amount > 0 else None)
    if fig_income_expense:
        st.plotly_chart(fig_income_expense, use_container_width=True)
    else:
        st.info("Insufficient data to display Income vs Expense chart.")

    # Keep original plots, perhaps in columns below the new one or adjust layout
    st.write("---") # Add separator
    col_viz1, col_viz2 = st.columns(2)
    with col_viz1:
        st.subheader("Expense Breakdown (All Categories)")
        
        # Add date slider for selecting fortnight
        if not current_df.empty:
            # Convert dates to datetime and get min/max
            min_date = pd.to_datetime(current_df['Date'].min())
            max_date = pd.to_datetime(current_df['Date'].max())
            
            # Create a list of fortnight end dates
            fortnight_dates = pd.date_range(start=min_date, end=max_date, freq='14D')
            if len(fortnight_dates) == 0:
                fortnight_dates = [max_date]
            
            # Create a dictionary of date strings to actual dates
            date_options = {date.strftime('%d %b %Y'): date for date in fortnight_dates}
            
            # Create the slider using the date strings
            selected_date_str = st.select_slider(
                "Select Fortnight End Date",
                options=list(date_options.keys()),
                value=list(date_options.keys())[-1]  # Default to most recent
            )
            
            # Convert back to datetime for the plot
            selected_date = date_options[selected_date_str]
        
        fig_pie = plot_category_pie(current_df, selected_date) # Plot pie based on selected fortnight
        if fig_pie: st.plotly_chart(fig_pie, use_container_width=True)
        else: st.info("No expense data available for pie chart.")

        st.subheader("Total Expenses per Category")
        fig_bar = plot_category_bar(current_df) # Plot bar based on all current data
        if fig_bar: st.plotly_chart(fig_bar, use_container_width=True)
        else: st.info("No expense data available for bar chart.")

    with col_viz2:
        st.subheader("General Expenses Over Time (Excl. School Fees)")
        # Use the original fortnightly expense calculation results for the timeseries plot
        fig_timeseries = plot_expenses_timeseries(fortnightly_expenses_df)
        if fig_timeseries: st.plotly_chart(fig_timeseries, use_container_width=True)
        else: st.info("No fortnightly expense data available to plot over time.")


# --- Initial State Message ---
# (Keep as provided)
elif not uploaded_file and st.session_state.get('df_processed') is None:
    st.info("Awaiting Excel file upload...")

# --- Sidebar Information ---
# (Keep as provided)
st.sidebar.title("About")
st.sidebar.info("""
    This app helps analyze bank transactions:
    1.  **Upload** your Excel statement.
    2.  Transactions are **auto-categorized**.
    3.  **Manually correct** categories using the editable table below.
    4.  View **fortnightly** income/expense summaries & charts.
    5.  Use **'What-If' sliders** to explore budget changes.
    6.  Explore interactive **charts**.
    7.  **Download** the categorized data.
    """)
st.sidebar.title("Categories & Keywords")
display_keywords = {cat: ', '.join(kw) for cat, kw in CATEGORY_KEYWORDS.items()}
with st.sidebar.expander("View Keywords (Code Level)"):
    st.json(display_keywords)
st.sidebar.markdown("*(To permanently add/modify keywords, edit the `CATEGORY_KEYWORDS` dictionary in the `app.py` script and restart the app.)*")

# --- END OF FILE app.py ---