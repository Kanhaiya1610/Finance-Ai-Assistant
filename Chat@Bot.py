import streamlit as st
import nltk
import speech_recognition as sr
import pyttsx3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from nltk.tokenize import word_tokenize
import numpy as np
import yfinance as yf
import time
import schedule
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Finance AI Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #2D336B;
    }
    .stButton button {
        background-color: #C5BAFF;
        color: black;
        padding: 10px 20px;
        border-radius: 5px;
        margin-top: 18px;
        padding-top: 13px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(1, 0, 1, 0.2);
        
    }
    .stButton button:hover {
        background-color: #493D9E;
        color: white;
        transform: translateY(+2px);
        box-shadow: 0 4px 6px rgba(1, 1, 1, 0.5);
    }
    .expense-card {
        background-color: #5b6ba8;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

class FinanceAssistant:
    def __init__(self):
        # Initialize NLTK
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            st.error(f"Error initializing NLTK: {str(e)}")

        # Initialize speech components
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        
        # Initialize session state
        if 'expenses' not in st.session_state:
            st.session_state.expenses = pd.DataFrame(
                columns=['date', 'amount', 'category', 'description']
            )
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Initialize budgets
        if 'budgets' not in st.session_state:
            st.session_state.budgets = {
                'food': 5000,
                'transport': 3000,
                'shopping': 4000,
                'bills': 10000,
                'entertainment': 2000,
                'others': 3000
            }
        
        # Initialize monthly targets
        if 'monthly_target' not in st.session_state:
            st.session_state.monthly_target = 25000
            
        # Initialize ML model for pattern recognition
        self.initialize_ml_components()

    def initialize_ml_components(self):
        """Initialize ML components for spending pattern analysis"""
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        
    def analyze_spending_patterns(self):
        """Analyze spending patterns and generate insights"""
        if not st.session_state.expenses.empty:
            # Group by date and category
            daily_spending = st.session_state.expenses.groupby(
                [st.session_state.expenses['date'].dt.date, 'category']
            )['amount'].sum().reset_index()
            
            # Generate insights
            insights = []
            
            # Check budget overruns
            for category in st.session_state.budgets.keys():
                monthly_spent = st.session_state.expenses[
                    (st.session_state.expenses['category'] == category) &
                    (st.session_state.expenses['date'] > datetime.now() - timedelta(days=30))
                ]['amount'].sum()
                
                budget = st.session_state.budgets[category]
                if monthly_spent > budget:
                    insights.append(f"⚠️ You've exceeded your {category} budget by ₹{monthly_spent - budget:,.2f}")
                elif monthly_spent > 0.8 * budget:
                    insights.append(f"⚠️ You're close to your {category} budget limit ({monthly_spent/budget*100:.1f}% used)")
            
            # Unusual spending patterns
            for category in daily_spending['category'].unique():
                cat_spending = daily_spending[daily_spending['category'] == category]['amount']
                if len(cat_spending) > 0:
                    mean_spending = cat_spending.mean()
                    std_spending = cat_spending.std()
                    if std_spending > mean_spending:
                        insights.append(f"📊 Your {category} spending shows high variation")
            
            return insights
        return []

    def get_budget_status(self):
        """Get current budget status for all categories"""
        if not st.session_state.expenses.empty:
            current_month = datetime.now().month
            monthly_expenses = st.session_state.expenses[
                st.session_state.expenses['date'].dt.month == current_month
            ]
            
            budget_status = {}
            for category, budget in st.session_state.budgets.items():
                spent = monthly_expenses[
                    monthly_expenses['category'] == category
                ]['amount'].sum()
                budget_status[category] = {
                    'budget': budget,
                    'spent': spent,
                    'remaining': budget - spent,
                    'percentage': (spent/budget*100) if budget > 0 else 0
                }
            return budget_status
        return {}

    def listen(self):
        try:
            with sr.Microphone() as source:
                st.info("🎤 Listening... Speak now!")
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio)
                return text
        except sr.WaitTimeoutError:
            st.warning("No speech detected. Please try again.")
            return None
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None

    def process_expense(self, text):
        # Extract amount using regex
        import re
        amounts = re.findall(r'\d+', text)
        if amounts:
            amount = float(amounts[0])
            
            # Categorize expense
            categories = {
                'food': ['food', 'meal', 'restaurant', 'lunch', 'dinner'],
                'transport': ['uber', 'taxi', 'fuel', 'gas', 'travel'],
                'shopping': ['bought', 'purchased', 'shopping', 'clothes'],
                'bills': ['bill', 'utility', 'electricity', 'water'],
                'entertainment': ['movie', 'game', 'entertainment']
            }
            
            category = 'others'
            for cat, keywords in categories.items():
                if any(keyword in text.lower() for keyword in keywords):
                    category = cat
                    break
            
            # Add to expenses
            new_expense = pd.DataFrame([{
                'date': datetime.now(),
                'amount': amount,
                'category': category,
                'description': text
            }])
            
            st.session_state.expenses = pd.concat(
                [st.session_state.expenses, new_expense], 
                ignore_index=True
            )
            
            return f"💰 Added expense: ₹{amount} for {category}"
        return "❌ Could not identify the expense amount"

class StockMarket:
    def __init__(self):
        self.cached_data = {}
        self.cache_timeout = 300  # 5 minutes cache
        
    def get_stock_price(self, symbol):
        try:
            # Check cache first
            if symbol in self.cached_data:
                if time.time() - self.cached_data[symbol]['timestamp'] < self.cache_timeout:
                    return self.cached_data[symbol]['data']
            
            # Fetch new data
            stock = yf.Ticker(symbol)
            info = stock.info
            current_price = info.get('regularMarketPrice', 0)
            prev_close = info.get('previousClose', 0)
            change = ((current_price - prev_close) / prev_close) * 100
            
            data = {
                'price': current_price,
                'change': change,
                'currency': info.get('currency', 'USD'),
                'company_name': info.get('longName', symbol)
            }
            
            # Update cache
            self.cached_data[symbol] = {
                'data': data,
                'timestamp': time.time()
            }
            
            return data
        except Exception as e:
            st.error(f"Error fetching stock data: {str(e)}")
            return None
    
    def get_historical_data(self, symbol, period='1mo'):
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            return hist
        except Exception as e:
            st.error(f"Error fetching historical data: {str(e)}")
            return None

class BillReminder:
    def __init__(self):
        if 'bills' not in st.session_state:
            st.session_state.bills = []
        if 'total_savings' not in st.session_state:
            st.session_state.total_savings = 0
    
    def add_bill(self, name, amount, due_date, recurring=False, frequency='monthly'):
        bill = {
            'name': name,
            'amount': amount,
            'due_date': due_date,
            'recurring': recurring,
            'frequency': frequency,
            'status': 'pending'
        }
        st.session_state.bills.append(bill)
    
    def get_upcoming_bills(self, days=7):
        today = datetime.now()
        upcoming = []
        for bill in st.session_state.bills:
            due_date = datetime.strptime(bill['due_date'], '%Y-%m-%d')
            if (due_date - today).days <= days and bill['status'] == 'pending':
                upcoming.append(bill)
        return upcoming
    
    def mark_as_paid(self, bill_index):
        """Update bill status and adjust savings when bill is paid"""
        bill = st.session_state.bills[bill_index]
        
        # Subtract bill amount from savings
        if st.session_state.total_savings >= bill['amount']:
            st.session_state.total_savings -= bill['amount']
            
            # Update bill status
            st.session_state.bills[bill_index]['status'] = 'paid'
            
            # Add to expenses
            new_expense = pd.DataFrame([{
                'date': datetime.now(),
                'amount': bill['amount'],
                'category': 'bills',
                'description': f"Paid bill: {bill['name']}"
            }])
            
            st.session_state.expenses = pd.concat(
                [st.session_state.expenses, new_expense],
                ignore_index=True
            )
            
            # Create next bill if recurring
            if bill['recurring']:
                old_due_date = datetime.strptime(bill['due_date'], '%Y-%m-%d')
                
                if bill['frequency'] == 'monthly':
                    new_due_date = old_due_date + timedelta(days=30)
                elif bill['frequency'] == 'weekly':
                    new_due_date = old_due_date + timedelta(days=7)
                
                self.add_bill(
                    bill['name'],
                    bill['amount'],
                    new_due_date.strftime('%Y-%m-%d'),
                    True,
                    bill['frequency']
                )
            
            return True, f"Bill paid and ₹{bill['amount']:,.2f} deducted from savings"
        else:
            return False, f"Insufficient savings to pay bill of ₹{bill['amount']:,.2f}"

class FinanceAnalyzer:
    def __init__(self):
        if 'total_savings' not in st.session_state:
            st.session_state.total_savings = 0
        if 'last_saved_amount' not in st.session_state:
            st.session_state.last_saved_amount = 0
    
    def calculate_financial_status(self, budgets, expenses, bills):
        """Calculate overall financial status including bills and savings"""
        
        financial_summary = {
            'total_budget': sum(budgets.values()),
            'total_expenses': 0,
            'total_bills': 0,
            'remaining_budget': 0,
            'potential_savings': 0,
            'category_analysis': {},
            'upcoming_bills': [],
            'status': 'neutral',
            'recommendations': []
        }
        
        # Calculate total expenses
        if not expenses.empty:
            current_month = datetime.now().month
            monthly_expenses = expenses[expenses['date'].dt.month == current_month]
            financial_summary['total_expenses'] = monthly_expenses['amount'].sum()
            
            # Category-wise analysis
            for category in budgets.keys():
                cat_expenses = monthly_expenses[
                    monthly_expenses['category'] == category
                ]['amount'].sum()
                
                financial_summary['category_analysis'][category] = {
                    'budget': budgets[category],
                    'spent': cat_expenses,
                    'remaining': budgets[category] - cat_expenses,
                    'percentage': (cat_expenses / budgets[category] * 100) if budgets[category] > 0 else 0
                }
        
        # Calculate total upcoming bills
        upcoming_bills = [bill for bill in bills if bill['status'] == 'pending']
        financial_summary['total_bills'] = sum(bill['amount'] for bill in upcoming_bills)
        financial_summary['upcoming_bills'] = upcoming_bills
        
        # Calculate potential savings
        financial_summary['remaining_budget'] = (
            financial_summary['total_budget'] - 
            financial_summary['total_expenses']
        )
        
        financial_summary['potential_savings'] = max(0, 
            financial_summary['remaining_budget'] - 
            financial_summary['total_bills']
        )
        
        if financial_summary['potential_savings'] > 0:
            financial_summary['status'] = 'positive'
        else:
            financial_summary['status'] = 'negative'
        
        return financial_summary

def create_budget_chart(budget_status):
    """Create a budget status chart using plotly"""
    categories = list(budget_status.keys())
    spent = [status['spent'] for status in budget_status.values()]
    budgets = [status['budget'] for status in budget_status.values()]
    
    fig = go.Figure(data=[
        go.Bar(name='Spent', x=categories, y=spent),
        go.Bar(name='Budget', x=categories, y=budgets)
    ])
    
    fig.update_layout(
        barmode='group',
        title='Budget vs Spending by Category',
        xaxis_title='Category',
        yaxis_title='Amount (₹)'
    )
    return fig

def create_stock_chart(symbol, data):
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=('Price', 'Volume'))
    
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{symbol} Stock Price',
        yaxis_title='Price',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    return fig

def create_tradingview_widget(symbol="NASDAQ:NVDA"):
    widget_html = f"""
    <div class="tradingview-widget-container">
        <div class="tradingview-widget-container__widget"></div>
        <div class="tradingview-widget-copyright">
            <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
                <span class="blue-text">Track all markets on TradingView</span>
            </a>
        </div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
        {{
            "width": "1078",
            "height": "732",
            "symbol": "{symbol}",
            "interval": "D",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "displayMode": "adaptive",
            "allow_symbol_change": true,
            "details": true,
            "calendar": false,
            "show_popup_button": true,
            "popup_width": "1000",
            "popup_height": "650",
            "support_host": "https://www.tradingview.com"
        }}
        </script>
    </div>
    """
    return widget_html

def create_mini_symbol_widget(symbol="FX:EURUSD"):
    mini_widget_html = f"""
    <div class="tradingview-widget-container">
        <div class="tradingview-widget-container__widget"></div>
        <div class="tradingview-widget-copyright">
            <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
                <span class="blue-text">Track all markets on TradingView</span>
            </a>
        </div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
        {{
            "symbol": "{symbol}",
            "width": 350,
            "height": 220,
            "locale": "en",
            "displayMode": "adaptive",
            "dateRange": "12M",
            "colorTheme": "dark",
            "isTransparent": false,
            "autosize": false,
            "largeChartUrl": ""
        }}
        </script>
    </div>
    """
    return mini_widget_html

def main():
    st.title("💰 Finance AI Assistant")
    assistant = FinanceAssistant()
    
    # Initialize components
    stock_market = StockMarket()
    bill_reminder = BillReminder()
    
    # Enhanced sidebar navigation with emojis and styling
    with st.sidebar:
        st.markdown("""
            <style>
            .sidebar-nav {
                padding: 1rem;
                background-color: #f8f9fa;
                border-radius: 10px;
                margin: 1rem 0;
            }
            .nav-header {
                color: #1f2937;
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 1rem;
                text-align: center;
            }
            .stRadio > label {
                font-size: 18px;
                font-weight: 500;
                padding: 10px;
                border-radius: 5px;
                margin: 5px 0;
                transition: all 0.3s ease;
            }
            .stRadio > label:hover {
                background-color: #e2e8f0;
                cursor: pointer;
            }
            </style>
            <div class="sidebar-nav">
                <div class="nav-header">🎯 Navigation</div>
            </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "",  # Empty label as we're using custom header
            [
                "💬 Chat Assistant",
                "📊 Expense Tracker",
                "💰 Budget Manager",
                "📈 Stock Tracker",
                "📅 Bill Reminders",
                "🔍 Financial Insights"
            ],
            key="navigation"
        )
    
    # Update the page conditions to match new labels
    if "💬 Chat Assistant" in page:
        # Chat page code...
        st.subheader("💬 Chat with your Finance Assistant")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_input("Type your message:", key="text_input")
            if user_input:
                response = assistant.process_expense(user_input)
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Assistant", response))
        
        with col2:
            if st.button("🎤 Voice Input"):
                voice_input = assistant.listen()
                if voice_input:
                    st.write(f"You said: {voice_input}")
                    response = assistant.process_expense(voice_input)
                    st.session_state.chat_history.append(("You", voice_input))
                    st.session_state.chat_history.append(("Assistant", response))
        
        # Display chat history
        st.subheader("Chat History")
        for role, message in st.session_state.chat_history:
            if role == "You":
                st.markdown(f"**You**: {message}")
            else:
                st.markdown(f"**Assistant**: {message}")
    
    elif "📊 Expense Tracker" in page:
        # Expense tracker code...
        st.subheader("💰 Expense Tracker")
        if not st.session_state.expenses.empty:
            # Display expenses in a nice format
            for _, row in st.session_state.expenses.iterrows():
                with st.container():
                    st.markdown(f"""
                        <div class='expense-card'>
                            <h4>₹{row['amount']} - {row['category'].title()}</h4>
                            <p>{row['description']}</p>
                            <small>{row['date'].strftime('%Y-%m-%d %H:%M')}</small>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No expenses recorded yet! Start by adding some expenses in the chat.")
    
    elif "💰 Budget Manager" in page:
        st.subheader("💰 Budget & Savings Manager")
        
        # Initialize finance analyzer
        finance_analyzer = FinanceAnalyzer()
        
        # Add Budget Setting Section
        with st.expander("🎯 Set Your Budgets", expanded=True):
            st.markdown("### Set Monthly Budget Limits")
            
            # Initialize budgets if not exists
            if 'budgets' not in st.session_state:
                st.session_state.budgets = {
                    'food': 0,
                    'transport': 0,
                    'shopping': 0,
                    'bills': 0,
                    'entertainment': 0,
                    'others': 0
                }
            
            # Create three columns for budget inputs
            col1, col2, col3 = st.columns(3)
            
            with col1:
                food_budget = st.number_input(
                    "🍜 Food Budget",
                    min_value=0,
                    value=st.session_state.budgets['food'],
                    step=100,
                    help="Set your monthly food budget"
                )
                
                transport_budget = st.number_input(
                    "🚗 Transport Budget",
                    min_value=0,
                    value=st.session_state.budgets['transport'],
                    step=100,
                    help="Set your monthly transport budget"
                )
            
            with col2:
                shopping_budget = st.number_input(
                    "🛍️ Shopping Budget",
                    min_value=0,
                    value=st.session_state.budgets['shopping'],
                    step=100,
                    help="Set your monthly shopping budget"
                )
                
                bills_budget = st.number_input(
                    "📃 Bills Budget",
                    min_value=0,
                    value=st.session_state.budgets['bills'],
                    step=100,
                    help="Set your monthly bills budget"
                )
            
            with col3:
                entertainment_budget = st.number_input(
                    "🎮 Entertainment Budget",
                    min_value=0,
                    value=st.session_state.budgets['entertainment'],
                    step=100,
                    help="Set your monthly entertainment budget"
                )
                
                others_budget = st.number_input(
                    "📦 Others Budget",
                    min_value=0,
                    value=st.session_state.budgets['others'],
                    step=100,
                    help="Set your monthly budget for other expenses"
                )
            
            # Calculate total budget
            total_budget = (food_budget + transport_budget + shopping_budget + 
                           bills_budget + entertainment_budget + others_budget)
            
            # Update button
            if st.button("💾 Update Budgets"):
                st.session_state.budgets.update({
                    'food': food_budget,
                    'transport': transport_budget,
                    'shopping': shopping_budget,
                    'bills': bills_budget,
                    'entertainment': entertainment_budget,
                    'others': others_budget
                })
                st.success(f"✅ Budgets updated successfully! Total Budget: ₹{total_budget:,.2f}")
        
        # Show current budget allocation
        if total_budget > 0:
            st.markdown("### 📊 Budget Allocation")
            
            # Create a pie chart of budget allocation
            budget_data = pd.DataFrame({
                'Category': st.session_state.budgets.keys(),
                'Amount': st.session_state.budgets.values()
            })
            
            fig = px.pie(
                budget_data,
                values='Amount',
                names='Category',
                title='Budget Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            st.plotly_chart(fig)
            
            # Show budget breakdown
            st.markdown("### 💰 Budget Breakdown")
            cols = st.columns(3)
            for i, (category, amount) in enumerate(st.session_state.budgets.items()):
                with cols[i % 3]:
                    percentage = (amount / total_budget * 100) if total_budget > 0 else 0
                    st.markdown(f"""
                        <div style='
                            padding: 10px;
                            border-radius: 5px;
                            background-color: {px.colors.qualitative.Set3[i]};
                            margin: 5px 0;
                        '>
                            <b>{category.title()}</b><br>
                            ₹{amount:,.2f}<br>
                            {percentage:.1f}%
                        </div>
                    """, unsafe_allow_html=True)
        
        # Get financial status
        financial_status = finance_analyzer.calculate_financial_status(
            st.session_state.budgets,
            st.session_state.expenses,
            st.session_state.bills
        )
        
        # Display financial overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Budget",
                f"₹{financial_status['total_budget']:,.2f}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Total Expenses & Bills",
                f"₹{(financial_status['total_expenses'] + financial_status['total_bills']):,.2f}",
                delta=f"-₹{(financial_status['total_expenses'] + financial_status['total_bills']):,.2f}",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                "Potential Savings",
                f"₹{financial_status['potential_savings']:,.2f}",
                delta=f"+₹{financial_status['potential_savings']:,.2f}" if financial_status['potential_savings'] > 0 else "₹0",
                delta_color="normal"
            )
        
        # Display detailed analysis
        st.markdown("### 📊 Detailed Analysis")
        
        # Category-wise analysis
        for category, analysis in financial_status['category_analysis'].items():
            with st.expander(f"{category.title()} Analysis"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Budget", f"₹{analysis['budget']:,.2f}")
                with col2:
                    st.metric("Spent", f"₹{analysis['spent']:,.2f}")
                with col3:
                    st.metric("Remaining", f"₹{analysis['remaining']:,.2f}")
                with col4:
                    st.progress(min(analysis['percentage'] / 100, 1.0))
                    st.text(f"{analysis['percentage']:.1f}% used")
        
        # Upcoming bills
        st.markdown("### 📅 Impact of Upcoming Bills")
        if financial_status['upcoming_bills']:
            for bill in financial_status['upcoming_bills']:
                st.warning(
                    f"📌 {bill['name']}: ₹{bill['amount']:,.2f} due on {bill['due_date']}"
                )
        else:
            st.info("No upcoming bills")
        
        # Recommendations
        st.markdown("### 💡 Financial Recommendations")
        for recommendation in financial_status['recommendations']:
            if recommendation.startswith("⚠️"):
                st.error(recommendation)
            elif recommendation.startswith("✅"):
                st.success(recommendation)
            else:
                st.info(recommendation)
        
        # Savings tracker
        st.markdown("### 💰 Savings Tracker")
        
        # Show potential new savings
        potential_savings = financial_status['potential_savings']
        
        if potential_savings > 0:
            st.info(f"Potential savings available: ₹{potential_savings:,.2f}")
            
            if st.button("Add to Savings"):
                # If there was a previous saved amount, subtract it
                if st.session_state.last_saved_amount > 0:
                    st.session_state.total_savings -= st.session_state.last_saved_amount
                
                # Add new potential savings
                st.session_state.total_savings += potential_savings
                st.session_state.last_saved_amount = potential_savings
                
                st.success(f"Updated savings! Added ₹{potential_savings:,.2f}")
        else:
            if st.session_state.last_saved_amount > 0:
                st.warning(f"⚠️ Your previous savings of ₹{st.session_state.last_saved_amount:,.2f} need to be adjusted due to new expenses/bills")
                
                if st.button("Adjust Savings"):
                    st.session_state.total_savings -= st.session_state.last_saved_amount
                    st.session_state.last_saved_amount = 0
                    st.success("Savings adjusted successfully!")
        
        # Display total savings
        st.metric(
            "Total Savings",
            f"₹{st.session_state.total_savings:,.2f}",
            delta=f"₹{potential_savings:,.2f}" if potential_savings > 0 else None
        )
    
    elif "📈 Stock Tracker" in page:
        st.subheader("📈 Stock Market Tracker")
        
        tab1, tab2 = st.tabs(["📊 Chart View", "💰 Market Overview"])
        
        with tab1:
            # Add dark theme to Streamlit
            st.markdown("""
                <style>
                .stApp {
                    background-color: #1E222D;
                    color: #FFFFFF;
                }
                .stTextInput > div > div > input {
                    background-color: #2A2E39;
                    color: #FFFFFF;
                }
                .stSelectbox > div > div > select {
                    background-color: #2A2E39;
                    color: #FFFFFF;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Stock search with improved styling
            symbol = st.text_input(
                "Enter Stock Symbol (e.g., NVDA, AAPL, GOOGL)",
                value="NVDA"
            ).upper()
            
            if symbol:
                tradingview_symbol = f"NASDAQ:{symbol}"
            else:
                tradingview_symbol = "NASDAQ:NVDA"  # Default symbol
            
            # Display TradingView Widget
            widget_html = create_tradingview_widget(tradingview_symbol)
            st.components.v1.html(widget_html, height=800)
            
            # Market Information with dark theme styling
            if symbol:
                stock_data = stock_market.get_stock_price(symbol)
                if stock_data:
                    st.markdown("""
                        <style>
                        .metric-container {
                            background-color: #2A2E39;
                            padding: 20px;
                            border-radius: 10px;
                            margin: 10px 0;
                        }
                        .metric-label {
                            color: #9598A1;
                            font-size: 14px;
                        }
                        .metric-value {
                            color: #FFFFFF;
                            font-size: 24px;
                            font-weight: bold;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-label">Current Price</div>
                                <div class="metric-value">{stock_data['currency']} {stock_data['price']:,.2f}</div>
                                <div style="color: {'#00C805' if stock_data['change'] > 0 else '#FF5000'}">
                                    {stock_data['change']:+.2f}%
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-label">Company</div>
                                <div class="metric-value">{stock_data['company_name']}</div>
                            </div>
                        """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### Market Overview")
            
            # Create 3x2 grid for different markets
            col1, col2, col3 = st.columns(3)
            
            # Major Forex Pairs
            with col1:
                st.markdown("#### Forex")
                st.components.v1.html(
                    create_mini_symbol_widget("FX:EURUSD"), 
                    height=250
                )
                st.components.v1.html(
                    create_mini_symbol_widget("FX:GBPUSD"), 
                    height=250
                )
            
            # Major Stock Indices
            with col2:
                st.markdown("#### Indices")
                st.components.v1.html(
                    create_mini_symbol_widget("NASDAQ:NDX"), 
                    height=250
                )
                st.components.v1.html(
                    create_mini_symbol_widget("FOREXCOM:SPXUSD"), 
                    height=250
                )
            
            # Popular Stocks
            with col3:
                st.markdown("#### Stocks")
                st.components.v1.html(
                    create_mini_symbol_widget("NASDAQ:AAPL"), 
                    height=250
                )
                st.components.v1.html(
                    create_mini_symbol_widget("NASDAQ:GOOGL"), 
                    height=250
                )
            
            # Additional Market Information
            st.markdown("### Market Summary")
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.markdown("""
                    #### Today's Market Movers
                    - EUR/USD: Major currency pair
                    - S&P 500: US stock market index
                    - NASDAQ: Tech-heavy index
                """)
            
            with summary_col2:
                st.markdown("""
                    #### Key Market Events
                    - Federal Reserve Meetings
                    - Economic Data Releases
                    - Company Earnings
                """)
    
    elif "📅 Bill Reminders" in page:
        st.subheader("📅 Bill Reminder System")
        
        # Add new bill
        with st.expander("Add New Bill"):
            col1, col2 = st.columns(2)
            with col1:
                bill_name = st.text_input("Bill Name")
                bill_amount = st.number_input("Amount", min_value=0.0)
            with col2:
                due_date = st.date_input("Due Date")
                recurring = st.checkbox("Recurring Bill")
            
            if recurring:
                frequency = st.selectbox("Frequency", ['monthly', 'weekly'])
            else:
                frequency = 'once'
            
            if st.button("Add Bill"):
                bill_reminder.add_bill(
                    bill_name,
                    bill_amount,
                    due_date.strftime('%Y-%m-%d'),
                    recurring,
                    frequency
                )
                st.success("Bill added successfully!")
        
        # Display upcoming bills
        st.subheader("Upcoming Bills")
        upcoming_bills = bill_reminder.get_upcoming_bills()
        
        if upcoming_bills:
            # Show current savings
            st.info(f"Current Savings: ₹{st.session_state.total_savings:,.2f}")
            
            for i, bill in enumerate(upcoming_bills):
                with st.container():
                    col1, col2, col3, col4 = st.columns([2,1,1,1])
                    with col1:
                        st.write(f"**{bill['name']}**")
                    with col2:
                        st.write(f"₹{bill['amount']:,.2f}")
                    with col3:
                        st.write(bill['due_date'])
                    with col4:
                        if st.button("Mark as Paid", key=f"pay_bill_{i}"):
                            success, message = bill_reminder.mark_as_paid(i)
                            if success:
                                st.success(message)
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(message)
        else:
            st.info("No upcoming bills!")
        
        # Show paid bills history
        st.subheader("Paid Bills History")
        paid_bills = [bill for bill in st.session_state.bills if bill['status'] == 'paid']
        if paid_bills:
            for bill in paid_bills:
                st.markdown(f"""
                    <div style='padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin: 5px 0;'>
                        ✅ {bill['name']} - ₹{bill['amount']:,.2f} - Paid
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No paid bills history")
    
    elif "🔍 Financial Insights" in page:
        # Insights code...
        st.subheader("📊 Financial Insights")
        
        # Generate and display insights
        insights = assistant.analyze_spending_patterns()
        if insights:
            for insight in insights:
                st.info(insight)
        else:
            st.write("Add more transactions to get personalized insights!")
        
        # Spending trends
        if not st.session_state.expenses.empty:
            st.subheader("Spending Trends")
            daily_spending = st.session_state.expenses.groupby(
                st.session_state.expenses['date'].dt.date
            )['amount'].sum().reset_index()
            
            fig = px.line(
                daily_spending,
                x='date',
                y='amount',
                title='Daily Spending Trend'
            )
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
