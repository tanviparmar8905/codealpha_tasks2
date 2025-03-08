import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import datetime
import time
import threading
import sqlite3
import concurrent.futures
import yfinance as yf

class AdvancedStockPortfolioTracker:
    """
    Advanced stock portfolio tracker with individual stock performance,
    and other enhancements.
    """

    def __init__(self, db_file="portfolio.db", refresh_interval=10):  # Update interval set to 10
        self.db_file = db_file
        self.refresh_interval = refresh_interval
        self.portfolio = {}
        self.portfolio_value_history = pd.DataFrame(columns=["Timestamp", "Value"])
        self.lock = threading.Lock()
        self.running = True
        self.conn = sqlite3.connect(self.db_file)
        self.create_tables()
        self.load_transactions()
        self.thread = threading.Thread(target=self._real_time_update)
        self.thread.daemon = True
        self.thread.start()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                type TEXT NOT NULL,
                shares REAL NOT NULL,
                price REAL NOT NULL,
                commission REAL DEFAULT 0
            )
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_value (
                timestamp TEXT PRIMARY KEY,
                value REAL NOT NULL
            )
        """
        )
        self.conn.commit()

    def load_transactions(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM transactions")
        rows = cursor.fetchall()
        self.transactions = [
            {
                "ticker": row[1],
                "date": row[2],
                "type": row[3],
                "shares": row[4],
                "price": row[5],
                "commission": row[6],
            }
            for row in rows
        ]

    def add_transaction(self, transaction):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO transactions (ticker, date, type, shares, price, commission)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                transaction["ticker"],
                transaction["date"],
                transaction["type"],
                transaction["shares"],
                transaction["price"],
                transaction.get("commission", 0),
            ),
        )
        self.conn.commit()
        self.transactions.append(transaction)

    def _process_transactions(self):
        self.portfolio = {}
        for transaction in self.transactions:
            ticker = transaction["ticker"]
            shares = transaction["shares"]
            transaction_type = transaction["type"]
            if ticker not in self.portfolio:
                self.portfolio[ticker] = 0
            if transaction_type == "buy":
                self.portfolio[ticker] += shares
            elif transaction_type == "sell":
                self.portfolio[ticker] -= shares
        self.portfolio = {ticker: shares for ticker, shares in self.portfolio.items() if shares > 0}

    def _get_real_time_price(self, ticker):
        try:
            ticker_data = yf.Ticker(ticker)
            current_price = ticker_data.info.get('currentPrice')
            if current_price is None:
                print(f"Real-time price not found for {ticker}")
            return current_price
        except Exception as e:
            print(f"Error fetching real-time data for {ticker}: {e}")
            return None

    def _real_time_update(self):
        while self.running:
            self._process_transactions()
            total_value = 0

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._get_real_time_price, ticker) for ticker in self.portfolio]
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    price = future.result()
                    if price:
                        ticker = list(self.portfolio.keys())[i]
                        total_value += self.portfolio[ticker] * price

            with self.lock:
                # Use concat with a new DataFrame
                new_data = pd.DataFrame({"Timestamp": [datetime.datetime.now()], "Value": [total_value]})
                self.portfolio_value_history = pd.concat([self.portfolio_value_history, new_data], ignore_index=True)

                # Create a new connection and cursor within this thread
                thread_conn = sqlite3.connect(self.db_file)
                cursor = thread_conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO portfolio_value (timestamp, value) VALUES (?, ?)",
                    (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), total_value),
                )
                thread_conn.commit()
                thread_conn.close()

            time.sleep(self.refresh_interval)

    def plot_portfolio_value(self):
        with self.lock:
            portfolio_value_history_copy = self.portfolio_value_history.copy()
            # Get a copy of the current portfolio holdings
            portfolio_holdings = self.portfolio.copy()

        if portfolio_value_history_copy.empty:
            print("No portfolio value data to plot.")
            return

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        fig.add_trace(
            go.Scatter(
                x=portfolio_value_history_copy["Timestamp"],
                y=portfolio_value_history_copy["Value"],
                mode="lines",
                name="Portfolio Value",
            ),
            row=1,
            col=1,
        )
        daily_returns = portfolio_value_history_copy["Value"].pct_change().dropna()
        fig.add_trace(go.Bar(x=daily_returns.index, y=daily_returns, name="Daily Returns"), row=2, col=1)
        fig.update_layout(title="Real-Time Portfolio Value and Daily Returns", xaxis_title="Time", yaxis_title="Value")

        # Add an annotation to display the current portfolio holdings
        holdings_text = "<br>".join([f"{ticker}: {shares} shares" for ticker, shares in portfolio_holdings.items()])
        fig.add_annotation(
            text=f"Current Holdings:<br>{holdings_text}",
            xref="paper",
            yref="paper",
            x=1.05,  # Position to the right of the graph
            y=0.5,   # Center vertically
            showarrow=False,
            align="left",
        )

        fig.show()

    def calculate_sharpe_ratio(self):
        with self.lock:
            portfolio_value_history_copy = self.portfolio_value_history.copy()
        daily_returns = portfolio_value_history_copy["Value"].pct_change().dropna()
        if daily_returns.empty:
            return 0  # Return 0 if no returns data is available
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * (252**0.5)
        return sharpe_ratio

    def calculate_max_drawdown(self):
        with self.lock:
            portfolio_value_history_copy = self.portfolio_value_history.copy()
        portfolio_values = portfolio_value_history_copy["Value"]
        peak = portfolio_values.iloc[0]
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown

    def get_stock_info(self, ticker):
        """Fetches and displays key metrics for a single stock."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            print(f"Stock: {ticker}")
            print(f"Current Price: {info.get('currentPrice')}")
            print(f"Day High/Low: {info.get('dayHigh')}/{info.get('dayLow')}")
            print(f"52-Week High/Low: {info.get('fiftyTwoWeekHigh')}/{info.get('fiftyTwoWeekLow')}")
            print(f"Volume: {info.get('volume')}")
            # ... add more metrics as needed

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    def stop(self):
        self.running = False
        self.thread.join()
        self.conn.close()

app = dash.Dash(__name__)

tracker = AdvancedStockPortfolioTracker(db_file="portfolio.db", refresh_interval=10)  # Update interval set to 10
tracker.add_transaction({"ticker": "AAPL", "date": "2023-01-01", "type": "buy", "shares": 10, "price": 150.0, "commission": 5.0})
tracker.add_transaction({"ticker": "MSFT", "date": "2023-02-15", "type": "buy", "shares": 5, "price": 280.0})
tracker.add_transaction({"ticker": "GOOG", "date": "2023-03-01", "type": "buy", "shares": 3, "price": 2000.0})

app.layout = html.Div([
    dcc.Graph(id='live-graph'),
    html.Div(id='stock-details'),
    dcc.Interval(
        id='interval-component',
        interval=10 * 1000,  # in milliseconds
        n_intervals=0
    )
])

@app.callback(
    Output('live-graph', 'figure'),
    Output('stock-details', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_dashboard(n):
    with tracker.lock:
        portfolio_value_history_copy = tracker.portfolio_value_history.copy()
        portfolio_holdings = tracker.portfolio.copy()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig.add_trace(
        go.Scatter(
            x=portfolio_value_history_copy["Timestamp"],
            y=portfolio_value_history_copy["Value"],
            mode="lines",
            name="Portfolio Value",
        ),
        row=1,
        col=1,
    )
    daily_returns = portfolio_value_history_copy["Value"].pct_change().dropna()
    fig.add_trace(go.Bar(x=daily_returns.index, y=daily_returns, name="Daily Returns"), row=2, col=1)
    fig.update_layout(title="Real-Time Portfolio Value and Daily Returns", xaxis_title="Time", yaxis_title="Value")

    holdings_text = "<br>".join([f"{ticker}: {shares} shares" for ticker, shares in portfolio_holdings.items()])
    fig.add_annotation(
        text=f"Current Holdings:<br>{holdings_text}",
        xref="paper",
        yref="paper",
        x=1.05,
        y=0.5,
        showarrow=False,
        align="left",
    )

    stock_details_html =[]
    for ticker in tracker.portfolio.keys():
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            # news = stock.news  # Removed news fetching
            stock_details_html.append(html.H3(f"Stock: {ticker}"))
            stock_details_html.append(html.P(f"Current Price: {info.get('currentPrice')}"))
            stock_details_html.append(html.P(f"Day High/Low: {info.get('dayHigh')}/{info.get('dayLow')}"))
            stock_details_html.append(html.P(f"52-Week High/Low: {info.get('fiftyTwoWeekHigh')}/{info.get('fiftyTwoWeekLow')}"))
            stock_details_html.append(html.P(f"Volume: {info.get('volume')}"))

            # Removed news display section

        except Exception as e:
            stock_details_html.append(html.P(f"Error fetching data for {ticker}: {e}"))

    return fig, html.Div(stock_details_html)

if __name__ == '__main__':
    app.run_server(debug=True)