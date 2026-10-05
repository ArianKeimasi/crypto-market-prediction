# Import the Tkinter library to build the graphical user interface
import tkinter as tk

# Import webbrowser so chart links can be opened in the user's browser
import webbrowser

# Import requests to collect live and historical market data from Binance API
import requests

# Import datetime to display live update times in the interface
from datetime import datetime

# Import pandas for table-based data cleaning and feature engineering
import pandas as pd  # type: ignore

# Import Linear Regression as the baseline prediction model
from sklearn.linear_model import LinearRegression  # type: ignore

# Import matplotlib for plotting trend graphs inside the application
import matplotlib.pyplot as plt

# Import TkAgg canvas so matplotlib graphs can be embedded into Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Create a separate Bitcoin page as a child window
class BitcoinWindow(tk.Toplevel):
    def __init__(self, master):
        # Call parent constructor so the Toplevel window is created correctly
        super().__init__(master)

        # Set the title and size of the Bitcoin window
        self.title("Bitcoin")
        self.geometry("1200x800")

        # Display the asset name at the top of the page
        self.lbl_btc = tk.Label(self, text="Bitcoin", font=("Calibri", 16))
        self.lbl_btc.place(x=520, y=120)

        # Add a small chart emoji beside the asset name for visual context
        self.emoji_chart = tk.Label(self, text="📈", font=("Calibri", 16))
        self.emoji_chart.place(x=605, y=120)

        # The original image-based chart area was replaced with a graph frame
        # so live matplotlib graphs can be displayed inside the GUI
        self.graph_frame = tk.Frame(self, width=800, height=450)
        self.graph_frame.place(x=215, y=270)

        # Create a button that opens the live Bitcoin chart in TradingView
        self.btn_link = tk.Button(
            self,
            text="open bitcoin chart 📊",
            font=("Calibri", 16),cursor="hand2",
            command=self.open_link
        )
        self.btn_link.place(x=700, y=120)

        # Create the prediction button that runs the model when clicked
        self.btn_predict = tk.Button(
            self,
            text="Prediction",
            font=("Calibri", 14),cursor="hand2",
            command=self.run_prediction
        )
        self.btn_predict.place(x=215, y=70)

        # Label used to display the predicted next closing price
        self.predicted_lbl = tk.Label(self, text="Predicted next close: —", font=("Calibri", 12))
        self.predicted_lbl.place(x=215, y=100)

        # Label used to display the up / down / flat arrow beside prediction output
        self.arrow_lbl = tk.Label(self, text="", font=("Calibri", 16))
        self.arrow_lbl.place(x=315, y=73)

        # Label used to display RSI value and interpretation
        self.rsi_lbl = tk.Label(self, text="RSI: —", font=("Calibri", 12))
        self.rsi_lbl.place(x=215, y=160)

        # Label used to display trading signal such as BUY / SELL / HOLD
        self.signal_lbl = tk.Label(self, text="Signal: —", font=("Calibri", 12, "bold"))
        self.signal_lbl.place(x=215, y=180)

        # Label used to explain why the system produced the signal
        self.reason_lbl = tk.Label(
            self,
            text="Reason: —",
            font=("Calibri", 11),
            wraplength=500,
            justify="left"
        )
        self.reason_lbl.place(x=215, y=205)

        # Label used to display a simple signal-strength score
        self.signal_strength_lbl = tk.Label(self, text="Signal Strength: —", font=("Calibri", 12))
        self.signal_strength_lbl.place(x=215, y=235)

        # Label used to display the live BTC price
        self.live_btc_price = tk.Label(self, text="live price", font=("Calibri", 16))
        self.live_btc_price.place(x=215, y=120)

        # Label used to display the time of the latest successful update
        self.last_update = tk.Label(self, text="live time", font=("Calibri", 12))
        self.last_update.place(x=215, y=140)

        # Store the previous and current live prices so price movement colour can be shown
        self.last_price = None
        self.price = None

        # Start the repeating live price update loop as soon as the window opens
        self.update_btc_price()

    # Draw a price / trend graph inside the graph frame
    def draw_graph(self, table):
        # Stop graph creation if no processed data table exists
        if table is None:
            raise ValueError("No table data available for graph")

        # Remove any previous graph widgets before drawing a new chart
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        # Create a matplotlib figure and axis sized to fit the GUI
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

        # Plot the real closing price line
        ax.plot(table["date"], table["close"], label="Close Price")

        # Plot the short-term moving average line
        ax.plot(table["date"], table["short trend (MA7)"], label="MA7")

        # Plot the medium-term moving average line
        ax.plot(table["date"], table["medium trend (MA30)"], label="MA30")

        # Add chart title and axis labels for readability
        ax.set_title("Bitcoin Price Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")

        # Add a legend so each line can be identified
        ax.legend()

        # Format date labels so they do not overlap too heavily
        fig.autofmt_xdate()

        # Embed the matplotlib figure into the Tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Close the standalone matplotlib figure to reduce memory use
        plt.close(fig)

    # Run the full prediction pipeline and update all interface outputs
    def run_prediction(self):
        try:
            # Do not run the model unless a live market price is available
            if self.price is None:
                raise ValueError("Live price unavailable")

            # Create the predictor for BTC daily candles using 90 historical entries
            model = BinancePredictor("BTCUSDT", "1d", 90)

            # Run the predictor and store the next predicted close
            prediction = model.run()

            # Draw the processed historical graph after prediction completes
            self.draw_graph(model.table)

            # Read RSI outputs from the model
            rsi_value = model.rsi
            rsi_signal = model.rsi_signal

            # Compare prediction against live market price
            current_price = self.price
            difference = prediction - current_price

            # Generate signal, explanation, and signal strength from the rules
            signal = model.get_trading_signal(current_price)
            reason = model.get_signal_reason(current_price)
            signal_strength = model.get_signal_strength_score(current_price)

            # Choose arrow direction and colour based on whether prediction is above or below live price
            if difference > 0:
                arrow = "⬆️"
                arrow_colour = "green"
            elif difference < 0:
                arrow = "⬇️"
                arrow_colour = "red"
            else:
                arrow = "➡️"
                arrow_colour = "black"

            # Update the main prediction label
            self.predicted_lbl.config(text=f"Predicted next close: ${prediction:,.2f}")

            # Update the arrow label
            self.arrow_lbl.config(text=arrow, fg=arrow_colour)

            # If RSI could not be calculated, show an unavailable message
            if rsi_value is None:
                self.rsi_lbl.config(text=f"RSI: unavailable - {rsi_signal}")
            else:
                # Otherwise show the numerical RSI and its interpretation
                self.rsi_lbl.config(text=f"RSI: {rsi_value:.2f} - {rsi_signal}")

            # Colour the signal label according to the signal strength / direction
            if signal == "BUY" or signal == "STRONG BUY":
                self.signal_lbl.config(text=f"Signal: {signal}", fg="green")
            elif signal == "SELL" or signal == "STRONG SELL":
                self.signal_lbl.config(text=f"Signal: {signal}", fg="red")
            else:
                self.signal_lbl.config(text="Signal: HOLD", fg="black")

            # Show the reason behind the system's decision
            self.reason_lbl.config(text=f"Reason: {reason}")

            # Show the signal-strength percentage
            self.signal_strength_lbl.config(text=f"Signal Strength: {signal_strength}%")

        except Exception as e:
            # Print technical error details to the terminal for debugging
            print("Bitcoin prediction error:", e)

            # Display safe fallback messages in the interface if prediction fails
            self.predicted_lbl.config(text="Prediction unavailable")
            self.arrow_lbl.config(text="!", fg="red")
            self.rsi_lbl.config(text="RSI: unavailable")
            self.signal_lbl.config(text="Signal: unavailable", fg="red")
            self.reason_lbl.config(text="Reason: unavailable")
            self.signal_strength_lbl.config(text="Signal Strength: unavailable")

    # Open the external Bitcoin chart in the browser
    def open_link(self):
        webbrowser.open("https://www.tradingview.com/chart/?symbol=BINANCE%3ABTCUSDT")

    # Repeatedly fetch and display the live BTC price
    def update_btc_price(self):
        try:
            # Request the current BTCUSDT market price from Binance
            response = requests.get(
                "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
                timeout=10
            )

            # Convert the returned text price to float
            self.price = float(response.json()["price"])

            # Default label colour is black unless price direction changes
            colour = "black"

            # Compare current price to previous price to show short-term movement
            if self.last_price is not None:
                if self.price > self.last_price:
                    colour = "green"
                elif self.price < self.last_price:
                    colour = "red"

            # Update the live price label
            self.live_btc_price.config(
                text="BTC Live Price: $" + "{:,.2f}".format(self.price),
                fg=colour
            )

            # Store current price for the next comparison cycle
            self.last_price = self.price

            # Display the time of the latest successful update
            now = datetime.now().strftime("%H:%M:%S")
            self.last_update.config(text=f"Last update : {now}")

        except Exception:
            # If the live price request fails, display an unavailable message
            self.live_btc_price.config(text="BTC Live Price: unavailable", fg="red")

            # Still show the time the failed attempt occurred
            now = datetime.now().strftime("%H:%M:%S")
            self.last_update.config(text=f"Last attempt : {now}")

        # Repeat the live price check every 2.5 seconds
        self.after(2500, self.update_btc_price)


# Create a separate Ethereum page as a child window
class EthereumWindow(tk.Toplevel):
    def __init__(self, master):
        # Call parent constructor so the Toplevel window is created correctly
        super().__init__(master)

        # Set the title and size of the Ethereum window
        self.title("Ethereum")
        self.geometry("1200x800")

        # Display the asset name at the top of the page
        self.lbl_eth = tk.Label(self, text="Ethereum", font=("Calibri", 16))
        self.lbl_eth.place(x=520, y=120)

        # Add a chart emoji for visual consistency with the Bitcoin page
        self.emoji_chart = tk.Label(self, text="📈", font=("Calibri", 16))
        self.emoji_chart.place(x=605, y=120)

        # Replace static image area with a frame used to embed matplotlib graphs
        self.graph_frame = tk.Frame(self, width=800, height=450)
        self.graph_frame.place(x=215, y=270)

        # Create a button that opens the external Ethereum chart
        self.btn_link = tk.Button(
            self,
            text="open ethereum chart 📊",
            font=("Calibri", 16),cursor="hand2",
            command=self.open_eth_link
        )
        self.btn_link.place(x=700, y=120)

        # Label used to display the live ETH price
        self.live_eth_price = tk.Label(self, text="live price", font=("Calibri", 16))
        self.live_eth_price.place(x=215, y=120)

        # Label used to display the latest successful update time
        self.last_update = tk.Label(self, text="live time", font=("Calibri", 12))
        self.last_update.place(x=215, y=140)

        # Store the previous and current live ETH prices
        self.last_price = None
        self.price = None

        # Start the repeating live ETH update loop
        self.update_eth_price()

        # Create the prediction button for Ethereum
        self.btn_predict = tk.Button(
            self,
            text="Prediction",
            font=("Calibri", 14),cursor="hand2",
            command=self.run_prediction
        )
        self.btn_predict.place(x=215, y=70)

        # Label used to display the predicted next closing price
        self.predicted_lbl = tk.Label(self, text="Predicted next close: —", font=("Calibri", 12))
        self.predicted_lbl.place(x=215, y=100)

        # Label used to display prediction direction arrow
        self.arrow_lbl = tk.Label(self, text="", font=("Calibri", 16))
        self.arrow_lbl.place(x=315, y=73)

        # Label used to display RSI information
        self.rsi_lbl = tk.Label(self, text="RSI: —", font=("Calibri", 12))
        self.rsi_lbl.place(x=215, y=160)

        # Label used to display final trading signal
        self.signal_lbl = tk.Label(self, text="Signal: —", font=("Calibri", 12, "bold"))
        self.signal_lbl.place(x=215, y=180)

        # Label used to explain the signal
        self.reason_lbl = tk.Label(
            self,
            text="Reason: —",
            font=("Calibri", 11),
            wraplength=500,
            justify="left"
        )
        self.reason_lbl.place(x=215, y=205)

        # Label used to display signal-strength percentage
        self.signal_strength_lbl = tk.Label(self, text="Signal Strength: —", font=("Calibri", 12))
        self.signal_strength_lbl.place(x=215, y=235)

    # Draw the Ethereum price / trend graph
    def draw_graph(self, table):
        # Stop graph creation if no processed table exists
        if table is None:
            raise ValueError("No table data available for graph")

        # Clear any old chart widgets before drawing a new graph
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        # Create figure and axis for the chart
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

        # Plot closing price and both moving averages
        ax.plot(table["date"], table["close"], label="Close Price")
        ax.plot(table["date"], table["short trend (MA7)"], label="MA7")
        ax.plot(table["date"], table["medium trend (MA30)"], label="MA30")

        # Add chart title and axis labels
        ax.set_title("Ethereum Price Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        fig.autofmt_xdate()

        # Embed the graph into the Tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Close matplotlib figure to reduce memory use
        plt.close(fig)

    # Run the full ETH prediction pipeline and update interface outputs
    def run_prediction(self):
        try:
            # Do not run prediction unless live market price exists
            if self.price is None:
                raise ValueError("Live price unavailable")

            # Create predictor for ETH daily candles using 90 historical entries
            model = BinancePredictor("ETHUSDT", "1d", 90)

            # Run the model and return the next predicted close
            prediction = model.run()

            # Draw the historical chart from the cleaned dataset
            self.draw_graph(model.table)

            # Read RSI results from the model
            rsi_value = model.rsi
            rsi_signal = model.rsi_signal

            # Compare predicted price to current live price
            current_price = self.price
            difference = prediction - current_price

            # Generate decision outputs from the model
            signal = model.get_trading_signal(current_price)
            reason = model.get_signal_reason(current_price)
            signal_strength = model.get_signal_strength_score(current_price)

            # Select arrow direction and colour based on expected price movement
            if difference > 0:
                arrow = "⬆️"
                arrow_colour = "green"
            elif difference < 0:
                arrow = "⬇️"
                arrow_colour = "red"
            else:
                arrow = "➡️"
                arrow_colour = "black"

            # Update prediction label and arrow label
            self.predicted_lbl.config(text=f"Predicted next close: ${prediction:,.2f}")
            self.arrow_lbl.config(text=arrow, fg=arrow_colour)

            # Show RSI value if available, otherwise show unavailable message
            if rsi_value is None:
                self.rsi_lbl.config(text=f"RSI: unavailable - {rsi_signal}")
            else:
                self.rsi_lbl.config(text=f"RSI: {rsi_value:.2f} - {rsi_signal}")

            # Display signal with colour-coded output
            if signal == "BUY" or signal == "STRONG BUY":
                self.signal_lbl.config(text=f"Signal: {signal}", fg="green")
            elif signal == "SELL" or signal == "STRONG SELL":
                self.signal_lbl.config(text=f"Signal: {signal}", fg="red")
            else:
                self.signal_lbl.config(text="Signal: HOLD", fg="black")

            # Display explanation and signal-strength score
            self.reason_lbl.config(text=f"Reason: {reason}")
            self.signal_strength_lbl.config(text=f"Signal Strength: {signal_strength}%")

        except Exception as e:
            # Print technical error details for debugging
            print("Ethereum prediction error:", e)

            # Display safe fallback messages if prediction fails
            self.predicted_lbl.config(text="Prediction unavailable")
            self.arrow_lbl.config(text="!", fg="red")
            self.rsi_lbl.config(text="RSI: unavailable")
            self.signal_lbl.config(text="Signal: unavailable", fg="red")
            self.reason_lbl.config(text="Reason: unavailable")
            self.signal_strength_lbl.config(text="Signal Strength: unavailable")

    # Open the external Ethereum chart in the browser
    def open_eth_link(self):
        webbrowser.open("https://www.tradingview.com/chart/?symbol=BINANCE%3AETHUSDT")

    # Repeatedly fetch and display the live ETH price
    def update_eth_price(self):
        try:
            # Request the current ETHUSDT market price from Binance
            response = requests.get(
                "https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT",
                timeout=10
            )

            # Convert returned text price to float
            self.price = float(response.json()["price"])

            # Default label colour unless price changes direction
            colour = "black"

            # Compare current and previous prices to indicate movement
            if self.last_price is not None:
                if self.price > self.last_price:
                    colour = "green"
                elif self.price < self.last_price:
                    colour = "red"

            # Update live ETH price label
            self.live_eth_price.config(
                text="ETH Live Price: $" + "{:,.2f}".format(self.price),
                fg=colour
            )

            # Store current price for next comparison
            self.last_price = self.price

            # Update time label with successful refresh time
            now = datetime.now().strftime("%H:%M:%S")
            self.last_update.config(text=f"Last update : {now}")

        except Exception:
            # Show unavailable output if API request fails
            self.live_eth_price.config(text="ETH Live Price: unavailable", fg="red")

            # Show time of failed attempt
            now = datetime.now().strftime("%H:%M:%S")
            self.last_update.config(text=f"Last attempt : {now}")

        # Repeat the live ETH update every 2.5 seconds
        self.after(2500, self.update_eth_price)


# Create the main menu window of the application
class MainWindow(tk.Tk):
    def __init__(self):
        # Call parent constructor so the root window is created correctly
        super().__init__()

        # Hide the main menu until the welcome screen has been closed
        self.withdraw()

        # Configure the main menu window appearance
        self.title("Main Menu")
        self.geometry("800x600")
        self.configure(bg="black")

        # Add the main menu instruction text
        self.text = tk.Label(
            self,
            text="Choose a page",
            font=("Calibri", 22, "bold"),
            fg="white",
            bg="black"
        )
        self.text.place(x=300, y=120)

        # Create a button that opens the Ethereum page
        self.open_ethereum = tk.Label(
            self,
            text="Ξ  Open Ethereum",
            font=("Calibri", 16, "bold"),
            bg="#2f2f2f",
            fg="white",
            width=16,
            height=2,
            cursor="hand2"
        )
        self.open_ethereum.place(x=550, y=250, anchor="center")

        self.open_ethereum.bind(
            "<Enter>",
            lambda e: self.open_ethereum.config(bg="#444444")
        )
        self.open_ethereum.bind(
            "<Leave>",
            lambda e: self.open_ethereum.config(bg="#2f2f2f")
        )
        self.open_ethereum.bind(
            "<Button-1>",
            lambda e: self.open_eth()
        )

        # Create a button that opens the Bitcoin page
        self.open_bitcoin = tk.Label(
            self,
            text="₿  Open Bitcoin",
            font=("Calibri", 16, "bold"),
            bg="#2f2f2f",
            fg="white",
            width=16,
            height=2,
            cursor="hand2"
        )
        self.open_bitcoin.place(x=250, y=250, anchor="center")

        self.open_bitcoin.bind(
            "<Enter>",
            lambda e: self.open_bitcoin.config(bg="#444444")
        )
        self.open_bitcoin.bind(
            "<Leave>",
            lambda e: self.open_bitcoin.config(bg="#2f2f2f")
        )
        self.open_bitcoin.bind(
            "<Button-1>",
            lambda e: self.open_btc()
        )

        # Store references to child windows so duplicate windows are not created
        self.btc_OpenOrNot = None
        self.eth_OpenOrNot = None

        # Open the welcome screen immediately when the program starts
        self.welcome_window = WelcomeWindow(self)

    # Open Bitcoin page or bring existing Bitcoin page to the front
    def open_btc(self):
        # Create a new Bitcoin window only if one is not already open
        if self.btc_OpenOrNot is None or not self.btc_OpenOrNot.winfo_exists():
            self.btc_OpenOrNot = BitcoinWindow(self)

            # Override close button behaviour so the stored reference is reset
            self.btc_OpenOrNot.protocol("WM_DELETE_WINDOW", self.close_btc)
        else:
            # If already open, bring it to the front instead of opening another copy
            self.btc_OpenOrNot.lift()

    # Close Bitcoin page and reset stored reference
    def close_btc(self):
        self.btc_OpenOrNot.destroy()
        self.btc_OpenOrNot = None

    # Open Ethereum page or bring existing Ethereum page to the front
    def open_eth(self):
        # Create a new Ethereum window only if one is not already open
        if self.eth_OpenOrNot is None or not self.eth_OpenOrNot.winfo_exists():
            self.eth_OpenOrNot = EthereumWindow(self)

            # Override close button behaviour so the stored reference is reset
            self.eth_OpenOrNot.protocol("WM_DELETE_WINDOW", self.close_eth)
        else:
            # If already open, bring it to the front instead of opening another copy
            self.eth_OpenOrNot.lift()

    # Close Ethereum page and reset stored reference
    def close_eth(self):
        self.eth_OpenOrNot.destroy()
        self.eth_OpenOrNot = None


# Create the first screen shown to the user before the main menu appears
class WelcomeWindow(tk.Toplevel):
    def __init__(self, master):
        # Call parent constructor so the Toplevel window is created correctly
        super().__init__(master)

        # Store reference to the hidden main root window
        self.master = master

        # Configure the welcome window appearance
        self.title("Crypto Market Predictor")
        self.geometry("1000x700")
        self.configure(bg="black")
        self.resizable(False, False)

        # Create a central frame to help organise the welcome screen layout
        self.main_frame = tk.Frame(self, bg="black")
        self.main_frame.place(relx=0.5, rely=0.5, anchor="center")

        # Add the large system title
        self.lbl_title = tk.Label(
            self,
            text="Crypto Market Predictor",
            font=("Calibri", 34, "bold"),
            fg="white",
            bg="black"
        )
        self.lbl_title.place(x=500, y=165, anchor="center")

        # Add a short message explaining the purpose of the tool
        self.lbl_message = tk.Label(
            self,
            text="Welcome to the crypto market analysis and\nprice forecasting application!",
            font=("Calibri", 17),
            fg="white",
            bg="black",
            justify="center"
        )
        self.lbl_message.place(x=500, y=255, anchor="center")

        # If the welcome screen is closed manually, close the full application too
        self.protocol("WM_DELETE_WINDOW", self.close_program)

        # Create the Continue label styled to behave like a button
        self.btn_continue = tk.Label(
            self,
            text="Continue",
            font=("Calibri", 16, "bold"),
            bg="#2f2f2f",
            fg="white",
            width=16,
            height=2,
            cursor="hand2"
        )
        self.btn_continue.place(x=500, y=390, anchor="center")

        # Add hover effect so the Continue label darkens when the mouse enters
        self.btn_continue.bind(
            "<Enter>",
            lambda e: self.btn_continue.config(bg="#444444")
        )

        # Return Continue label to normal colour when the mouse leaves
        self.btn_continue.bind(
            "<Leave>",
            lambda e: self.btn_continue.config(bg="#2f2f2f")
        )

        # Bind mouse click so Continue opens the main menu
        self.btn_continue.bind(
            "<Button-1>",
            lambda e: self.open_main_window()
        )

        # Create the User Policy label styled as a second interactive button
        self.btn_policy = tk.Label(
            self,
            text="User Policy      🔗",
            font=("Calibri", 16, "bold"),
            bg="#2f2f2f",
            fg="white",
            width=20,
            height=2,
            cursor="hand2"
        )
        self.btn_policy.place(x=500, y=475, anchor="center")

        # Add hover effect to the User Policy label
        self.btn_policy.bind(
            "<Enter>",
            lambda e: self.btn_policy.config(bg="#444444")
        )

        # Return User Policy label to normal colour when the mouse leaves
        self.btn_policy.bind(
            "<Leave>",
            lambda e: self.btn_policy.config(bg="#2f2f2f")
        )

        # Bind mouse click so User Policy opens a separate policy window
        self.btn_policy.bind(
            "<Button-1>",
            lambda e: self.open_policy_window()
        )

        # Add a footer disclaimer to remind users the tool is educational only
        self.lbl_footer = tk.Label(
            self,
            text="For educational use only – not financial advice",
            font=("Calibri", 11),
            fg="#bbbbbb",
            bg="black"
        )
        self.lbl_footer.place(relx=0.5, rely=0.93, anchor="center")

    # Close the welcome window and show the main menu
    def open_main_window(self):
        # Release any modal grab if present
        self.grab_release()

        # Destroy the welcome screen
        self.destroy()

        # Reveal the hidden main window
        self.master.deiconify()

    # Open a separate policy window explaining the limitations of the tool
    def open_policy_window(self):
        # Create a child window connected to the welcome screen
        policy_window = tk.Toplevel(self)

        # Configure policy window appearance
        policy_window.title("User Policy")
        policy_window.geometry("600x350")
        policy_window.configure(bg="black")
        policy_window.resizable(False, False)

        # Add policy window title
        lbl_policy_title = tk.Label(
            policy_window,
            text="User Policy",
            font=("Calibri", 22, "bold"),
            fg="white",
            bg="black"
        )
        lbl_policy_title.pack(pady=(20, 15))

        # Create the policy guidance text displayed to the user
        policy_text = (
            "• This tool is for educational purposes only.\n"
            "• Predictions are based on historical data and simple models.\n"
            "• The system does not guarantee future market accuracy.\n"
            "• Users should not rely on this tool as financial advice.\n"
            "• Always carry out your own research before making decisions."
        )

        # Display the policy text in a readable wrapped label
        lbl_policy_text = tk.Label(
            policy_window,
            text=policy_text,
            font=("Calibri", 14),
            fg="white",
            bg="black",
            justify="left",
            wraplength=520
        )
        lbl_policy_text.pack(padx=20, pady=10)

        # Create a close label styled like a button so the policy window can be dismissed
        btn_close_policy = tk.Label(
            policy_window,
            text="Close",
            font=("Calibri", 16, "bold"),
            bg="#2f2f2f",
            fg="white",
            width=18,
            height=2,
            cursor="hand2",
        )
        btn_close_policy.pack(pady=20)

        # Bind click action so the policy window closes
        btn_close_policy.bind(
            "<Button-1>",
            lambda e: policy_window.destroy()
        )

        # Add hover effect to the close control
        btn_close_policy.bind(
            "<Enter>",
            lambda e: btn_close_policy.config(bg="#444444")
        )
        btn_close_policy.bind(
            "<Leave>",
            lambda e: btn_close_policy.config(bg="#2f2f2f")
        )

    # Close the entire application if the welcome screen is closed
    def close_program(self):
        self.master.destroy()


# Create the prediction engine used by both BTC and ETH pages
class BinancePredictor:
    def __init__(self, symbol="BTCUSDT", interval="1d", limit=90):
        # Store API request settings
        self.symbol = symbol
        self.interval = interval
        self.limit = limit

        # Binance endpoint used to collect candle data
        self.url = "https://api.binance.com/api/v3/klines"

        # These attributes are filled later during processing
        self.table = None
        self.model = None
        self.prediction = None
        self.rsi = None
        self.rsi_signal = None

    # Calculate the latest RSI value from the cleaned historical table
    def calculate_rsi(self, period=14):
        # Do not attempt RSI unless enough data exists
        if self.table is None or len(self.table) < period + 1:
            return None

        # Calculate one-step price changes between rows
        delta = self.table["close"].diff()

        # Keep positive changes as gains and negative changes as losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Compute rolling average gain and rolling average loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # If latest average loss is missing, RSI cannot be calculated yet
        if pd.isna(avg_loss.iloc[-1]):
            return None

        # If loss is zero, RSI is maximised at 100
        if avg_loss.iloc[-1] == 0:
            return 100.0

        # Compute relative strength and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # If latest RSI is missing, return None safely
        if pd.isna(rsi.iloc[-1]):
            return None

        # Return the latest RSI as a float
        return float(rsi.iloc[-1])

    # Convert numerical RSI into an interpretable message
    def interpret_rsi(self):
        # If no RSI exists, report that clearly
        if self.rsi is None:
            return "No RSI data"
        elif self.rsi < 30:
            return "Oversold (Buy Signal)"
        elif self.rsi > 70:
            return "Overbought (Sell Signal)"
        else:
            return "Neutral (Hold)"

    # Combine predicted price and RSI into a final signal
    def get_trading_signal(self, current_price):
        # If prediction failed, default to HOLD
        if self.prediction is None:
            return "HOLD"

        # Fallback logic if RSI is unavailable
        if self.rsi is None:
            if self.prediction > current_price:
                return "BUY"
            elif self.prediction < current_price:
                return "SELL"
            else:
                return "HOLD"

        # Stronger / weaker signal logic when prediction is above live price
        if self.prediction > current_price:
            if self.rsi < 30:
                return "STRONG BUY"
            elif self.rsi <= 70:
                return "BUY"
            else:
                return "HOLD"

        # Stronger / weaker signal logic when prediction is below live price
        elif self.prediction < current_price:
            if self.rsi > 70:
                return "STRONG SELL"
            elif self.rsi >= 30:
                return "SELL"
            else:
                return "HOLD"

        # If prediction equals live price, do not act
        return "HOLD"

    # Explain in words why a signal was produced
    def get_signal_reason(self, current_price):
        # If prediction failed, return an unavailable message
        if self.prediction is None:
            return "No prediction available"

        # Simpler explanation if RSI is unavailable
        if self.rsi is None:
            if self.prediction > current_price:
                return "Predicted price is above current price"
            elif self.prediction < current_price:
                return "Predicted price is below current price"
            return "Predicted price equals current price"

        # Provide combined price + RSI explanation
        if self.prediction > current_price and self.rsi < 30:
            return "Predicted price is above current price and RSI suggests the asset may be oversold"
        elif self.prediction > current_price and self.rsi <= 70:
            return "Predicted price is above current price and RSI remains within a normal range"
        elif self.prediction < current_price and self.rsi > 70:
            return "Predicted price is below current price and RSI suggests the asset may be overbought"
        elif self.prediction < current_price and self.rsi >= 30:
            return "Predicted price is below current price and RSI remains within a normal range"
        else:
            return "Prediction and RSI together suggest holding position"

    # Convert signal strength into a simple signal-strength score
    def get_signal_strength_score(self, current_price):
        # First get the signal for the current market situation
        signal = self.get_trading_signal(current_price)

        # Strong signals receive the highest rule-based strength score
        if signal == "STRONG BUY" or signal == "STRONG SELL":
            return 80
        elif signal == "BUY" or signal == "SELL":
            return 65
        else:
            return 50

    # Run the full prediction engine from API fetch to next-close forecast
    def run(self):
        # Prepare Binance API request parameters
        params = {"symbol": self.symbol, "interval": self.interval, "limit": self.limit}

        # Request historical candle data from Binance
        response = requests.get(self.url, params=params, timeout=10)
        raw_data = response.json()

        # Validate that the returned data is a non-empty list
        if not isinstance(raw_data, list) or len(raw_data) == 0:
            raise ValueError("No historical data returned")

        # Define the raw Binance kline column names
        columns = [
            "Open time", "Open price", "High price", "Low price", "Close price",
            "Volume (Base)", "Close time", "Quote asset volume", "Number of trades",
            "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore (always 0)"
        ]

        # Convert raw JSON list into a DataFrame
        table = pd.DataFrame(raw_data, columns=columns)

        # Create cleaned columns needed for modelling
        table["date"] = pd.to_datetime(table["Open time"], unit="ms")
        table["close"] = table["Close price"].astype(float)
        table["volume"] = table["Volume (Base)"].astype(float)

        # Keep only the required fields
        table = table[["date", "close", "volume"]]

        # Create short-term and medium-term moving average features
        table["short trend (MA7)"] = table["close"].rolling(7).mean()
        table["medium trend (MA30)"] = table["close"].rolling(30).mean()

        # Remove rows with missing moving-average values and reset index
        table = table.dropna().reset_index(drop=True)

        # Stop if the cleaned dataset is too small for next-step prediction
        if len(table) < 2:
            raise ValueError("Not enough cleaned historical data for prediction")

        # Store cleaned table for later graphing and RSI calculation
        self.table = table

        # Calculate RSI and convert it into a readable interpretation
        self.rsi = self.calculate_rsi()
        self.rsi_signal = self.interpret_rsi()

        # Build feature matrix from current row values except the final row
        array_all = table[["close", "short trend (MA7)", "medium trend (MA30)", "volume"]].iloc[:-1].values

        # Build target array from next-day close values
        array_close = table["close"].shift(-1).dropna().values

        # Train the Linear Regression model
        self.model = LinearRegression().fit(array_all, array_close)

        # Select the most recent feature row for next-step prediction
        array_all_2 = table[["close", "short trend (MA7)", "medium trend (MA30)", "volume"]].iloc[[-1]].values

        # Generate and store the final prediction
        self.prediction = float(self.model.predict(array_all_2)[0])

        # Return the predicted next close
        return self.prediction

# Create the main application window and start the Tkinter event loop
MainWindow().mainloop()