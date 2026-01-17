import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import io
import os
from typing import List, Optional
from models import Trade
from data_provider import OHLCVData

class ChartGenerator:
    """Generates performance charts for strategy reports."""
    
    @staticmethod
    def generate_performance_chart(data: OHLCVData, trades: List[Trade], 
                                 title: str = "Strategy Performance",
                                 filename: str = "chart.png") -> Optional[str]:
        """
        Generates a price chart with trade markers and saves it to a file.
        
        Args:
            data: OHLCVData object containing price data.
            trades: List of Trade objects to overlay.
            title: Chart title.
            filename: Output filename.
            
        Returns:
            Absolute path to the generated image file, or None if failed.
        """
        try:
            # Create DataFrame for easier plotting
            df = pd.DataFrame({
                'timestamp': data.timestamps,
                'close': data.close,
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Setup Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot Price
            ax.plot(df.index, df['close'], label='Price', color='black', linewidth=1, alpha=0.7)
            
            # Plot Trades
            long_entries = []
            long_exits = []
            short_entries = []
            short_exits = []
            
            for trade in trades:
                try:
                    # Parse timestamps
                    entry_ts = pd.to_datetime(trade.entry_time)
                    exit_ts = pd.to_datetime(trade.exit_time)
                    
                    if trade.direction == "LONG":
                        long_entries.append((entry_ts, trade.entry_price))
                        long_exits.append((exit_ts, trade.exit_price))
                        # Draw line connecting entry and exit
                        ax.plot([entry_ts, exit_ts], [trade.entry_price, trade.exit_price], 
                                color='green', linestyle='--', linewidth=0.8, alpha=0.5)
                    else:
                        short_entries.append((entry_ts, trade.entry_price))
                        short_exits.append((exit_ts, trade.exit_price))
                        # Draw line connecting entry and exit
                        ax.plot([entry_ts, exit_ts], [trade.entry_price, trade.exit_price], 
                                color='red', linestyle='--', linewidth=0.8, alpha=0.5)

                except Exception as e:
                    print(f"Skipping trade plot due to date error: {e}")
                    continue

            # Scatter Plot Markers
            if long_entries:
                le_x, le_y = zip(*long_entries)
                ax.scatter(le_x, le_y, marker='^', color='green', s=100, label='Long Entry', zorder=5)
            
            if long_exits:
                lx_x, lx_y = zip(*long_exits)
                ax.scatter(lx_x, lx_y, marker='v', color='black', s=50, label='Long Exit', zorder=5)

            if short_entries:
                se_x, se_y = zip(*short_entries)
                ax.scatter(se_x, se_y, marker='v', color='red', s=100, label='Short Entry', zorder=5)

            if short_exits:
                sx_x, sx_y = zip(*short_exits)
                ax.scatter(sx_x, sx_y, marker='^', color='black', s=50, label='Short Exit', zorder=5)

            # Formatting
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend()
            
            # Format Date Axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save
            plt.savefig(filename, dpi=100)
            plt.close(fig)
            
            return os.path.abspath(filename)
            
        except Exception as e:
            print(f"Chart Generation Error: {e}")
            return None
