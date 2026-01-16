# Strategy Research Engine

A fully automated, Render-deployable system for generating, backtesting, and filtering rule-based trading strategies with Telegram notifications.

## 🚀 Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your credentials

# Run single cycle
python main.py
```

### Deploy to Render

1. Push to GitHub
2. Create new **Cron Job** on Render
3. Set schedule: `* * * * *` (every minute)
4. Add environment variables in dashboard
5. Deploy!

## 📁 Project Structure

```
strategy_research_engine/
├── main.py                 # Entry point (cron-safe)
├── config.py               # Configuration management
├── models.py               # Data models
├── strategy_generator.py   # Generates rule-based strategies
├── strategy_validator.py   # Validates strategy logic
├── backtest_engine.py      # Runs backtests on OHLCV
├── consistency_filter.py   # Filters by consistency rules
├── strategy_repository.py  # Stores passing strategies
├── telegram_notifier.py    # Sends alerts
├── indicators.py           # Non-repainting indicators
├── data_provider.py        # OHLCV data provider
├── render.yaml             # Render deployment config
├── requirements.txt        # Dependencies
└── .env.example            # Environment template
```

## ⚙️ Environment Variables

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | From @BotFather |
| `TELEGRAM_CHAT_ID` | Your chat/group ID |
| `STRATEGIES_PER_CYCLE` | Strategies per run (default: 5) |
| `STORAGE_PATH` | JSON storage path (default: strategies.json) |

## 📊 How It Works

1. **Generate** → Creates N rule-based strategies using Price Action, VWAP, EMA, RSI
2. **Validate** → Checks for executability, no look-ahead bias
3. **Backtest** → Tests on 30/60/180/365 days with OHLCV data
4. **Filter** → Passes only if ALL periods profitable, drawdown < 25%
5. **Store** → Saves winning strategies to JSON
6. **Notify** → Sends Telegram alert with full metrics

## 🎯 Strategy Rules

- Entry: Price Action (70%) + VWAP + EMA (max 2) + RSI (optional)
- Risk-Reward: 1:1.5 to 1:3
- Intraday only
- No ML, no sentiment, no news

## ✅ Consistency Filter

Strategy passes ONLY if:
- All 4 periods are profitable
- Max drawdown < 25%
- Expectancy > 0
- No single trade > 30% of total profit

## 📱 Telegram Alert Format

```
🎯 NEW STRATEGY PASSED

📊 Market: NSE:NIFTY50
⏱ Timeframe: 5m
📈 Trades/Year: 156

📉 Win Rate: 58.3%
📉 Max Drawdown: 18.2%
📈 Profit Factor: 1.85

📊 Performance:
• 30D: +2.1%
• 60D: +4.8%
• 180D: +12.3%
• 365D: +28.7%
```

## 🚀 Render Deployment (Free Tier Hack)

Since Render Cron Jobs are paid, we use a **Web Service** with a keep-alive loop.

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Switch to Free Tier Web Service"
git push
```

### Step 2: Create Web Service on Render
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **New → Web Service** (NOT Cron Job)
3. Connect your repo
4. Settings:
   - **Name**: strategy-engine
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn main:app`
   - **Plan**: Free

### Step 3: Prevent Sleep (Crucial!)
Render Free Tier spins down after 15 minutes of inactivity. To keep it running 24/7:
1. Copy your Render URL (e.g., `https://strategy-engine.onrender.com`)
2. Go to [UptimeRobot](https://uptimerobot.com) (Free)
3. Create a new **HTTP Monitor**
4. Paste your Render URL
5. Set interval to **5 minutes**

This ping keeps the background loop running forever for free.

### Step 4: Environment Variables
Add these in Render Dashboard:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `STRATEGIES_PER_CYCLE` = 5
