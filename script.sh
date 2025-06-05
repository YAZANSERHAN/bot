#!/bin/bash

set -e

echo "[✓] Updating system..."
sudo apt update && sudo apt upgrade -y

echo "[✓] Installing essential packages..."
sudo apt install -y \
    git curl wget build-essential \
    python3 python3-pip python3-venv \
    redis-server postgresql postgresql-contrib \
    libpq-dev

echo "[✓] Enabling Redis and PostgreSQL..."
sudo systemctl enable --now redis postgresql

echo "[✓] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[✓] Upgrading pip..."
pip install --upgrade pip

echo "[✓] Installing Python libraries..."
pip install pandas numpy scikit-learn xgboost lightgbm \
    python-telegram-bot aiohttp psycopg2-binary redis \
    yfinance apscheduler cryptography

echo "[✓] Setup complete!"
echo
echo "📌 Next steps:"
echo "1. Place your cryptobot.py file here."
echo "2. Create a .env file with your keys:"
echo
echo "Example:"
cat <<EOF
TELEGRAM_BOT_TOKEN=your_token_here
DATABASE_URL=postgresql://user:pass@localhost:5432/yourdb
REDIS_URL=redis://localhost:6379
ENCRYPTION_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
BINANCE_API_KEY=your_binance_key
BINANCE_API_SECRET=your_binance_secret
EOF
echo
echo "3. Run: source venv/bin/activate && python cryptobot.py"

