import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 120
sns.set_theme(style="whitegrid")

csv_filename = 'data.csv'

df = pd.read_csv(csv_filename, engine='python', on_bad_lines='skip')

df.columns = df.columns.str.strip().str.lower()

if len(df.columns) == 1:
    df = df.iloc[:,0].str.replace('"','').str.split(',', expand=True)
    df.columns = ['timestamp','open','high','low','close','volume_btc','volume_usd']

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

for col in ['open','high','low','close','volume_btc','volume_usd']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

if df.empty:
    raise Exception("File is still corrupted. Re-download CSV properly.")

df = df.sort_values('timestamp')

df['date'] = df['timestamp'].dt.date

df['candle_range'] = (df['high'] - df['low']) / df['close']

daily = df.groupby('date').agg(
    open_price=('open','first'),
    close_price=('close','last'),
    total_vol=('volume_btc','sum'),
    avg_range=('candle_range','mean')
).reset_index()

daily['date'] = pd.to_datetime(daily['date'])
daily['return'] = (daily['close_price'] - daily['open_price']) / daily['open_price']

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-9)

daily['score'] = (
    normalize(daily['return'])*0.5 +
    normalize(daily['total_vol'])*0.3 +
    normalize(1/(daily['avg_range']+1e-9))*0.2
)

threshold = daily['score'].median()
daily['sentiment'] = daily['score'].apply(lambda x: 'Greed' if x >= threshold else 'Fear')

np.random.seed(42)
trades = []

for _, row in daily.iterrows():
    sentiment = row['sentiment']
    ret = row['return']
    n = np.random.randint(50,80)

    for _ in range(n):
        leverage = np.random.choice([2,3,5,10,15,20])
        position = np.random.uniform(500,20000)
        direction = np.random.choice(['Long','Short'])

        edge = ret if direction=='Long' else -ret
        noise = np.random.normal(0,0.01)

        pnl = (edge + noise) * leverage * position
        pnl = np.clip(pnl, -position*leverage*0.15, position*leverage*0.15)

        trades.append({
            'date': row['date'],
            'pnl': pnl,
            'leverage': leverage,
            'position': position,
            'sentiment': sentiment
        })

trades = pd.DataFrame(trades)

merged = pd.merge(trades, daily, on='date', how='inner')

fear = merged[merged['sentiment_x']=='Fear']['pnl']
greed = merged[merged['sentiment_x']=='Greed']['pnl']

t,p = stats.ttest_ind(fear, greed, equal_var=False)

print("T-test:", t, p)
print("\nPnL Mean:")
print(merged.groupby('sentiment_x')['pnl'].mean())

plt.figure(figsize=(8,4))
sns.boxplot(data=merged, x='sentiment_x', y='pnl')
plt.axhline(0)
plt.show()

print("DONE")