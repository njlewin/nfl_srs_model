import pandas as pd
from config import *
from datetime import datetime
past = pd.read_csv(PAST_FILE)
future = pd.read_csv(FUTURE_FILE)
df = pd.DataFrame({
    'last_updated': [datetime.now()],
    'week': [max([week for week in past['week'].unique() if week not in future['week'].unique()])]
})
df.to_csv(LAST_UPDATED_FILE, index = False)
