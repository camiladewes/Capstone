import pandas as pd
from sqlalchemy import create_engine

# Substitua com suas credenciais
DB_URL = "postgresql://postgres:WkSEqzfJFvLzpPjlhpDToolnZdJMFFwV@yamabiko.proxy.rlwy.net:50743/railway"

# Conectar ao banco
engine = create_engine(DB_URL)

# Exportar tabela
df = pd.read_sql_table('price_predictions', engine)
df.to_csv('price_predictions.csv', index=False)
print("Dados exportados para price_predictions.csv!")