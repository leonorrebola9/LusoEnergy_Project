
import pandas as pd
import numpy as np


'''Nossa janela é 1 de maio de 2008 a 31 de outubro de 2009
'''

# ==================================================
# 1. CONFIGURAÇÕES
# ==================================================

SEED_GRUPO = 107335

FICHEIRO_ORIGINAL = "household_power_consumption.txt"
FICHEIRO_FINAL = "consumption.csv"

TAMANHO_BLOCO = 200_000


# ==================================================
# 2. CALCULAR A JANELA DE 18 MESES
# ==================================================

rng = np.random.default_rng(SEED_GRUPO)

inicio = pd.Timestamp(
    rng.choice(
        pd.date_range(
            start='2007-01-01',
            end='2009-05-01',
            freq='MS'
        )
    )
)

fim = inicio + pd.DateOffset(months=18)

print("=" * 50)
print("JANELA TEMPORAL DO GRUPO")
print("=" * 50)

print(f"Seed: {SEED_GRUPO}")
print(f"Início: {inicio.date()}")
print(f"Fim: {fim.date()}")
print("=" * 50)


# ==================================================
# 3. LER E FILTRAR O DATASET POR BLOCOS
# ==================================================

partes = []

for chunk in pd.read_csv(
    FICHEIRO_ORIGINAL,
    sep=';',
    na_values=['?'],
    chunksize=TAMANHO_BLOCO,
    low_memory=False
):

    # Converter data e hora numa única coluna
    chunk['datetime'] = pd.to_datetime(
        chunk['Date'] + ' ' + chunk['Time'],
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce'
    )

    # Selecionar apenas os registos da janela
    chunk = chunk[
        (chunk['datetime'] >= inicio) &
        (chunk['datetime'] < fim)
    ]

    # Guardar apenas os blocos com dados
    if not chunk.empty:
        partes.append(chunk)

    print(f"Bloco processado: {len(chunk)} registos selecionados")


# ==================================================
# 4. JUNTAR OS BLOCOS
# ==================================================

df = pd.concat(partes, ignore_index=True)

# Remover as colunas originais de data e hora
df = df.drop(columns=['Date', 'Time'])

# Criar índice temporal
df = df.set_index('datetime').sort_index()


# ==================================================
# 5. GUARDAR O DATASET REDUZIDO EM CSV
# ==================================================

df.to_csv(
    FICHEIRO_FINAL,
    sep=';',
    decimal='.',
    encoding='utf-8',
    index=True,
    index_label='datetime'
)


# ==================================================
# 6. VERIFICAR RESULTADOS
# ==================================================

print("\n" + "=" * 50)
print("DATASET REDUZIDO COM SUCESSO")
print("=" * 50)

print(f"Ficheiro: {FICHEIRO_FINAL}")
print(f"Total de registos: {len(df):,}")
print(f"Total de variáveis: {len(df.columns)}")
print(f"Data inicial: {df.index.min()}")
print(f"Data final: {df.index.max()}")

print("\nPrimeiros registos:")
print(df.head())

print("\nValores em falta por variável:")
print(df.isna().sum())

print("=" * 50)
