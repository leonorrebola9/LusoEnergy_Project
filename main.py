import numpy as np
import pandas as pd

SOMA_DOS_NUMEROS_DE_ALUNO = 107335

rng = np.random.default_rng(SOMA_DOS_NUMEROS_DE_ALUNO) 
inicio = rng.choice(pd.date_range('2007-01-01', '2009-05-01', freq='MS')) 
fim = inicio + pd.DateOffset(months=18)

import pandas as pd

def carregar_dados(caminho, janela=None):
    """
    Lê o ficheiro bruto da LusoEnergia e devolve um DataFrame com
    índice temporal (DatetimeIndex) e dtypes justificados.

    Parâmetros
    ----------
    caminho : str
        Caminho para household_power_consumption.txt (separador ';').
    janela : tuple(str, str) ou None
        (início, fim) — janela de 18 meses do grupo. Se None, devolve tudo.
    """
    colunas_numericas = [
        'Global_active_power', 'Global_reactive_power', 'Voltage',
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]

    # na_values='?' é resolvido pelo parser ANTES do cast para float32,
    # por isso dtype explícito e sentinela funcionam no mesmo passo.
    df = pd.read_csv(
        caminho, sep=';', na_values='?',
        dtype={c: 'float32' for c in colunas_numericas},
    )

    # Date+Time -> DatetimeIndex, vetorizado (sem apply linha a linha)
    df.index = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df = df.drop(columns=['Date', 'Time']).sort_index()
    df.index.name = 'Datetime'

    if janela is not None:
        inicio, fim = janela
        df = df.loc[(df.index >= inicio) & (df.index < fim)]

    # --- asserts de validação ---
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_unique, "timestamps duplicados"
    assert df.index.is_monotonic_increasing, "índice não ordenado"
    assert list(df.columns) == colunas_numericas
    potencias = ['Global_active_power', 'Global_reactive_power', 'Global_intensity',
                 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    assert (df[potencias].dropna() >= 0).all().all(), "potência/intensidade negativa"
    assert (df['Voltage'].dropna() > 0).all(), "tensão <= 0"

    return df

# uso:
df_grupo = carregar_dados(
    'household_power_consumption.txt',
    janela=('2008-05-01', '2009-11-01')
)