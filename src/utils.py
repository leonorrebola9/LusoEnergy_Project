"""
utils.py - funcoes partilhadas pelos notebooks do projeto LusoEnergia.

    from utils import carregar_dados, tratar_missing, dias_validos

Contem apenas o que e reutilizado em varias partes: carregamento e validacao dos
dados, grelha temporal ao minuto, tratamento de valores em falta e selecao de dias
validos. O que e especifico de uma parte fica no respetivo notebook.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Caminho por defeito: <raiz do projeto>/data/consumption.csv (funciona de qualquer notebook)
CSV_PADRAO = Path(__file__).resolve().parent.parent / "data" / "consumption.csv"

# --------------------------------------------------------------------------
# Dicionario de dados (escala de Stevens, unidades, dtypes)
# --------------------------------------------------------------------------
DICIONARIO_DADOS = {
    "Global_active_power": {
        "dtype": "float64", "unidade": "kW", "escala": "razao",
        "nota": "Potencia ativa media do minuto. Nao pode ser negativa.",
    },
    "Global_reactive_power": {
        "dtype": "float64", "unidade": "kVAr", "escala": "razao",
        "nota": "Potencia reativa media do minuto. Nao pode ser negativa.",
    },
    "Voltage": {
        "dtype": "float64", "unidade": "V", "escala": "razao",
        "nota": "Tensao media. Rede de ~240 V: valores muito fora de 200-260 V seriam suspeitos.",
    },
    "Global_intensity": {
        "dtype": "float64", "unidade": "A", "escala": "razao",
        "nota": "Intensidade de corrente media do minuto.",
    },
    "Sub_metering_1": {
        "dtype": "float64", "unidade": "Wh", "escala": "razao",
        "nota": "Cozinha (forno, micro-ondas, ...). ATENCAO: Wh por minuto, nao kW.",
    },
    "Sub_metering_2": {
        "dtype": "float64", "unidade": "Wh", "escala": "razao",
        "nota": "Lavandaria (maquina de lavar, secador, frigorifico, luz).",
    },
    "Sub_metering_3": {
        "dtype": "float64", "unidade": "Wh", "escala": "razao",
        "nota": "Aquecedor de agua e ar condicionado.",
    },
}
COLUNAS = list(DICIONARIO_DADOS)
DTYPES = {c: d["dtype"] for c, d in DICIONARIO_DADOS.items()}


# --------------------------------------------------------------------------
# Carregamento e validacao
# --------------------------------------------------------------------------
def carregar_dados(caminho=CSV_PADRAO, validar=True):
    """Le o CSV da janela do grupo e devolve um DataFrame tipado.

    O ficheiro tem uma coluna 'datetime' (gerada por preparar_dataset.py) e as
    7 variaveis numericas; valores em falta vem vazios (ou '?', tratado como NaN).

    Devolve um DataFrame com DatetimeIndex (unico, ordenado) e 7 colunas float64.
    """
    df = pd.read_csv(
        caminho, sep=";", na_values=["?"], dtype=DTYPES,
        index_col="datetime", parse_dates=["datetime"],
    )
    df = df[COLUNAS].sort_index()
    if validar:
        validar_dados(df)
    return df


def validar_dados(df):
    """Verificacoes de integridade (falham com AssertionError se algo estiver mal)."""
    # 1. indice temporal do tipo certo
    assert isinstance(df.index, pd.DatetimeIndex), "o indice tem de ser DatetimeIndex"
    # 2. indice unico (sem leituras duplicadas)
    assert df.index.is_unique, f"{df.index.duplicated().sum()} timestamps duplicados"
    # 3. indice ordenado
    assert df.index.is_monotonic_increasing, "o indice nao esta ordenado"
    # 4. colunas e tipos esperados
    assert list(df.columns) == COLUNAS, "colunas inesperadas"
    assert all(df[c].dtype == DTYPES[c] for c in COLUNAS), "dtype inesperado"
    # 5. potencias, intensidade e submedicoes nao negativas
    nao_neg = [c for c in COLUNAS if c != "Voltage"]
    assert (df[nao_neg].min() >= 0).all(), f"valores negativos em {list(df[nao_neg].min()[df[nao_neg].min() < 0].index)}"
    # 6. tensao fisicamente plausivel
    assert df["Voltage"].dropna().between(180, 280).all(), "tensao fora de 180-280 V"
    # 7. nenhuma sentinela '?' sobrou (as colunas ja sao numericas)
    assert all(pd.api.types.is_float_dtype(df[c]) for c in COLUNAS), "ha colunas nao numericas"
    # 8. leituras ao minuto: nenhum passo de tempo inferior a 1 min
    assert (df.index.to_series().diff().dropna() >= pd.Timedelta("1min")).all(), "passo < 1 min"
    # 9. as 7 colunas tipicamente falham em conjunto -> deteta mascaras incoerentes
    assert df.isna().all(axis=1).sum() == (df.isna().sum(axis=1) > 0).sum(), (
        "ha linhas com missing parcial: rever o pressuposto 'falha total do equipamento'"
    )
    return True


# --------------------------------------------------------------------------
# Grelha temporal, lacunas e tratamento de missing
# --------------------------------------------------------------------------
def reindexar_minuto(df):
    """Garante grelha completa ao minuto: minutos sem linha passam a NaN."""
    grelha = pd.date_range(df.index.min(), df.index.max(), freq="min")
    return df.reindex(grelha).rename_axis("datetime")


def tabela_lacunas(df, coluna="Global_active_power"):
    """Uma linha por lacuna (sequencia consecutiva de minutos em falta)."""
    df = reindexar_minuto(df)
    em_falta = df[coluna].isna()
    grupo = (em_falta != em_falta.shift()).cumsum()[em_falta]
    idx = em_falta[em_falta].index.to_series()
    t = idx.groupby(grupo).agg(inicio="first", fim="last", minutos="size")
    t["duracao_horas"] = (t["minutos"] / 60).round(2)
    return t.reset_index(drop=True).sort_values("inicio")


def missing_por_dia(df, coluna="Global_active_power"):
    """Fracao de minutos em falta por dia (0-1)."""
    return reindexar_minuto(df)[coluna].isna().groupby(lambda t: t.normalize()).mean()


def tratar_missing(df, limite_interp_min=180):
    """Estrategia de tratamento, DIFERENTE conforme a duracao da lacuna.

    * Lacunas <= limite_interp_min (3 h): interpolacao linear no tempo.
      A simulacao feita na Parte I (notebook 01) mostra erro absoluto medio de ~0,06 kW
      para 1 min e ~0,47 kW para 3 h, abaixo do erro de usar a media (~0,75 kW)
      ou o perfil hora x dia da semana (~0,57 kW). Em 3 h ainda e util, mas o
      erro ja e relevante: as imputadas ficam marcadas em `imputado`.
    * Lacunas > limite_interp_min (ex.: 15 h, 55 h): ficam NaN. A partir de
      ~12 h a interpolacao e pior do que o perfil sazonal e, a 2 dias, nenhum
      metodo bate ~0,6 kW de erro (o desvio-padrao do consumo): imputar seria
      inventar dados. Os dias afetados sao excluidos das agregacoes diarias
      (`dias_validos`).

    Devolve (df_tratado, serie_booleana_imputado).
    """
    df = reindexar_minuto(df)
    original_nan = df.isna().any(axis=1)
    tratado = df.interpolate(method="time", limit_area="inside")
    # interpolate preenche TUDO (inclui lacunas longas): reverter as longas
    t = tabela_lacunas(df)
    for _, r in t[t["minutos"] > limite_interp_min].iterrows():
        tratado.loc[r["inicio"]:r["fim"]] = np.nan
    imputado = original_nan & ~tratado.isna().any(axis=1)
    return tratado, imputado.rename("imputado")


def dias_validos(df, max_frac_missing=0.1, coluna="Global_active_power"):
    """Datas cuja fracao de missing e <= max_frac_missing (para agregacoes diarias)."""
    f = missing_por_dia(df, coluna)
    return f[f <= max_frac_missing].index