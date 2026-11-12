"""
Checkpoint 1 - Parte I (1.1 e 1.2): carregamento, tipagem e auditoria de qualidade.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1.1  Dicionario de tipos de dados (escala de Stevens) por variavel
# --------------------------------------------------------------------------
# Todas as variaveis sao medidas fisicas com zero absoluto -> escala de RAZAO.
# Ficam em float64: precisam de representar NaN (um int64 nao o permite) e
# 790k x 7 x 8 B ~ 44 MB, portanto float32 nao compensa a perda de precisao.
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


def carregar_dados(caminho="consumption.csv", validar=True):
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
# 1.2  Auditoria de qualidade
# --------------------------------------------------------------------------
def reindexar_minuto(df):
    """Garante grelha completa ao minuto: minutos sem linha passam a NaN."""
    grelha = pd.date_range(df.index.min(), df.index.max(), freq="min")
    return df.reindex(grelha).rename_axis("datetime")


def resumo_missing(df):
    """Missing por variavel (contagem e %)."""
    n = df.isna().sum()
    return pd.DataFrame({"n_missing": n, "pct": (100 * n / len(df)).round(3)})


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


def dias_sem_leituras(df, coluna="Global_active_power", limiar=0.5):
    """Dias em que mais de `limiar` dos minutos faltam."""
    f = missing_por_dia(df, coluna)
    return f[f > limiar]


def calendario_missing(df, coluna="Global_active_power"):
    """Matriz dia x hora com a fracao de minutos em falta (para o heatmap)."""
    s = reindexar_minuto(df)[coluna].isna()
    m = s.groupby([s.index.normalize(), s.index.hour]).mean().unstack()
    m.index.name, m.columns.name = "dia", "hora"
    return m


def plot_missing(df, caminho_png=None, coluna="Global_active_power"):
    """Figura de relatorio sobre dados em falta (2 paineis, mesma escala temporal):
    (cima) % de minutos em falta por mes; (baixo) calendario mes x dia do mes com a
    fracao de cada dia em falta. Escala sequencial de um so tom (azul)."""
    import calendar

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    TINTA, TINTA_2, GRELHA = "#2b2b28", "#6b6a64", "#e4e3de"
    AZUL = "#2a78d6"
    cmap = LinearSegmentedColormap.from_list("azul", ["#e9f1fc", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])

    f = missing_por_dia(df, coluna)                      # fracao por dia
    mensal = missing_por_periodo(df)[coluna]             # % por mes
    meses = mensal.index
    MESES_PT = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
    rotulos = [f"{MESES_PT[m.month - 1]}/{m:%y}" for m in meses]

    # matriz mes x dia-do-mes (NaN onde o dia nao existe)
    mat = np.full((len(meses), 31), np.nan)
    for i, m in enumerate(meses):
        for d in range(1, calendar.monthrange(m.year, m.month)[1] + 1):
            mat[i, d - 1] = f.get(pd.Timestamp(m.year, m.month, d), np.nan)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 8.2), gridspec_kw={"height_ratios": [1, 1.9], "hspace": 0.38})
    fig.patch.set_facecolor("white")

    # --- painel 1: % em falta por mes ------------------------------------
    x = np.arange(len(meses))
    ax1.bar(x, mensal.values, width=0.55, color=AZUL)
    ax1.set_xticks(x, rotulos, rotation=45, ha="right", fontsize=9, color=TINTA_2)
    ax1.set_ylabel("Minutos em falta (% do mes)", fontsize=9, color=TINTA_2)
    ax1.grid(axis="y", color=GRELHA, lw=0.8)
    ax1.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax1.spines[sp].set_visible(False)
    ax1.spines["bottom"].set_color(GRELHA)
    ax1.tick_params(axis="y", labelsize=9, colors=TINTA_2, length=0)
    ax1.tick_params(axis="x", length=0)
    for i in np.argsort(mensal.values)[-2:]:
        ax1.annotate(f"{mensal.values[i]:.2f}%", (x[i], mensal.values[i]), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=9, color=TINTA)
    ax1.set_title("COncentração de minutos em falta por mês",
                  loc="left", fontsize=12, color=TINTA, fontweight="bold")

    # --- painel 2: calendario mes x dia ----------------------------------
    im = ax2.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0, vmax=1, aspect="auto",
                    interpolation="nearest")
    ax2.set_xticks(range(0, 31, 2), [str(d) for d in range(1, 32, 2)], fontsize=8.5, color=TINTA_2)
    ax2.set_yticks(range(len(meses)), rotulos, fontsize=9, color=TINTA_2)
    ax2.set_xlabel("Dia do mes", fontsize=9, color=TINTA_2)
    ax2.set_xticks(np.arange(-0.5, 31, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(meses), 1), minor=True)
    ax2.grid(which="minor", color="white", lw=2)           # 2px de folga entre celulas
    ax2.tick_params(which="both", length=0)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    for i in range(len(meses)):
        for j in range(31):
            v = mat[i, j]
            if v > 0.3:
                ax2.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=6.5,
                         color="white" if v > 0.5 else TINTA)
    ax2.set_title("Dias inteiros sem leituras",
                  loc="left", fontsize=12, color=TINTA, fontweight="bold")
    cb = fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.02)
    cb.set_label("Fracao do dia em falta", fontsize=9, color=TINTA_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.5, colors=TINTA_2, length=0)

    fig.text(0.01, 0.005, f"Variavel: {coluna} (as 7 variaveis falham em simultaneo). "
             f"Janela: {df.index.min():%d/%m/%Y} a {df.index.max():%d/%m/%Y}.",
             fontsize=8, color=TINTA_2)
    if caminho_png:
        fig.savefig(caminho_png, dpi=160, bbox_inches="tight", facecolor="white")
    return fig


def classificar_lacunas(t, curta_max_min=5, media_max_min=180):
    """Classifica cada lacuna: curta (<=5 min), media (6-180 min) ou longa (>180 min)."""
    t = t.copy()
    t["tipo"] = np.select(
        [t["minutos"] <= curta_max_min, t["minutos"] <= media_max_min],
        ["curta", "media"], default="longa")
    return t


def tratar_missing(df, limite_interp_min=180):
    """Estrategia de tratamento, DIFERENTE conforme a duracao da lacuna.

    * Lacunas <= limite_interp_min (3 h): interpolacao linear no tempo.
      A simulacao (`avaliar_imputacao`) mostra erro absoluto medio de ~0,06 kW
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


def missing_por_periodo(df, freq="MS"):
    """Missing (%) por variavel e por periodo (mes por defeito)."""
    d = reindexar_minuto(df)
    return (100 * d.isna().groupby(pd.Grouper(freq=freq)).mean()).round(2)


def teste_mecanismo(df, n_perm=2000, seed=42, coluna="Global_active_power"):
    """Testa se o missing depende do consumo observado (contra MCAR).

    Teste de permutacao: o consumo medio nos 30 min ANTES de cada lacuna e
    comparado com o de instantes aleatorios (sem missing). Se o p-valor for
    pequeno, o aparecimento de lacunas esta associado a valores observados
    -> nao e MCAR (e pelo menos MAR).
    """
    rng = np.random.default_rng(seed)
    s = reindexar_minuto(df)[coluna]
    t = tabela_lacunas(df, coluna)
    antes = np.array([s.loc[: r.inicio - pd.Timedelta("1min")].dropna().tail(30).mean() for r in t.itertuples()])
    pos = np.flatnonzero(~s.isna().values)
    pos = pos[pos > 60]
    perm = np.array([np.mean([s.iloc[j - 30:j].mean() for j in rng.choice(pos, len(t))]) for _ in range(n_perm)])
    return {
        "consumo medio antes das lacunas (kW)": round(float(antes.mean()), 3),
        "consumo medio em instantes aleatorios (kW)": round(float(perm.mean()), 3),
        "p-valor (unilateral)": round(float(np.mean(perm >= antes.mean())), 4),
        "lacunas a dia 13 do mes": int((t["inicio"].dt.day == 13).sum()),
        "lacunas ao domingo": int((t["inicio"].dt.dayofweek == 6).sum()),
        "n lacunas": len(t),
    }


def avaliar_imputacao(df, comprimentos=(1, 2, 5, 10, 30, 60, 180, 720, 2880),
                      n_sim=200, seed=42, coluna="Global_active_power"):
    """Simula lacunas de varios comprimentos em zonas SEM missing e mede o erro
    absoluto medio (kW) de 4 metodos: interpolacao linear, ultimo valor (ffill),
    perfil medio hora x dia da semana e media global. Justifica o limiar de tratamento."""
    rng = np.random.default_rng(seed)
    s = reindexar_minuto(df)[coluna]
    perfil = s.groupby([s.index.dayofweek, s.index.hour]).mean()
    media = s.mean()
    linhas = []
    for L in comprimentos:
        erros = {"linear": [], "ffill": [], "perfil_sazonal": [], "media_global": []}
        n = 0
        while n < n_sim:
            i = int(rng.integers(200, len(s) - L - 200))
            w = s.iloc[i - 120:i + L + 120]
            if w.isna().any():
                continue
            n += 1
            m = w.copy()
            m.iloc[120:120 + L] = np.nan
            real = w.iloc[120:120 + L]
            sazonal = pd.Series(perfil.reindex(list(zip(real.index.dayofweek, real.index.hour))).values, index=real.index)
            erros["linear"].append((m.interpolate("time").iloc[120:120 + L] - real).abs().mean())
            erros["ffill"].append((m.ffill().iloc[120:120 + L] - real).abs().mean())
            erros["perfil_sazonal"].append((sazonal - real).abs().mean())
            erros["media_global"].append((media - real).abs().mean())
        linhas.append({"comprimento_min": L, **{k: round(float(np.mean(v)), 3) for k, v in erros.items()}})
    return pd.DataFrame(linhas).set_index("comprimento_min")


def dias_validos(df, max_frac_missing=0.1, coluna="Global_active_power"):
    """Datas cuja fracao de missing e <= max_frac_missing (para agregacoes diarias)."""
    f = missing_por_dia(df, coluna)
    return f[f <= max_frac_missing].index


def diagnostico_mecanismo(df, coluna="Global_active_power"):
    """Evidencia empirica para discutir MCAR / MAR / MNAR (nao e prova)."""
    s = reindexar_minuto(df)[coluna]
    falta = s.isna()
    horas = falta.groupby(falta.index.hour).mean()
    dow = falta.groupby(falta.index.dayofweek).mean()
    # consumo imediatamente antes de cada lacuna vs consumo geral
    t = tabela_lacunas(df, coluna)
    antes = [s.loc[:r.inicio - pd.Timedelta("1min")].dropna().tail(30).mean() for r in t.itertuples()]
    return {
        "% minutos em falta": round(100 * falta.mean(), 3),
        "n lacunas": len(t),
        "lacuna mais longa (h)": float(t["duracao_horas"].max()),
        "falha simultanea nas 7 variaveis": bool(
            (reindexar_minuto(df).isna().sum(axis=1).isin([0, len(COLUNAS)])).all()),
        "missing por hora (min-max %)": (round(100 * horas.min(), 2), round(100 * horas.max(), 2)),
        "missing por dia da semana (min-max %)": (round(100 * dow.min(), 2), round(100 * dow.max(), 2)),
        "consumo medio nos 30 min antes das lacunas": round(float(np.nanmean(antes)), 3),
        "consumo medio geral": round(float(s.mean()), 3),
    }


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ficheiro = sys.argv[1] if len(sys.argv) > 1 else "consumption.csv"
    out = Path("figures")
    out.mkdir(exist_ok=True)

    df = carregar_dados(ficheiro)
    print(f"{len(df):,} linhas | {df.index.min()} -> {df.index.max()}")
    print(df.dtypes, "\n")
    print(resumo_missing(df), "\n")

    t = classificar_lacunas(tabela_lacunas(df))
    print(t.to_string(index=False), "\n")
    print(t.groupby("tipo")["minutos"].agg(["count", "sum", "max"]), "\n")
    print("Diagnostico do mecanismo:")
    for k, v in diagnostico_mecanismo(df).items():
        print(f"  {k}: {v}")

    print("\nMissing (%) por mes (Global_active_power):")
    print(missing_por_periodo(df)["Global_active_power"].to_string())

    print("\nTeste de mecanismo (permutacao):")
    for k, v in teste_mecanismo(df).items():
        print(f"  {k}: {v}")

    print("\nErro absoluto medio (kW) por metodo e comprimento da lacuna:")
    print(avaliar_imputacao(df))

    tratado, imp = tratar_missing(df)
    print(f"\nImputados: {imp.sum()} minutos | NaN restantes: {tratado['Global_active_power'].isna().sum()}")
    print("Dias excluidos (>10% em falta apos tratamento):",
          sorted(set(missing_por_dia(tratado).pipe(lambda f: f[f > 0.1]).index.date)))

    plot_missing(df, out / "missing.png")
    print("Figura guardada em", out / "missing.png")