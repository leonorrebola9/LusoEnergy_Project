"""
1.3) Sumário estatístico com juízo
- Localização/dispersão adequadas à forma (média vs mediana, com skew)
- Outliers: regra 1.5xIQR e z-score robusto (mediana/MAD)
- Investigação dos extremos (erro? evento real? fraude potencial?)
- Sumário estratificado: hora do dia x dia da semana (pivot com n)

Adaptar apenas a secção CONFIG.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mad(s):
    """MAD escalado (x1.4826 -> comparável ao desvio-padrão se normal)."""
    s = pd.Series(s).dropna()
    return 1.4826 * np.median(np.abs(s - np.median(s)))

# ============================ CONFIG ============================
FICHEIRO   = "consumption.csv"          # caminho do dataset
SEP        = ";"                  # separador do CSV
DECIMAL    = "."                  # "," se os números vierem tipo 12,5
COL_TEMPO  = "datetime"           # coluna data/hora
FMT_TEMPO  = "%Y-%m-%d %H:%M:%S"  # formato: 2008-05-01 00:01:00
COL_ID     = None                 # ex.: "cliente_id" (None se não existir)
COL_TEMP   = None                 # ex.: "temperatura" (para testar ondas de frio; None se não existir)
# variáveis do sumário descritivo (inclui os sub-contadores)
VARS       = ["Global_active_power", "Global_reactive_power", "Voltage",
              "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
# variáveis do estudo de outliers: sem os sub-contadores, que são ~0 quase sempre
# (MAD = 0 e IQR = 0 -> qualquer valor > 0 seria "outlier"); analisam-se à parte
VARS_OUT   = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity"]
VAR_ESTRAT = "Global_active_power"  # variável para o pivot hora x dia da semana
MIN_EPISODIO = 5                  # nº mínimo de registos consecutivos para um extremo ser "evento" (5 = 5 min)
# ================================================================

df = pd.read_csv(FICHEIRO, sep=SEP, decimal=DECIMAL)
# o ';' no fim de cada linha cria uma coluna vazia ("Unnamed: N"): remover
df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
df.columns = df.columns.str.strip()
df[COL_TEMPO] = pd.to_datetime(df[COL_TEMPO].astype(str).str.strip(),
                               format=FMT_TEMPO, errors="coerce")
print("Datas inválidas descartadas:", df[COL_TEMPO].isna().sum())
print(df[COL_TEMPO].agg(["min", "max"]).to_string())
print("Intervalo típico entre registos:", df[COL_TEMPO].sort_values().diff().mode()[0])
df = df.dropna(subset=[COL_TEMPO]).sort_values(COL_TEMPO).reset_index(drop=True)

df["hora"] = df[COL_TEMPO].dt.hour
df["dia_semana"] = df[COL_TEMPO].dt.dayofweek  # 0=Seg
df["dia_semana_nome"] = df[COL_TEMPO].dt.day_name()
df["data"] = df[COL_TEMPO].dt.date

# --- validação: as variáveis analisadas têm de existir e ser NUMÉRICAS ---
for col in dict.fromkeys(list(VARS) + list(VARS_OUT) + [VAR_ESTRAT]):
    if col not in df.columns:
        raise SystemExit(f"A coluna '{col}' não existe. Colunas disponíveis: {list(df.columns)}")
    if col == COL_TEMPO:
        raise SystemExit(f"'{col}' é a coluna de data/hora, não pode ser uma variável a analisar. "
                         f"Põe em VARS / VAR_ESTRAT o nome da coluna numérica (ex.: consumo).")
    if not pd.api.types.is_numeric_dtype(df[col]):
        if df[col].dtype == object:  # texto com vírgula decimal, espaços, etc.
            df[col] = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n_ok = df[col].notna().sum()
    print(f"Coluna '{col}': {n_ok} valores numéricos, {df[col].isna().sum()} em falta/inválidos")
    if n_ok == 0:
        raise SystemExit(f"A coluna '{col}' não tem valores numéricos. Verifica o nome e o DECIMAL.")


# ---------------------- 1) LOCALIZAÇÃO / DISPERSÃO ----------------------
def sumario(s: pd.Series) -> dict:
    s = s.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    mediana = s.median()
    mad_val = mad(s)  # MAD escalado (~desvio-padrão)
    skew = s.skew()
    kurt = s.kurt()  # excesso
    cv = s.std() / s.mean() if s.mean() != 0 else np.nan
    # Decisão média vs mediana
    if abs(skew) < 0.5:
        escolha = "média ± desvio-padrão (distribuição ~simétrica)"
    else:
        escolha = "mediana + IQR/MAD (assimetria forte, |skew|>=0.5)"
    return {
        "n": len(s), "média": s.mean(), "mediana": mediana,
        "desvio-padrão": s.std(), "IQR": q3 - q1, "MAD (escalado)": mad_val,
        "min": s.min(), "max": s.max(), "skewness": skew,
        "curtose (excesso)": kurt, "CV": cv,
        "média-mediana": s.mean() - mediana,
        "medida recomendada": escolha,
    }


resumo = pd.DataFrame({v: sumario(df[v]) for v in VARS})
print("\n=== SUMÁRIO DESCRITIVO ===")
print(resumo.to_string())

# Histogramas + boxplots para justificar visualmente a escolha
for v in VARS:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    df[v].hist(bins=50, ax=ax[0])
    ax[0].axvline(df[v].mean(), color="r", ls="--", label="média")
    ax[0].axvline(df[v].median(), color="g", ls="-", label="mediana")
    ax[0].set_title(f"Distribuição de {v}")
    ax[0].legend()
    df.boxplot(column=v, ax=ax[1])
    ax[1].set_title(f"Boxplot de {v}")
    plt.tight_layout()
    plt.savefig(f"dist_{v}.png", dpi=130)
    plt.close()


# ---------------------------- 2) OUTLIERS ----------------------------
def outliers_iqr(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def outliers_zrobusto(s, limiar=3.5):
    med = s.median()
    m = mad(s)
    if m == 0:
        print(f"AVISO: MAD = 0 em '{s.name}' (mais de metade dos valores iguais): "
              f"o z-score robusto não é aplicável a esta variável.")
    z = (s - med) / m if m > 0 else pd.Series(0, index=s.index)
    return z.abs() > limiar, z


for v in VARS_OUT:
    df[f"{v}_out_iqr"] = outliers_iqr(df[v])
    flag_z, z = outliers_zrobusto(df[v])
    df[f"{v}_out_z"] = flag_z
    df[f"{v}_zrob"] = z
    # z-score clássico só para comparação (mostra o masking da média/desvio)
    df[f"{v}_z_classico"] = (df[v] - df[v].mean()) / df[v].std(ddof=0)

    ambos = df[f"{v}_out_iqr"] & df[f"{v}_out_z"]
    so_iqr = df[f"{v}_out_iqr"] & ~df[f"{v}_out_z"]
    so_z = ~df[f"{v}_out_iqr"] & df[f"{v}_out_z"]
    print(f"\n=== OUTLIERS em '{v}' ===")
    print(f"IQR 1.5x: {df[f'{v}_out_iqr'].sum()} | z robusto>3.5: {flag_z.sum()} "
          f"| ambos: {ambos.sum()} | só IQR: {so_iqr.sum()} | só z-rob: {so_z.sum()}")


# ------------------- 3) INVESTIGAÇÃO DOS EXTREMOS -------------------
def investigar(v):
    ext = df[df[f"{v}_out_iqr"] | df[f"{v}_out_z"]].copy()
    if ext.empty:
        print("Sem extremos.")
        return ext

    # a) Erros de medição: valores impossíveis
    ext["erro_neg_ou_zero"] = ext[v] <= 0
    ext["erro_valor_repetido"] = ext[v].isin(
        df[v].value_counts()[lambda x: x > 20].index)  # valor "preso" (sensor)

    # b) Eventos reais: EPISÓDIOS = extremos em registos consecutivos.
    #    (com dados ao minuto, "vários extremos no mesmo dia" é sempre verdade;
    #     o que distingue um evento real é a persistência no tempo)
    flag = df[f"{v}_out_z"]
    grupo = (flag != flag.shift()).cumsum()
    duracao = flag.groupby(grupo).transform("sum").where(flag, 0)  # nº de registos do episódio
    ext["duracao_episodio"] = duracao.loc[ext.index]
    ext["evento_cluster_dia"] = ext["duracao_episodio"] >= MIN_EPISODIO

    n_ep = grupo[flag].nunique()
    n_ep_longos = grupo[flag & (duracao >= MIN_EPISODIO)].nunique()
    print(f"\nEpisódios de extremos (z robusto): {n_ep} | "
          f"com duração >= {MIN_EPISODIO} registos: {n_ep_longos}")

    # c) Ondas de frio: temperatura extrema no mesmo instante
    if COL_TEMP:
        t_lim = df[COL_TEMP].quantile(0.05)
        ext["frio_extremo"] = ext[COL_TEMP] <= t_lim
        print(f"Extremos coincidentes com frio (<= p5 = {t_lim:.1f}): "
              f"{ext['frio_extremo'].mean():.0%}")

    # d) Fraude potencial: pico isolado, fora do padrão horário, e/ou sempre o mesmo ID
    med_hora = df.groupby("hora")[v].transform("median")
    mad_hora = df.groupby("hora")[v].transform(
        lambda s: mad(s))
    ext["z_condicional_hora"] = ((df[v] - med_hora) / mad_hora).loc[ext.index]
    ext["pico_isolado"] = ~ext["evento_cluster_dia"] & (ext["z_condicional_hora"].abs() > 5)
    if COL_ID:
        cont = ext[COL_ID].value_counts()
        print("\nIDs com mais extremos (possível fraude/avaria):")
        print(cont.head(10).to_string())

    # Classificação heurística (a validar manualmente!)
    def classifica(r):
        if r["erro_neg_ou_zero"] or r["erro_valor_repetido"]:
            return "possível ERRO"
        if r["evento_cluster_dia"] or (COL_TEMP and r.get("frio_extremo", False)):
            return "provável EVENTO REAL"
        if r["pico_isolado"]:
            return "suspeito (FRAUDE/avaria?) - rever"
        return "indeterminado"

    ext["classificacao"] = ext.apply(classifica, axis=1)
    print("\nClassificação heurística dos extremos:")
    print(ext["classificacao"].value_counts().to_string())
    print("\nTop 15 extremos:")
    print(ext.sort_values(v, ascending=False)
             .head(15)[[COL_TEMPO, v, f"{v}_zrob", "classificacao"]].to_string())
    return ext


extremos = {v: investigar(v) for v in VARS_OUT}

# Série temporal com extremos marcados
for v in VARS_OUT:
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df[COL_TEMPO], df[v], lw=0.5, label=v)
    m = df[f"{v}_out_z"]
    ax.scatter(df.loc[m, COL_TEMPO], df.loc[m, v], c="r", s=12, label="outlier z-rob")
    ax.legend()
    ax.set_title(f"{v} ao longo do tempo (outliers a vermelho)")
    plt.tight_layout()
    plt.savefig(f"serie_{v}.png", dpi=130)
    plt.close()


# ---------------- 4) SUMÁRIO ESTRATIFICADO hora x dia semana ----------------
ordem = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

pivot_med = df.pivot_table(index="hora", columns="dia_semana_nome",
                           values=VAR_ESTRAT, aggfunc="median")[ordem]
pivot_iqr = df.pivot_table(index="hora", columns="dia_semana_nome",
                           values=VAR_ESTRAT,
                           aggfunc=lambda s: s.quantile(.75) - s.quantile(.25))[ordem]
pivot_n = df.pivot_table(index="hora", columns="dia_semana_nome",
                         values=VAR_ESTRAT, aggfunc="count")[ordem]
pivot_media = df.pivot_table(index="hora", columns="dia_semana_nome",
                             values=VAR_ESTRAT, aggfunc="mean")[ordem]

print("\n=== MEDIANA por hora x dia da semana ===")
print(pivot_med.round(2).to_string())
print("\n=== n por célula ===")
print(pivot_n.to_string())
print(f"\nCélulas com n < 30 (estimativas frágeis): {(pivot_n < 30).sum().sum()}")

# Formato longo (tabela única com tudo, útil para o relatório)
estrat = (df.groupby(["hora", "dia_semana_nome"])[VAR_ESTRAT]
            .agg(n="count", media="mean", mediana="median", desvio="std",
                 q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75))
            .reset_index())
estrat.to_csv("sumario_estratificado.csv", index=False)

# Heatmap
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
for a, p, t in zip(ax, [pivot_med, pivot_n], ["Mediana", "n (observações)"]):
    im = a.imshow(p.values, aspect="auto", cmap="viridis")
    a.set_xticks(range(7)); a.set_xticklabels([d[:3] for d in ordem])
    a.set_yticks(range(24)); a.set_yticklabels(range(24))
    a.set_title(f"{t} - {VAR_ESTRAT}")
    plt.colorbar(im, ax=a)
plt.tight_layout()
plt.savefig("heatmap_hora_dia.png", dpi=130)
plt.close()

# Exportar tudo para Excel
with pd.ExcelWriter("sumario_estatistico.xlsx") as xw:
    resumo.to_excel(xw, sheet_name="descritivo")
    pivot_med.to_excel(xw, sheet_name="mediana_hora_dia")
    pivot_media.to_excel(xw, sheet_name="media_hora_dia")
    pivot_iqr.to_excel(xw, sheet_name="IQR_hora_dia")
    pivot_n.to_excel(xw, sheet_name="n_hora_dia")
    for v, e in extremos.items():
        if not e.empty:
            e.to_excel(xw, sheet_name=f"extremos_{v}"[:31], index=False)

print("\nFicheiros gerados: sumario_estatistico.xlsx, sumario_estratificado.csv, "
      "dist_*.png, serie_*.png, heatmap_hora_dia.png")