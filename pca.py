"""
2.2) Perfis diarios com reducao de dimensao
-------------------------------------------
- Matriz de perfis: 1 linha por dia, 24 colunas = media horaria do consumo
- Padronizar (z-score por hora, ao longo dos dias) e aplicar PCA
- Variancia explicada + scree plot (com Kaiser e broken-stick)
- Loadings e interpretacao em linguagem de consumo (perfis +-2 desvios por PC)
- Projecao dos dias em PC1 x PC2 coloridos por estacao e por dia util/fim de semana
- Comparacao com t-SNE (varias perplexidades) e UMAP, SEMPRE com seed fixa
- Metricas objetivas para discutir o que cada projecao revela e esconde

Adaptar apenas a seccao CONFIG.
Dependencias: numpy, pandas, matplotlib (PCA) + scikit-learn (t-SNE) + umap-learn (opcional).
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# UMAP com random_state usa 1 thread (garante reprodutibilidade) e avisa disso: aviso esperado
warnings.filterwarnings("ignore", category=FutureWarning, module="umap")
warnings.filterwarnings("ignore", message=".*n_jobs.*overridden.*")

# ============================ CONFIG ============================
FICHEIRO   = "data/consumption.csv"            # caminho do dataset
SEP        = ";"                    # separador do CSV
DECIMAL    = "."                    # "," se os numeros vierem tipo 12,5
COL_TEMPO  = "datetime"             # coluna data/hora
FMT_TEMPO  = "%Y-%m-%d %H:%M:%S"
VAR        = "Global_active_power"  # variavel de consumo usada nos perfis
UNIDADE    = "kW"                   # so para os rotulos dos graficos

MIN_MIN_HORA       = 30             # minutos validos minimos para aceitar a media de uma hora
MAX_HORAS_EM_FALTA = 0              # dias com mais horas em falta do que isto sao descartados
                                    # (se > 0, as horas em falta sao interpoladas dentro do dia)
FERIADOS   = []                     # ex.: ["2008-12-25", "2009-01-01"] -> tratados como nao uteis

SEED       = 42                     # seed fixa (t-SNE e UMAP)
USAR_TSNE  = True
PERPLEXIDADES     = [10, 30, 60]    # para mostrar a (in)estabilidade do t-SNE
PERPLEX_PRINCIPAL = 30              # a usada na comparacao com PCA/UMAP
USAR_UMAP  = True                   # ignorado se umap-learn nao estiver instalado
UMAP_VIZINHOS = 15
UMAP_MIN_DIST = 0.1
# ================================================================

# ------------------------- estilo e cores -------------------------
# Cores categoricas validadas (daltonismo + contraste). Estacoes: azul/verde/amarelo/vermelho
# (+ forma do marcador como codificacao secundaria). Tipo de dia: azul/laranja.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
AZUL, LARANJA = "#2a78d6", "#eb6834"
EST_ORDEM = ["Inverno", "Primavera", "Verão", "Outono"]
EST_COR   = {"Inverno": "#2a78d6", "Primavera": "#008300", "Verão": "#eda100", "Outono": "#e34948"}
EST_MARC  = {"Inverno": "o", "Primavera": "s", "Verão": "^", "Outono": "D"}
MES_EST   = {12: "Inverno", 1: "Inverno", 2: "Inverno", 3: "Primavera", 4: "Primavera", 5: "Primavera",
             6: "Verão", 7: "Verão", 8: "Verão", 9: "Outono", 10: "Outono", 11: "Outono"}
TIPO_ORDEM = ["Dia útil", "Fim de semana / feriado"]
TIPO_COR   = {"Dia útil": AZUL, "Fim de semana / feriado": LARANJA}
TIPO_MARC  = {"Dia útil": "o", "Fim de semana / feriado": "s"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
    "font.size": 10, "legend.frameon": False,
})


def guardar(fig, nome):
    fig.savefig(nome, dpi=150)
    plt.close(fig)
    print(f"  -> {nome}")


def guardar_csv(df_, nome):
    df_.to_csv(nome, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"  -> {nome}")


# =============== 1) MATRIZ DE PERFIS (dia x 24 horas) ===============
print("=== 1) MATRIZ DE PERFIS ===")
cabecalho = [c.strip() for c in pd.read_csv(FICHEIRO, sep=SEP, nrows=0).columns]
for c in (COL_TEMPO, VAR):
    if c not in cabecalho:
        raise SystemExit(f"A coluna '{c}' não existe. Colunas disponíveis: {cabecalho}")

df = pd.read_csv(FICHEIRO, sep=SEP, decimal=DECIMAL,
                 usecols=lambda c: c.strip() in (COL_TEMPO, VAR))
df.columns = df.columns.str.strip()
df[COL_TEMPO] = pd.to_datetime(df[COL_TEMPO].astype(str).str.strip(), format=FMT_TEMPO, errors="coerce")
if df[VAR].dtype == object:  # "?" para valores em falta, virgulas decimais, espacos
    df[VAR] = df[VAR].astype(str).str.strip().str.replace(",", ".", regex=False)
df[VAR] = pd.to_numeric(df[VAR], errors="coerce")
print(f"Registos: {len(df)} | datas inválidas: {df[COL_TEMPO].isna().sum()} | "
      f"{VAR} em falta: {df[VAR].isna().sum()}")
df = df.dropna(subset=[COL_TEMPO]).sort_values(COL_TEMPO)

# media horaria (so aceita a hora se tiver minutos validos suficientes)
hora_ts = df[COL_TEMPO].dt.floor("60min")
g = df.groupby(hora_ts)[VAR].agg(["mean", "count"])
g.loc[g["count"] < MIN_MIN_HORA, "mean"] = np.nan
t = pd.DataFrame({"data": g.index.normalize(), "hora": g.index.hour, "v": g["mean"].to_numpy()})
X = t.pivot(index="data", columns="hora", values="v").reindex(columns=range(24))

n_dias_total = len(X)
falta = X.isna().sum(axis=1)
X = X[falta <= MAX_HORAS_EM_FALTA].copy()
if MAX_HORAS_EM_FALTA > 0:
    X = X.apply(lambda r: r.interpolate(limit_direction="both"), axis=1)
print(f"Dias na série: {n_dias_total} | descartados por horas em falta: {n_dias_total - len(X)} "
      f"| dias na matriz: {len(X)}")
if len(X) < 30:
    raise SystemExit("Menos de 30 dias completos: insuficiente para PCA/t-SNE. "
                     "Revê MIN_MIN_HORA / MAX_HORAS_EM_FALTA.")

datas = pd.to_datetime(X.index)
meta = pd.DataFrame(index=X.index)
nao_util = np.asarray((datas.dayofweek >= 5) | datas.isin(pd.to_datetime(FERIADOS)))
meta["tipo_dia"] = np.where(nao_util, TIPO_ORDEM[1], TIPO_ORDEM[0])
meta["estacao"] = pd.Series(np.asarray(datas.month), index=meta.index).map(MES_EST)
meta["dia_semana"] = np.asarray(datas.day_name())
print("Dias por estação:", meta["estacao"].value_counts().reindex(EST_ORDEM).dropna().astype(int).to_dict())
print("Dias por tipo:", meta["tipo_dia"].value_counts().to_dict())

# ====================== 2) PADRONIZAR + PCA ======================
print("\n=== 2) PCA (dados padronizados por hora) ===")
mu = X.mean()
sd = X.std(ddof=1)
if (sd == 0).any():
    raise SystemExit(f"Horas sem variância (desvio-padrão 0): {list(sd[sd == 0].index)}")
Z = (X - mu) / sd
Zv = Z.to_numpy()
n, p = Zv.shape

U, S, Vt = np.linalg.svd(Zv, full_matrices=False)
eig = S ** 2 / (n - 1)               # valores proprios (variancia de cada PC)
var_ratio = eig / eig.sum()
V = Vt.T.copy()                      # colunas = direcoes dos PCs (vetores proprios)
for k in range(V.shape[1]):          # convencao de sinal (os PCs so sao definidos a menos de sinal)
    inverter = V[:, k].sum() < 0 if k == 0 else V[np.argmax(np.abs(V[:, k])), k] < 0
    if inverter:
        V[:, k] *= -1
scores = Zv @ V
carga = V * np.sqrt(eig)             # loadings = correlacao entre cada hora e cada PC

cum = np.cumsum(var_ratio)
bs = np.array([sum(1.0 / i for i in range(k, p + 1)) / p for k in range(1, p + 1)])  # broken-stick
print(f"Variância explicada: PC1 {var_ratio[0]:.1%} | PC2 {var_ratio[1]:.1%} | PC3 {var_ratio[2]:.1%} "
      f"| acumulada PC1-3: {cum[2]:.1%}")
print(f"PCs para 80% da variância: {int(np.searchsorted(cum, 0.80) + 1)} | para 90%: "
      f"{int(np.searchsorted(cum, 0.90) + 1)}")
print(f"Critério de Kaiser (valor próprio > 1): {int((eig > 1).sum())} PCs | "
      f"broken-stick (PC > limiar aleatório): {int((var_ratio > bs).sum())} PCs")

# sensibilidade: PCA SEM padronizar (so centrar) -> horas com mais variancia dominam
Xc = X.to_numpy() - X.to_numpy().mean(axis=0)
eig_c = np.linalg.svd(Xc, compute_uv=False) ** 2 / (n - 1)
vr_c = eig_c / eig_c.sum()
print(f"Sensibilidade (sem padronizar): PC1 {vr_c[0]:.1%} | PC2 {vr_c[1]:.1%} | PC3 {vr_c[2]:.1%} "
      f"(padronizado: {var_ratio[0]:.1%} | {var_ratio[1]:.1%} | {var_ratio[2]:.1%})")


def faixas(horas):
    horas = sorted(horas)
    out, ini, prev = [], None, None
    for h in horas:
        if ini is None:
            ini = prev = h
        elif h == prev + 1:
            prev = h
        else:
            out.append((ini, prev))
            ini = prev = h
    if ini is not None:
        out.append((ini, prev))
    return ", ".join(f"{a}h" if a == b else f"{a}h-{b}h" for a, b in out) or "nenhuma"


print("\n--- Interpretação automática dos loadings (confirma sempre nos gráficos) ---")
for k in range(3):
    v = carga[:, k]
    m = np.max(np.abs(v))
    pos = [h for h in range(24) if v[h] >= 0.5 * m]
    neg = [h for h in range(24) if v[h] <= -0.5 * m]
    if np.all(v > 0) or np.all(v < 0):
        txt = (f"todas as horas têm o mesmo sinal -> PC{k+1} ALTO = consumo mais elevado em TODO o dia "
               f"(efeito de 'nível'); contribuem mais: {faixas(pos or neg)}")
    else:
        txt = (f"CONTRASTE -> PC{k+1} ALTO = mais consumo em [{faixas(pos)}] e menos em [{faixas(neg)}]; "
               f"PC baixo = o inverso")
    if var_ratio[k] < bs[k]:
        txt += "  [ATENÇÃO: abaixo do limiar broken-stick -> provavelmente ruído, não interpretar]"
    print(f"PC{k+1} ({var_ratio[k]:.1%}): {txt}")

# dias extremos em cada PC (para investigar/ilustrar)
print("\n--- Dias com scores extremos ---")
sc_df = pd.DataFrame(scores[:, :5], index=X.index, columns=[f"PC{i+1}" for i in range(5)])
for k in range(3):
    col = f"PC{k+1}"
    alto = sc_df[col].nlargest(3).index.strftime("%Y-%m-%d").tolist()
    baixo = sc_df[col].nsmallest(3).index.strftime("%Y-%m-%d").tolist()
    print(f"{col}: mais altos {alto} | mais baixos {baixo}")


def eta2(x, grupos):
    """Fraccao da variancia do score explicada pelo agrupamento (0 a 1)."""
    x = pd.Series(x, index=meta.index)
    gm = x.mean()
    total = ((x - gm) ** 2).sum()
    entre = x.groupby(grupos).agg(lambda s: len(s) * (s.mean() - gm) ** 2).sum()
    return entre / total


tab_eta = pd.DataFrame({
    "estação": [eta2(scores[:, k], meta["estacao"]) for k in range(3)],
    "tipo de dia": [eta2(scores[:, k], meta["tipo_dia"]) for k in range(3)],
}, index=["PC1", "PC2", "PC3"])
print("\nFracção da variância de cada PC explicada pelo agrupamento (eta2):")
print(tab_eta.round(3).to_string())

# ---------------- figuras do PCA ----------------
print("\nFiguras do PCA:")
horas = np.arange(24)

# (a) scree plot
kmax = min(12, len(eig))
fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
ax.bar(np.arange(1, kmax + 1), var_ratio[:kmax] * 100, color=AZUL, width=0.6,
       label="Variância explicada por PC")
ax.plot(np.arange(1, kmax + 1), cum[:kmax] * 100, color=INK2, marker="o", ms=4, lw=1.6,
        label="Variância acumulada")
ax.plot(np.arange(1, kmax + 1), bs[:kmax] * 100, color=LARANJA, ls="--", lw=1.4,
        label="Limiar broken-stick (aleatório)")
for k in range(min(3, kmax)):
    if var_ratio[k] > 0.30:   # barra alta: rótulo dentro da barra (evita colidir com a linha acumulada)
        ax.text(k + 1, var_ratio[k] * 100 / 2, f"{var_ratio[k]:.1%}", ha="center", va="center",
                fontsize=9.5, color="#ffffff", fontweight="bold")
    else:
        ax.text(k + 1, var_ratio[k] * 100 + 1.5, f"{var_ratio[k]:.1%}", ha="center", fontsize=8.5, color=INK)
ax.axhline(80, color=AXIS, lw=0.8, ls=":")
ax.axhline(90, color=AXIS, lw=0.8, ls=":")
ax.set_xticks(np.arange(1, kmax + 1))
ax.set_xlabel("Componente principal")
ax.set_ylabel("% da variância")
ax.set_ylim(0, 102)
ax.set_title(f"Scree plot - perfis diários de {VAR} (n={n} dias)", loc="left")
ax.legend(loc="center right", fontsize=9)
guardar(fig, "fig_pca_scree.png")

# (b) loadings por hora
fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
for k, ax in enumerate(axs):
    ax.axhline(0, color=AXIS, lw=1)
    ax.plot(horas, carga[:, k], color=AZUL, marker="o", ms=4, lw=1.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("correlação hora-PC")
    ax.set_title(f"PC{k+1} - {var_ratio[k]:.1%} da variância", loc="left")
axs[-1].set_xticks(range(0, 24, 2))
axs[-1].set_xlabel("Hora do dia")
guardar(fig, "fig_pca_loadings.png")

# (c) "o que e um PC alto?" -> perfil medio +-2 desvios ao longo de cada PC (unidades originais)
fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True, constrained_layout=True)
for k, ax in enumerate(axs):
    delta = 2 * np.sqrt(eig[k]) * V[:, k] * sd.to_numpy()
    media = mu.to_numpy()
    ax.fill_between(horas, media - delta, media + delta, color=GRID, alpha=0.7, lw=0)
    ax.plot(horas, media + delta, color=AZUL, lw=2, label="PC alto (+2 desvios)")
    ax.plot(horas, media - delta, color=LARANJA, lw=2, label="PC baixo (-2 desvios)")
    ax.plot(horas, media, color=INK2, lw=1.4, ls="--", label="perfil médio")
    ax.set_title(f"PC{k+1} ({var_ratio[k]:.1%})", loc="left")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hora do dia")
axs[0].set_ylabel(f"{VAR} ({UNIDADE})")
axs[0].legend(loc="upper left", fontsize=8.5)
guardar(fig, "fig_pca_perfis_por_pc.png")

# (d) projecao PC1 x PC2: por estacao e por tipo de dia


def dispersao(ax, E, grupos, ordem, cores, marcas, titulo, xl, yl, ticks=True, legenda=True):
    for gr in ordem:
        m = (grupos == gr).to_numpy()
        if not m.any():
            continue
        ax.scatter(E[m, 0], E[m, 1], s=24, c=cores[gr], marker=marcas[gr], edgecolors=SURFACE,
                   linewidths=0.5, alpha=0.9, label=f"{gr} (n={int(m.sum())})")
    ax.set_title(titulo, loc="left")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    if ticks:
        ax.axhline(0, color=AXIS, lw=0.8, zorder=0)
        ax.axvline(0, color=AXIS, lw=0.8, zorder=0)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if legenda:
        ax.legend(loc="best", fontsize=8.5, handletextpad=0.3)


E_pca = scores[:, :2]
xl, yl = f"PC1 ({var_ratio[0]:.1%})", f"PC2 ({var_ratio[1]:.1%})"
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
dispersao(axs[0], E_pca, meta["estacao"], EST_ORDEM, EST_COR, EST_MARC, "PCA: dias por estação do ano", xl, yl)
dispersao(axs[1], E_pca, meta["tipo_dia"], TIPO_ORDEM, TIPO_COR, TIPO_MARC, "PCA: dias úteis vs fim de semana", xl, yl)
guardar(fig, "fig_pca_projecao.png")

ests = [e for e in EST_ORDEM if (meta["estacao"] == e).any()]
fig, axs = plt.subplots(1, len(ests), figsize=(3.8 * len(ests), 3.9), sharex=True, sharey=True,
                        constrained_layout=True)
for ax, e in zip(np.atleast_1d(axs), ests):
    m = (meta["estacao"] == e).to_numpy()
    ax.scatter(E_pca[:, 0], E_pca[:, 1], s=14, c=AXIS, edgecolors="none", alpha=0.6)
    ax.scatter(E_pca[m, 0], E_pca[m, 1], s=24, c=EST_COR[e], marker=EST_MARC[e], edgecolors=SURFACE,
               linewidths=0.5)
    ax.set_title(f"{e} (n={int(m.sum())})", loc="left")
    ax.set_xlabel(xl)
np.atleast_1d(axs)[0].set_ylabel(yl)
guardar(fig, "fig_pca_estacoes_facetas.png")

# ====================== 3) t-SNE e UMAP ======================
embeddings = {"PCA (PC1 x PC2)": E_pca}
metricas = []
extra_cols = {}

if USAR_TSNE:
    print("\n=== 3) t-SNE / UMAP (seed fixa = %d) ===" % SEED)
    try:
        from sklearn.manifold import TSNE, trustworthiness
        from sklearn.metrics import pairwise_distances
    except ImportError:
        print("scikit-learn não instalado: 'pip install scikit-learn' para correr o t-SNE. "
              "O PCA já foi concluído.")
        USAR_TSNE = False

if USAR_TSNE:
    rng = np.random.default_rng(SEED)
    idx = np.arange(n) if n <= 1500 else np.sort(rng.choice(n, 1500, replace=False))
    iu = np.triu_indices(len(idx), 1)
    d_orig = pairwise_distances(Zv[idx])[iu]

    def medir(nome, E, kl=None):
        tw = trustworthiness(Zv, E, n_neighbors=10)
        rho = pd.Series(d_orig).corr(pd.Series(pairwise_distances(E[idx])[iu]), method="spearman")
        metricas.append({"projeção": nome, "trustworthiness (k=10)": tw,
                         "correlação de distâncias (Spearman)": rho, "KL (só t-SNE)": kl})

    medir("PCA (PC1 x PC2)", E_pca)

    perps = sorted({pp for pp in PERPLEXIDADES + [PERPLEX_PRINCIPAL] if pp < n})
    tsne_emb = {}
    for pp in perps:
        ts = TSNE(n_components=2, perplexity=pp, init="pca", learning_rate="auto", random_state=SEED)
        tsne_emb[pp] = ts.fit_transform(Zv)
        medir(f"t-SNE (perplexidade {pp})", tsne_emb[pp], kl=ts.kl_divergence_)
        extra_cols[f"tSNE{pp}_1"], extra_cols[f"tSNE{pp}_2"] = tsne_emb[pp][:, 0], tsne_emb[pp][:, 1]
    pp_princ = PERPLEX_PRINCIPAL if PERPLEX_PRINCIPAL in tsne_emb else perps[len(perps) // 2]
    embeddings[f"t-SNE (perplexidade {pp_princ})"] = tsne_emb[pp_princ]

    if USAR_UMAP:
        try:
            import umap
            um = umap.UMAP(n_neighbors=UMAP_VIZINHOS, min_dist=UMAP_MIN_DIST, random_state=SEED)
            E_umap = um.fit_transform(Zv)
            embeddings["UMAP"] = E_umap
            medir("UMAP", E_umap)
            extra_cols["UMAP_1"], extra_cols["UMAP_2"] = E_umap[:, 0], E_umap[:, 1]
        except ImportError:
            print("umap-learn não instalado ('pip install umap-learn'): UMAP ignorado.")

    tab_met = pd.DataFrame(metricas).set_index("projeção")
    print("\nMétricas de fidelidade (1 = perfeito):")
    print(tab_met.round(3).to_string())
    print("  trustworthiness: os vizinhos próximos no mapa são mesmo vizinhos nos dados originais?")
    print("  correlação de distâncias: as distâncias GLOBAIS entre dias foram preservadas?")

    # figuras t-SNE
    print("\nFiguras t-SNE/UMAP:")
    fig, axs = plt.subplots(2, len(perps), figsize=(4.4 * len(perps), 8.4), constrained_layout=True,
                            squeeze=False)
    for j, pp in enumerate(perps):
        dispersao(axs[0, j], tsne_emb[pp], meta["tipo_dia"], TIPO_ORDEM, TIPO_COR, TIPO_MARC,
                  f"t-SNE, perplexidade {pp}", "t-SNE 1 (sem unidades)", "t-SNE 2 (sem unidades)",
                  ticks=False, legenda=(j == 0))
        dispersao(axs[1, j], tsne_emb[pp], meta["estacao"], EST_ORDEM, EST_COR, EST_MARC,
                  f"t-SNE, perplexidade {pp}", "t-SNE 1 (sem unidades)", "t-SNE 2 (sem unidades)",
                  ticks=False, legenda=(j == 0))
    guardar(fig, "fig_tsne_perplexidades.png")

    nomes = list(embeddings)
    fig, axs = plt.subplots(2, len(nomes), figsize=(4.8 * len(nomes), 8.6), constrained_layout=True,
                            squeeze=False)
    for j, nome in enumerate(nomes):
        E = embeddings[nome]
        linha = tab_met.loc[nome]
        sub = f"{nome}\ntrust={linha['trustworthiness (k=10)']:.3f} | dist={linha['correlação de distâncias (Spearman)']:.3f}"
        e_pca = nome.startswith("PCA")
        dispersao(axs[0, j], E, meta["tipo_dia"], TIPO_ORDEM, TIPO_COR, TIPO_MARC, sub,
                  xl if e_pca else "dim 1 (sem unidades)", yl if e_pca else "dim 2 (sem unidades)",
                  ticks=e_pca, legenda=(j == 0))
        dispersao(axs[1, j], E, meta["estacao"], EST_ORDEM, EST_COR, EST_MARC, sub,
                  xl if e_pca else "dim 1 (sem unidades)", yl if e_pca else "dim 2 (sem unidades)",
                  ticks=e_pca, legenda=(j == 0))
    guardar(fig, "fig_comparacao_projecoes.png")

# ====================== 4) EXPORTAR ======================
print("\nFicheiros de dados:")
saida = meta.join(sc_df)
for nome_col, vals in extra_cols.items():
    saida[nome_col] = vals
guardar_csv(saida, "scores_e_projecoes.csv")
guardar_csv(X.round(4), "matriz_perfis_diarios.csv")
guardar_csv(pd.DataFrame(carga[:, :5], index=[f"{h}h" for h in horas],
                         columns=[f"PC{i+1}" for i in range(5)]).round(4), "loadings_pca.csv")
guardar_csv(pd.DataFrame({"valor_próprio": eig, "variância_%": var_ratio * 100,
                          "acumulada_%": cum * 100, "broken_stick_%": bs * 100},
                         index=[f"PC{i+1}" for i in range(len(eig))]).round(4), "variancia_explicada.csv")
if USAR_TSNE:
    guardar_csv(tab_met.round(4), "metricas_projecoes.csv")
print("\nConcluído.")