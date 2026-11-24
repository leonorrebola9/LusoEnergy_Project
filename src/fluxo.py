"""
fluxo.py - estatísticas online para a Parte IV (uma leitura de cada vez).

Cada classe recebe as leituras pela ordem em que chegam, com atualizar(...), e
nunca olha para a série completa. Assim, o mesmo código serve para o replay no
notebook e para o dashboard.
"""
from collections import deque
import math

import pandas as pd


class Welford:
    """Média e variância globais pelo algoritmo de Welford (custo constante por leitura)."""

    def __init__(self):
        self.n = 0
        self.media = 0.0
        self._m2 = 0.0                       # soma dos quadrados dos desvios à média

    def atualizar(self, x):
        self.n += 1
        delta = x - self.media               # desvio face à média ANTES de a atualizar
        self.media += delta / self.n
        self._m2 += delta * (x - self.media) # desvio antigo x desvio novo

    @property
    def variancia(self):
        """Variância amostral (ddof=1), comparável a pandas .var()."""
        return self._m2 / (self.n - 1) if self.n > 1 else float("nan")

    @property
    def desvio(self):
        return math.sqrt(self.variancia)


class JanelaDeslizante:
    """Média e desvio-padrão das leituras dos últimos `minutos` minutos.

    A janela é definida pelo TEMPO e não pelo número de leituras: se faltarem
    minutos, a janela fica com menos leituras, mas nunca abrange mais de 60 min.
    Guarda somas correntes, por isso cada atualização tem custo constante.
    """

    def __init__(self, minutos=60):
        self.largura = pd.Timedelta(minutes=minutos)
        self.leituras = deque()              # pares (instante, valor)
        self.soma = 0.0
        self.soma2 = 0.0

    def atualizar(self, t, x):
        self.leituras.append((t, x))
        self.soma += x
        self.soma2 += x * x
        while t - self.leituras[0][0] >= self.largura:   # tira o que já saiu da janela
            _, velho = self.leituras.popleft()
            self.soma -= velho
            self.soma2 -= velho * velho

    @property
    def n(self):
        return len(self.leituras)

    @property
    def media(self):
        return self.soma / self.n if self.n else float("nan")

    @property
    def desvio(self):
        if self.n < 2:
            return float("nan")
        var = (self.soma2 - self.soma * self.soma / self.n) / (self.n - 1)
        return math.sqrt(max(var, 0.0))      # max(): evita valores negativos minúsculos por arredondamento


class EWMA:
    """Média móvel exponencial: cada leitura nova pesa `alpha`, o passado pesa (1 - alpha)."""

    def __init__(self, alpha):
        self.alpha = alpha
        self.media = None

    def atualizar(self, x):
        self.media = x if self.media is None else self.alpha * x + (1 - self.alpha) * self.media


class AlarmeZ:
    """Alarme de anomalias: z-score da leitura face à janela ANTERIOR (sem a própria leitura).

    Dá alarme quando |z| > limiar, desde que a janela tenha pelo menos `min_leituras`.
    """

    def __init__(self, minutos=60, limiar=3.0, min_leituras=30):
        self.janela = JanelaDeslizante(minutos)
        self.limiar = limiar
        self.min_leituras = min_leituras

    def atualizar(self, t, x):
        """Devolve (z, alarme) para a leitura x e só depois a junta à janela."""
        z, alarme = float("nan"), False
        if self.janela.n >= self.min_leituras and self.janela.desvio > 0:
            z = (x - self.janela.media) / self.janela.desvio
            alarme = abs(z) > self.limiar
        self.janela.atualizar(t, x)
        return z, alarme