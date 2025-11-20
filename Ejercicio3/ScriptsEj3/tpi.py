import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.io.wavfile as wav  
from IPython.display import Audio # reproducir en colab
from scipy.linalg import toeplitz

# Funciones auxiliares
# ********************

def param_lpc(xs, P): ## EMI, ULI, esto lo saqué del ejercicio 1a.
    """
    Calcula los parámetros LPC de un segmento de señal.
    xs: segmento de señal (ya ventaneado)
    P: orden del predictor (cantidad de coeficientes)
    Retorna: (a, G) donde a son los coeficientes LPC y G es la ganancia
    """
    xs = np.asarray(xs, dtype=float)
    N = len(xs)
    
    # Autocorrelación no sesgada para lags 0 a P
    z = np.array([np.dot(xs[:N-k], xs[k:]) / (N-k) for k in range(P+1)])
    
    # Matriz de Toeplitz R (P x P) y vector r (P x 1)
    R = toeplitz(z[:P])
    r = z[1:P+1]
    
    # Resolver R*a = r (ecuaciones de Yule-Walker)
    try:
        a = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        # Si R es singular, usar mínimos cuadrados
        a = np.linalg.lstsq(R, r, rcond=None)[0]
    
    # Ganancia: G² = z[0] - a'*r
    G2 = z[0] - np.dot(a, r)
    G = np.sqrt(max(G2, 1e-12))  # Evitar raíz de negativos por errores numéricos
    
    return a, G

def pitch_lpc(xs, a, alpha, fs):
    # completar


def gen_pulsos(fp, N, fs):
    """
    Genera un tren de impulsos periodico en el tiempo.
    fp: frecuencia fundamental (pitch) del tren de impulsos [Hz].
    N: cantidad de puntos que posee el array de la secuencia generada.
    fs: frecuencia de muestreo [Hz].
    Retorna: tren de impulsos (con varianza normalizada) de frecuencia fp.
    """
    M = round(fs / fp)
    p = np.zeros(N)
    p[0::M] = np.sqrt(fs / fp)
    return p

def pitch_sintetico(i, fs=8000):
    """
    Genera una frecuencia de pitch artificial para sustituir la frecuencia real
    Recibe: i (índice del segmento actual), fs (sample rate)
    Retorna: frecuencia de pitch artificial
    """
    fc, fa, f1, f2 = 200, 100, 250, 71
    return fc + fa*np.sin(2*np.pi*f1/fs*i) * np.sin(2*np.pi*f2/fs*i)


# Código principal
# ****************

# -- Cargar audio (obtener señal y samplerate (fs)) --


# -- Parámetros generales --
# definir largo de cada segmento 
# definir ventana (hamming, hanning, etc.)
# cantidad de coeficientes del modelo
# umbral para detección de pitch (0<alpha<1)


# -- Codificación LPC --

data_lpc = []   # lista para almacenar los parámetros LPC de todos los segmentos

# (ciclo) Iteración para obtener cada segmento y sus parámetros LPC/pitch, 
# suponiendo un solapamiento del 50% entre segmentos.
     # Segmentar de señal 
     # Estimar (a, G y fp)
     # Actualizar lista data_lpc


# -- Decodificación LPC --

# (ciclo) Iterar sobre la lista de parámetros data_lpc y reconstruir señal
     # Extraer parámetros asociados al segmento actual 
     # Generar segmento sintético (usando excitación sorda o sonora)
     # Reconstruir señal con los segmentos sintéticos


# -- Reproducir audio de la señal reconstruida --

