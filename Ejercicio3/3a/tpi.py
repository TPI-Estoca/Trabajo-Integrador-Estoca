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
    """
    Detecta el pitch y clasifica el segmento como sonoro o sordo.
    xs: segmento de señal (ya ventaneado)
    a: coeficientes LPC
    alpha: umbral de decisión (0 < alpha < 1)
    fs: frecuencia de muestreo [Hz]
    Retorna: (G, fp) donde G es la ganancia y fp el pitch (0 si es sordo)
    """
    xs = np.asarray(xs, dtype=float)
    N = len(xs)
    
    # Calcular la autocorrelación para detección de pitch
    # Usamos un rango típico para voz: 50-500 Hz
    min_lag = int(fs / 500)  # Lag mínimo (~2ms para pitch máximo 500Hz)
    max_lag = int(fs / 50)   # Lag máximo (~20ms para pitch mínimo 50Hz)
    
    # Autocorrelación normalizada
    r0 = np.dot(xs, xs)  # Energía en lag 0
    
    if r0 < 1e-10:  # Segmento de silencio
        return 0, 0
    
    # Calcular autocorrelación para el rango de lags posibles
    acf = np.array([np.dot(xs[:N-k], xs[k:]) for k in range(min_lag, min(max_lag, N))])
    
    # Normalizar
    acf_norm = acf / r0
    
    # Encontrar el máximo (excluyendo lag 0)
    if len(acf_norm) > 0:
        max_idx = np.argmax(acf_norm)
        max_acf = acf_norm[max_idx]
        pitch_lag = max_idx + min_lag
    else:
        max_acf = 0
        pitch_lag = min_lag
    
    # Decisión sonoro/sordo basada en el máximo de autocorrelación
    if max_acf > alpha:
        # SONORO: calcular pitch
        fp = fs / pitch_lag
        
        # Calcular ganancia del residuo
        # e[n] = x[n] - sum(a[k]*x[n-k])
        residuo = np.copy(xs)
        for k in range(len(a)):
            residuo[k+1:] -= a[k] * xs[:-k-1]
        
        G = np.sqrt(np.mean(residuo**2))
    else:
        # SORDO: sin pitch periódico
        fp = 0
        
        # Ganancia para ruido blanco
        residuo = np.copy(xs)
        for k in range(len(a)):
            residuo[k+1:] -= a[k] * xs[:-k-1]
        
        G = np.sqrt(np.mean(residuo**2))
    
    return G, fp

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

# -- Cargar audio --
fs, signal_audio = wav.read('..\\AudiosEj3\\audio_01.wav')  # o el nombre del archivo
signal_audio = signal_audio.astype(float)  # Convertir a float


# -- Parámetros generales --
L = 240                          # Largo de cada segmento (30ms a 8kHz)
ventana = np.hamming(L)          # Ventana de Hamming
P = 12                           # Orden del predictor (coeficientes LPC)
alpha = 0.4                      # Umbral para detección sonoro/sordo
hop = L // 2                     # Solapamiento del 50%


# -- Codificación LPC --
data_lpc = []
num_segmentos = (len(signal_audio) - L) // hop + 1

for i in range(num_segmentos):
    inicio = i * hop
    fin = inicio + L
    
    if fin > len(signal_audio):
        break
    
    # Extraer y ventanear segmento
    segmento = signal_audio[inicio:fin] * ventana
    
    # Estimar parámetros LPC
    a, G_param = param_lpc(segmento, P)
    G, fp = pitch_lpc(segmento, a, alpha, fs)
    
    # Guardar parámetros
    data_lpc.append((a, G, fp))


# -- Decodificación LPC --
signal_recons = np.zeros(len(signal_audio))

for i, (a, G, fp) in enumerate(data_lpc):
    inicio = i * hop
    
    # Generar excitación
    if fp > 0:  # SONORO
        excitacion = gen_pulsos(fp, L, fs)
    else:       # SORDO
        excitacion = np.random.randn(L)
    
    excitacion *= G  # Escalar por ganancia
    
    # Filtrar con modelo LPC: H(z) = G / (1 + sum(a_k * z^-k))
    segmento_sint = signal.lfilter([1], np.concatenate(([1], a)), excitacion)
    
    # Overlap-add
    signal_recons[inicio:inicio+L] += segmento_sint * ventana


# -- Reproducir/guardar --
signal_recons = signal_recons / np.max(np.abs(signal_recons))  # Normalizar
wav.write('audios_reconstruidos\\audio_01-reconstruido.wav', fs, signal_recons.astype(np.int16))


