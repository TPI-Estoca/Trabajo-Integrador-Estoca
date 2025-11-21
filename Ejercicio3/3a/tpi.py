import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.io.wavfile as wav  
from IPython.display import Audio # reproducir en colab
from scipy.linalg import toeplitz
import os # Para manejo de archivos/carpetas

# Funciones auxiliares
# ********************

def param_lpc(xs, P):
    """
    Calcula los parámetros LPC (coeficientes y ganancia del error de predicción)
    usando el método de autocorrelación de Yule-Walker.
    xs: segmento de señal (ya ventaneado)
    P: orden del predictor (cantidad de coeficientes)
    Retorna: (a, G) donde a son los coeficientes LPC y G es la ganancia del error.
    """
    xs = np.asarray(xs, dtype=float)
    N = len(xs)
    
    # Autocorrelación (solo r[0] a r[P])
    r_full = np.correlate(xs, xs, mode='full')
    r = r_full[N-1 : N+P] # r[0] a r[P]
    
    # Matriz de Toeplitz R (P x P) y vector r_vec (P x 1)
    R = toeplitz(r[:P])
    r_vec = r[1:P+1]
    
    # Resolver R*a = r_vec (ecuaciones de Yule-Walker)
    try:
        a = np.linalg.solve(R, r_vec)
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(R, r_vec, rcond=None)[0]
    
    # Ganancia G (Energía del error de predicción): G² = r[0] - a'*r_vec
    G2 = r[0] - np.dot(a, r_vec)
    G = np.sqrt(max(G2, 1e-12)) 
    
    return a, G

def pitch_lpc(xs, fs):
    """
    Detecta el pitch del segmento usando autocorrelación.
    xs: segmento de señal (ya ventaneado)
    fs: frecuencia de muestreo [Hz]
    Retorna: (fp, max_acf) donde fp es el pitch [Hz] (0 si no se detecta) y 
             max_acf es el valor máximo de autocorrelación normalizada.
    """
    xs = np.asarray(xs, dtype=float)
    N = len(xs)
    
    # Rango de pitch: 50 Hz (max lag) a 500 Hz (min lag)
    min_lag = int(fs / 500)
    max_lag = int(fs / 50)
    
    if N < max_lag or N < min_lag:
         return 0, 0

    r0 = np.dot(xs, xs) # Energía en lag 0
    if r0 < 1e-10: # Silencio
        return 0, 0
    
    # Autocorrelación normalizada en el rango de lags
    r_full = np.correlate(xs, xs, mode='full')
    # Extraemos el rango de lags relevante: indices de N-1+min_lag a N-1+max_lag
    r_lags = r_full[N-1 + min_lag : N-1 + max_lag + 1]
    
    acf_norm = r_lags / r0
        
    # Encontrar el máximo
    max_idx = np.argmax(acf_norm)
    max_acf = acf_norm[max_idx]
    pitch_lag = max_idx + min_lag
    
    fp = fs / pitch_lag
    
    return fp, max_acf

def gen_pulsos(fp, N, fs):
    """
    Genera un tren de impulsos periodico en el tiempo con varianza normalizada.
    """
    M = round(fs / fp)
    p = np.zeros(N)
    p[0::M] = np.sqrt(fs / fp) # Varianza normalizada
    return p

def pitch_sintetico(i, fs=8000):
    """
    Genera una frecuencia de pitch artificial para sustituir la frecuencia real
    """
    fc, fa, f1, f2 = 200, 100, 250, 71
    return fc + fa*np.sin(2*np.pi*f1/fs*i) * np.sin(2*np.pi*f2/fs*i)

def plot_spectrogram(signal_audio, fs, title, filename):
    """
    Genera y guarda el espectrograma de una señal de audio.

    signal_audio: La señal de audio (array de numpy).
    fs: Frecuencia de muestreo (Hz).
    title: Título del gráfico.
    filename: Nombre del archivo para guardar la imagen.
    """
    # Parámetros para el espectrograma
    # Nperseg: Largo de la ventana de análisis (por ejemplo, 512 puntos)
    # Noverlap: Solapamiento (por ejemplo, 50% de Nperseg)
    nperseg = 512 
    noverlap = nperseg // 2

    # Cálculo del espectrograma
    f, t, Sxx = signal.spectrogram(
        signal_audio, 
        fs, 
        nperseg=nperseg, 
        noverlap=noverlap, 
        window='hann', 
        scaling='spectrum' # o 'density'
    )

    # Convertir a dB (logarítmico)
    # Evitar log(0)
    Sxx_dB = 10 * np.log10(Sxx + 1e-12) 

    plt.figure(figsize=(12, 6))
    
    # Crear el mapa de color (spectrograma)
    plt.pcolormesh(t, f, Sxx_dB, shading='gouraud')
    
    # Configuración de ejes y etiquetas
    plt.ylabel('Frecuencia [Hz]')
    plt.xlabel('Tiempo [s]')
    plt.title(title)
    
    # Barra de color para la intensidad
    cbar = plt.colorbar(label='Intensidad [dB]')
    
    # Limitar el eje Y (por ejemplo, a 4000 Hz, si fs=8000)
    plt.ylim(0, fs / 2)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"Espectrograma '{filename}' generado con éxito.")


# Código principal (Ejercicio 3a)
# ****************

# --- Cargar audio ---
fs, signal_audio = wav.read('..\\AudiosEj3\\audio_01.wav')
signal_audio = signal_audio.astype(float) 


# --- Parámetros generales ---
L = 240 # Largo de cada segmento (30ms a 8kHz)
ventana = np.hamming(L)
P = 12 # Orden del predictor (coeficientes LPC). Típicamente P = fs/1000 + 2
alpha = 0.4 # Umbral de autocorrelación para detección sonoro/sordo (ajustable)
hop = L // 2 # Solapamiento del 50%


# --- Codificación LPC ---
data_lpc = []
# Se agrega una pequeña cantidad de ceros al final para que el último segmento tenga el largo L
signal_audio_padded = np.pad(signal_audio, (0, L), 'constant')
num_segmentos = (len(signal_audio_padded) - L) // hop + 1

for i in range(num_segmentos):
    inicio = i * hop
    fin = inicio + L
    
    segmento = signal_audio_padded[inicio:fin] * ventana
    
    # 1. Estimar LPC (a y Ganancia G del error de predicción)
    a, G = param_lpc(segmento, P)
    
    # 2. Estimar Pitch
    fp_est, max_acf = pitch_lpc(segmento, fs)
    
    # 3. Clasificación Sonoro/Sordo
    if max_acf > alpha:
        fp = fp_est # SONORO: usar pitch estimado
    else:
        fp = 0 # SORDO: pitch = 0
        
    # Guardar parámetros
    data_lpc.append((a, G, fp))


# --- Decodificación LPC ---
signal_recons = np.zeros(len(signal_audio_padded))

for i, (a, G, fp) in enumerate(data_lpc):
    inicio = i * hop
    
    # Generar excitación E
    if fp > 0: # SONORO (Tren de pulsos)
        excitacion = gen_pulsos(fp, L, fs)
    else: # SORDO (Ruido blanco)
        excitacion = np.random.randn(L)
    
    # Escalar excitación por la Ganancia G
    excitacion *= G
    
    # Filtrar con modelo LPC: H(z) = G / (1 + sum(a_k * z^-k))
    # En Python, signal.lfilter(b, a, x) usa H(z) = B(z)/A(z) donde A(z) = 1 + sum(a_k * z^-k)
    # Por lo tanto, b debe ser [1] y a debe ser [1, a1, a2, ...]
    a_lfilter = np.concatenate(([1], -a)) # El filtro de síntesis tiene coeficientes *negativos*
    segmento_sint = signal.lfilter([1], a_lfilter, excitacion)
    
    # Overlap-add (método para reconstrucción con solapamiento)
    signal_recons[inicio:inicio+L] += segmento_sint * ventana


# --- Reproducir/guardar ---
# Recortar el padding al final y normalizar
# --- Reemplaza la sección de Reproducir/Guardar con esto ---

# 1. Recortar el padding al final
signal_recons_final = signal_recons[:len(signal_audio)]

# 2. Normalizar la amplitud máxima a 1.0
max_abs = np.max(np.abs(signal_recons_final))
if max_abs > 1e-10: # Evitar división por cero si es silencio total
    signal_recons_norm = signal_recons_final / max_abs
else:
    signal_recons_norm = signal_recons_final # Silencio

# 3. Escalar al rango de 16 bits (32767) y convertir a entero
audio_int16 = (signal_recons_norm * 32767).astype(np.int16)

# Guardar como WAV
wav.write('audios_reconstruidos\\audio_01-reconstruido_3a.wav', fs, audio_int16)


# --- Generación de Espectrogramas ---

# 1. Espectrograma de la Señal Original
plot_spectrogram(
    signal_audio, 
    fs, 
    'Espectrograma de la señal original', 
    'espectrograma_original_3a.png'
)

# 2. Espectrograma de la Señal Reconstruida
plot_spectrogram(
    signal_recons_final, 
    fs, 
    'Espectrograma de la señal reconstruida (3a)', 
    'espectrograma_reconstruido_3a.png'
)

# Reproducir si estás en Colab/Jupyter
# Audio(signal_recons_final, rate=fs)
print("Reconstrucción para el Ejercicio 3a (Pitch estimado) completada y guardada.")