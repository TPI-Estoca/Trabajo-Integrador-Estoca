# ==============================================================================
# EJERCICIO 1B: Comparación de Espectros (Periodograma vs. PSD LPC)
# ==============================================================================
#
# OBJETIVO: Este script compara la Densidad Espectral de Potencia (PSD) 
# paramétrica estimada por el modelo LPC con el Periodograma (espectro real) 
# de los fonemas aislados.
#
# El LPC solo modela la ENVOLVENTE espectral (los Formantes). Al superponer 
# ambos gráficos, se demuestra la precisión del modelo All-Pole para diferentes 
# órdenes (P) de predicción.
#
# Requiere que el archivo 'Ej1a.py' contenga la función 'param_lpc'.
#
# ==============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

from Ej1a import param_lpc

# Ruta a la carpeta del script actual
base_dir = os.path.dirname(os.path.abspath(__file__))

# Carpeta donde están los audios (por ejemplo: ./AudiosEj1)
ruta = os.path.join(base_dir, "AudiosEj1")

# Carpeta donde se guardarán las imágenes generadas
ruta_imagenes = os.path.join(base_dir, "Imagenes_Ej1b")
os.makedirs(ruta_imagenes, exist_ok=True)

# --- EJERCICIO 1B---
def calcular_periodograma(x, fs):
    N = len(x)
    X = np.fft.fft(x)
    Pxx = (1/N) * np.abs(X)**2
    Pxx = Pxx[:N//2 + 1]
    Pxx[1:-1] *= 2
    Pxx /= fs
    f = np.linspace(0, fs/2, len(Pxx))
    return f, Pxx

def calcular_psd_lpc(a, G, fs, P, f):
    """Calcula la PSD del modelo LPC según la fórmula (7)."""
    omega = 2 * np.pi * f / fs  # Frecuencias angulares normalizadas

    # Calculamos el denominador: |1 - sum(a_k * e^(-jωk))|^2
    denominador = np.ones_like(omega, dtype=complex)
    for k in range(1, P+1):
        denominador -= a[k-1] * np.exp(-1j * omega * k)

    # Módulo al cuadrado del denominador
    denominador_abs2 = np.abs(denominador)**2

    # S_U(ω) para ruido blanco a 200 Hz (frecuencia de pitch)
    # Para señales sonoras, podemos aproximar S_U como constante o usar
    # una delta de Dirac en f0 = 200 Hz
    # Opción simple: S_U = 1 (ruido blanco normalizado)
    S_U = 1.0

    # Calculamos S_X según fórmula (7)
    S_X = (G**2 / denominador_abs2) * S_U

    return S_X

def comparar_psd(x, fs, a, G, P, titulo="Comparación entre Periodograma y PSD LPC"):
    # 1. Periodograma original
    f, Pxx = calcular_periodograma(x, fs)
    Pxx_db = 10 * np.log10(Pxx + 1e-12)

    # 2. PSD del modelo LPC
    psd_lpc = calcular_psd_lpc(a, G, fs, P, f)
    psd_lpc_db = 10 * np.log10(psd_lpc + 1e-12)

    # 3. Ajuste de escala
    max_psd_lpc_db = np.max(psd_lpc_db)
    max_periodograma_db = np.max(Pxx_db)
    psd_lpc_db -= (max_psd_lpc_db - max_periodograma_db)

    # 4. Gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(f, Pxx_db, label='Periodograma (original)', alpha=0.7)
    plt.plot(f, psd_lpc_db, label='PSD (modelo LPC)', color='r', linestyle='--')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Densidad espectral de potencia (dB/Hz)')
    plt.title(titulo)
    plt.legend()
    plt.grid(True)

    # --- Guardar figura ---
    nombre_sin_espacios = titulo.replace(" ", "_").replace(":", "")
    nombre_archivo = os.path.join(ruta_imagenes, f"{nombre_sin_espacios}.png")
    plt.savefig(nombre_archivo, dpi=300)
    plt.close()  # cierra la figura para evitar acumulación en memoria

# --- Helper para leer y normalizar ---
def leer_wav_mono_norm(path):
    fs, x = wavfile.read(path)
    x = x.astype(float)
    if x.ndim > 1:
        x = x[:, 0]
    x /= (np.max(np.abs(x)) + 1e-12)
    return fs, x

def graficar_respuesta_temporal(x, fs, titulo="Señal en el tiempo"):
    """Grafica la señal de audio en el dominio temporal."""
    t = np.arange(len(x)) / fs  # eje de tiempo en segundos
    plt.figure(figsize=(10, 4))
    plt.plot(t, x, color='steelblue')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud normalizada')
    plt.title(titulo)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_imagenes, f"Temporal_{nombre}.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    # --- Parámetros generales ---
    nombres = ["a", "e", "s", "sh"]
    ordenes_P = [5, 10, 30]  # distintos órdenes LPC a analizar

    # --- Procesamiento de cada audio ---
    for nombre in nombres:
        fs, x = leer_wav_mono_norm(os.path.join(ruta, f"{nombre}.wav"))
        graficar_respuesta_temporal(x, fs, titulo=f"Señal '{nombre}.wav' en el tiempo")

        # Para cada valor de P, calculamos y comparamos PSD
        for P in ordenes_P:
            a, G = param_lpc(x, P)
            comparar_psd(
                x,
                fs,
                a,
                G,
                P,
                titulo=f"Comparación PSD: señal '{nombre}' (P={P})"
            )