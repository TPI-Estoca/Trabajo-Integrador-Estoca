# Cargo las librebrerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

from Ej1a import param_lpc

# Ruta de los audios
ruta_audios = r"C:\Users\ecava\OneDrive\Documents\Facultad Emi\2C2025\ESTOCA\TP Integrador\AudiosEj1"

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

def comparar_psd(x, fs, a, G, P):
    # Compara el periodograma de la señal original con la PSD LPC.
    # 1. Periodograma original
    f, Pxx = calcular_periodograma(x, fs)
    Pxx_db = 10 * np.log10(Pxx + 1e-12)  # Convertimos a dB

    # 2. PSD del modelo LPC
    psd_lpc = calcular_psd_lpc(a, G, fs, P, f)
    psd_lpc_db = 10 * np.log10(psd_lpc + 1e-12)  # Convertimos a dB

    # 3. Aseguramos que las escalas están bien alineadas
    # Normalizamos las escalas para que la comparación sea válida
    max_psd_lpc_db = np.max(psd_lpc_db)
    max_periodograma_db = np.max(Pxx_db)

    # Ajustamos la escala de la PSD LPC para que tenga la misma magnitud que el periodograma
    psd_lpc_db -= (max_psd_lpc_db - max_periodograma_db)

    # 4. Graficamos ambos
    plt.figure(figsize=(10, 6))
    plt.plot(f, Pxx_db, label='Periodograma (original)', alpha=0.7)
    plt.plot(f, psd_lpc_db, label='PSD (modelo LPC)', color='r', linestyle='--')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Densidad espectral de potencia (dB/Hz)')
    plt.title('Comparación entre Periodograma y PSD LPC')
    plt.legend()
    plt.grid(True)
    plt.show()

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
    plt.show()

if __name__ == "__main__":
    fa, xa = leer_wav_mono_norm(r"C:\Users\ecava\OneDrive\Documents\Facultad Emi\2C2025\ESTOCA\TP Integrador\AudiosEj1\a.wav")
    fe, xe = leer_wav_mono_norm(r"C:\Users\ecava\OneDrive\Documents\Facultad Emi\2C2025\ESTOCA\TP Integrador\AudiosEj1\e.wav")
    fs, xs = leer_wav_mono_norm(r"C:\Users\ecava\OneDrive\Documents\Facultad Emi\2C2025\ESTOCA\TP Integrador\AudiosEj1\s.wav")
    fsh, xsh = leer_wav_mono_norm(r"C:\Users\ecava\OneDrive\Documents\Facultad Emi\2C2025\ESTOCA\TP Integrador\AudiosEj1\sh.wav")

    # Gráfico temporal
    graficar_respuesta_temporal(xa, fa, titulo="Señal 'a.wav' en el tiempo")
    graficar_respuesta_temporal(xe, fe, titulo="Señal 'e.wav' en el tiempo")
    graficar_respuesta_temporal(xs, fs, titulo="Señal 's.wav' en el tiempo")
    graficar_respuesta_temporal(xsh, fsh, titulo="Señal 'sh.wav' en el tiempo")

    # Calcular LPC para los diferentes P
    P = 5

    aa, Ga = param_lpc(xa, P)
    ae, Ge = param_lpc(xe, P)
    ass, Gs = param_lpc(xs, P)
    ash, Gsh = param_lpc(xsh, P)

    # Comparar PSD y Periodograma
    comparar_psd(xa, fa, aa, Ga, P)
    comparar_psd(xe, fe, ae, Ge, P)
    comparar_psd(xs, fs, ass, Gs, P)
    comparar_psd(xsh, fsh, ash, Gsh, P)