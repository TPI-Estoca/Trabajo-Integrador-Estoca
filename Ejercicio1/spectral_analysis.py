# ==============================================================================
# SPECTRAL ANALYSIS SCRIPT (Exercise 1b - Enhanced)
# ==============================================================================
#
# OBJETIVO: Realizar análisis espectral completo de un archivo de audio:
# 1. Cargar archivo WAV
# 2. Calcular Periodograma (espectro real)
# 3. Calcular PSD del modelo LPC
# 4. Mostrar ambos en el dominio frecuencial (subplot superior, más grande)
# 5. Mostrar señal en dominio temporal (subplot inferior, más pequeño)
# 6. Guardar imagen en Ejercicio1/
#
# USO:
#   python spectral_analysis.py e.wav
#   python spectral_analysis.py a.wav
#
# ==============================================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.linalg import toeplitz

# Ruta a la carpeta del script actual
base_dir = os.path.dirname(os.path.abspath(__file__))

# Carpeta donde están los audios
ruta_audios = os.path.join(base_dir, "AudiosEj1")

# --- HELPER FUNCTIONS ---

def leer_wav_mono_norm(path):
    """Lee archivo WAV, convierte a mono y normaliza."""
    fs, x = wavfile.read(path)
    x = x.astype(float)
    if x.ndim > 1:
        x = x[:, 0]
    x /= (np.max(np.abs(x)) + 1e-12)
    return fs, x

def calcular_periodograma(x, fs):
    """Calcula el Periodograma (espectro real)."""
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

    # S_U(ω) para ruido blanco (aproximación simple)
    S_U = 1.0

    # Calculamos S_X según fórmula (7)
    S_X = (G**2 / denominador_abs2) * S_U

    return S_X

def param_lpc(xs, P):
    """
    Calcula los parámetros LPC (coeficientes y ganancia del error de predicción)
    usando el método de autocorrelación de Yule-Walker.
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

def analizar_espectro(nombre_archivo, P=10):
    """
    Realiza análisis espectral completo y crea figura con dos subplots.
    
    Parámetros:
    -----------
    nombre_archivo : str
        Nombre del archivo (e.g., 'e.wav')
    P : int
        Orden del modelo LPC (default=10)
    """
    
    # 1. Cargar audio
    ruta_completa = os.path.join(ruta_audios, nombre_archivo)
    if not os.path.exists(ruta_completa):
        print(f"Error: Archivo '{ruta_completa}' no encontrado.")
        sys.exit(1)
    
    fs, x = leer_wav_mono_norm(ruta_completa)
    print(f"✓ Audio cargado: {nombre_archivo} (fs={fs} Hz, {len(x)} muestras)")
    
    # 2. Calcular parámetros LPC
    a, G = param_lpc(x, P)
    print(f"✓ Parámetros LPC calculados: P={P}, G={G:.6f}")
    
    # 3. Calcular Periodograma
    f, Pxx = calcular_periodograma(x, fs)
    Pxx_db = 10 * np.log10(Pxx + 1e-12)
    print(f"✓ Periodograma calculado")
    
    # 4. Calcular PSD LPC
    psd_lpc = calcular_psd_lpc(a, G, fs, P, f)
    psd_lpc_db = 10 * np.log10(psd_lpc + 1e-12)
    
    # 5. Ajuste de escala para PSD LPC
    max_psd_lpc_db = np.max(psd_lpc_db)
    max_periodograma_db = np.max(Pxx_db)
    psd_lpc_db -= (max_psd_lpc_db - max_periodograma_db)
    print(f"✓ PSD LPC calculada y ajustada")
    
    # 6. Crear figura con dos subplots (frecuencia más grande que tiempo)
    fig = plt.figure(figsize=(14, 10))
    
    # GridSpec para controlar el tamaño de los subplots
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 0.8], hspace=0.35)
    
    # Subplot 1: Análisis espectral (más grande)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(f, Pxx_db, label='Periodograma (original)', alpha=0.7, linewidth=1.5)
    ax1.plot(f, psd_lpc_db, label=f'PSD (modelo LPC, P={P})', color='r', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Frecuencia [Hz]', fontsize=11)
    ax1.set_ylabel('Densidad espectral de potencia [dB/Hz]', fontsize=11)
    ax1.set_title(f'Análisis Espectral - {nombre_archivo} (fs={fs} Hz)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Señal en el dominio temporal (más pequeño)
    ax2 = fig.add_subplot(gs[1])
    t = np.arange(len(x)) / fs  # eje de tiempo en segundos
    ax2.plot(t, x, color='steelblue', linewidth=0.8)
    ax2.set_xlabel('Tiempo [s]', fontsize=11)
    ax2.set_ylabel('Amplitud normalizada', fontsize=11)
    ax2.set_title('Señal en el dominio temporal', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 7. Guardar imagen
    nombre_sin_ext = os.path.splitext(nombre_archivo)[0]
    filename_salida = os.path.join(base_dir, f"spectral_analysis_{nombre_sin_ext}_P{P}.png")
    plt.savefig(filename_salida, dpi=300, bbox_inches='tight')
    print(f"✓ Imagen guardada: {filename_salida}")
    
    plt.show()

# --- MAIN ---
if __name__ == "__main__":
    
    # Obtener nombre de archivo desde argumentos o usar default
    if len(sys.argv) > 1:
        nombre_archivo = sys.argv[1]
    else:
        nombre_archivo = "e.wav"
    
    # Orden LPC (opcional, desde argumentos)
    P = 10
    if len(sys.argv) > 2:
        try:
            P = int(sys.argv[2])
        except ValueError:
            print(f"Advertencia: Orden LPC inválido, usando P={P}")
    
    print(f"\n{'='*60}")
    print(f"SPECTRAL ANALYSIS - Exercise 1b")
    print(f"{'='*60}")
    
    analizar_espectro(nombre_archivo, P=P)
    
    print(f"\n{'='*60}")
    print("✓ Análisis completado")
    print(f"{'='*60}\n")
