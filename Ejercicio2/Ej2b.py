# ==============================================================================
# EJERCICIO 2B: Síntesis y Comparación de Fonemas Aislados con LPC
# ==============================================================================
#
# OBJETIVO: Este script demuestra la síntesis del habla (fonemas) usando el modelo
# Fuente-Filtro de LPC. Para un fonema dado ('a.wav', 's.wav', etc.):
# 1. Analiza el audio original para obtener los coeficientes LPC (Filtro).
# 2. Detecta el Pitch (f_p) para determinar la excitación (Fuente).
# 3. Sintetiza la señal generando una excitación (pulsos o ruido) y filtrándola.
# 4. Compara el Periodograma de la señal original y la sintetizada para validar 
#    la fidelidad del modelo LPC para diferentes órdenes (P).
#
# Requiere que el archivo 'Ej2a.py' contenga las funciones 'param_lpc' y 
# 'pitch_lpc' necesarias.
#
# ==============================================================================


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Importar funciones de ejercicios anteriores
from Ej2a import param_lpc, pitch_lpc

# Ruta a la carpeta del script actual
base_dir = os.path.dirname(os.path.abspath(__file__))

# Carpeta donde están los audios
ruta_audios = os.path.join(base_dir, "..", "Ejercicio1", "AudiosEj1")

# Carpeta donde se guardarán las imágenes generadas
ruta_imagenes = os.path.join(base_dir, "Imagenes_Ej2b")
os.makedirs(ruta_imagenes, exist_ok=True)

# --- EJERCICIO 2B ---

def calcular_periodograma(x, fs):
    """Calcula el periodograma de una señal."""
    N = len(x)
    X = np.fft.fft(x)
    Pxx = (1/N) * np.abs(X)**2
    Pxx = Pxx[:N//2 + 1]
    Pxx[1:-1] *= 2
    Pxx /= fs
    f = np.linspace(0, fs/2, len(Pxx))
    return f, Pxx

def generar_excitacion(N, fs, fp):
    """
    Genera la señal de excitación U(n).
    
    Parámetros
    ----------
    N : int
        Número de muestras a generar.
    fs : float
        Frecuencia de muestreo [Hz].
    fp : float
        Frecuencia de pitch [Hz]. Si fp=0, genera ruido blanco.
    
    Retorna
    -------
    U : ndarray
        Señal de excitación (tren de impulsos o ruido blanco).
    """
    if fp > 0:
        # SONORO: Tren de impulsos periódico
        Np = int(fs / fp)  # Periodo en muestras
        U = np.zeros(N)
        U[::Np] = np.sqrt(Np * fs)  # Normalización
    else:
        # SORDO: Ruido blanco gaussiano
        U = np.random.randn(N)
    
    return U

def sintetizar_lpc(a, G, N, fs, fp):
    """
    Sintetiza una señal usando el modelo LPC.
    
    Parámetros
    ----------
    a : ndarray
        Coeficientes LPC [a1, a2, ..., aP].
    G : float
        Ganancia del modelo.
    N : int
        Número de muestras a generar.
    fs : float
        Frecuencia de muestreo [Hz].
    fp : float
        Frecuencia de pitch [Hz] (0 para señal sorda).
    
    Retorna
    -------
    x_sintetico : ndarray
        Señal sintetizada.
    """
    P = len(a)
    
    # Generar excitación
    U = generar_excitacion(N, fs, fp)
    
    # Filtro IIR: X(n) = sum(a[k]*X(n-k)) + G*U(n)
    x_sintetico = np.zeros(N)
    for n in range(N):
        x_sintetico[n] = G * U[n]
        for k in range(min(P, n)):
            x_sintetico[n] += a[k] * x_sintetico[n - k - 1]
    
    return x_sintetico

def comparar_periodogramas_superpuestos(x_original, x_sintetico, fs, nombre_archivo, P, fp):
    """
    Compara los periodogramas del original y sintético superpuestos.
    ESTA ES LA FUNCIÓN QUE CUMPLE CON LA CONSIGNA 2B.
    
    Parámetros
    ----------
    x_original : ndarray
        Señal original.
    x_sintetico : ndarray
        Señal sintetizada.
    fs : float
        Frecuencia de muestreo.
    nombre_archivo : str
        Nombre del archivo.
    P : int
        Orden del modelo LPC usado.
    fp : float
        Frecuencia de pitch detectada.
    """
    # Calcular periodogramas
    f_orig, Pxx_orig = calcular_periodograma(x_original, fs)
    f_sint, Pxx_sint = calcular_periodograma(x_sintetico, fs)
    
    # Convertir a dB
    Pxx_orig_db = 10 * np.log10(Pxx_orig + 1e-12)
    Pxx_sint_db = 10 * np.log10(Pxx_sint + 1e-12)
    
    # Crear figura
    plt.figure(figsize=(12, 6))
    plt.plot(f_orig, Pxx_orig_db, label='Periodograma Original', 
             color='steelblue', linewidth=1.5, alpha=0.8)
    plt.plot(f_sint, Pxx_sint_db, label='Periodograma Sintético', 
             color='crimson', linewidth=1.5, linestyle='--', alpha=0.8)
    
    plt.xlabel('Frecuencia [Hz]', fontsize=11)
    plt.ylabel('Densidad espectral de potencia [dB/Hz]', fontsize=11)
    
    tipo = "SONORO" if fp > 0 else "SORDO"
    plt.title(f"Comparación de Periodogramas: '{nombre_archivo}' (P={P}, fp={fp:.1f} Hz, {tipo})", 
              fontsize=12, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, fs/2])
    plt.tight_layout()
    
    # Guardar figura
    nombre_base = nombre_archivo.replace(".wav", "")
    nombre_archivo_img = os.path.join(ruta_imagenes, f"Periodogramas_superpuestos_{nombre_base}_P{P}.png")
    plt.savefig(nombre_archivo_img, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  └─ Periodogramas superpuestos guardados: {nombre_archivo_img}")

# --- Helper para leer y normalizar ---
def leer_wav_mono_norm(path):
    fs, x = wavfile.read(path)
    x = x.astype(float)
    if x.ndim > 1:
        x = x[:, 0]
    x /= (np.max(np.abs(x)) + 1e-12)
    return fs, x

def procesar_fonema_completo(nombre_archivo, ordenes_P=[10, 30, 50], alpha=0.3):
    """
    Procesa un fonema: sintetiza y compara con múltiples valores de P.
    
    Parámetros
    ----------
    nombre_archivo : str
        Nombre del archivo .wav.
    ordenes_P : list
        Lista de órdenes P a probar.
    alpha : float
        Umbral de detección de pitch.
    """
    print(f"\n{'='*60}")
    print(f"Procesando: {nombre_archivo}")
    print(f"{'='*60}")
    
    # Cargar audio
    ruta_completa = os.path.join(ruta_audios, nombre_archivo)
    fs, x_original = leer_wav_mono_norm(ruta_completa)
    N = len(x_original)
    
    print(f"  • Frecuencia de muestreo: {fs} Hz")
    print(f"  • Duración: {N/fs:.3f} s ({N} muestras)")
    
    # Procesar para cada orden P
    for P in ordenes_P:
        print(f"\n  >> Analizando con P = {P}")
        
        # Calcular coeficientes LPC
        a, G = param_lpc(x_original, P)
        print(f"     • Ganancia G = {G:.6f}")
        
        # Detectar pitch
        fp = pitch_lpc(x_original, a, alpha, fs)
        
        if fp > 0:
            print(f"     • Pitch detectado: {fp:.2f} Hz → SONORO")
        else:
            print(f"     • Sin pitch detectado → SORDO")
        
        # Sintetizar señal
        x_sintetico = sintetizar_lpc(a, G, N, fs, fp)
        x_sintetico /= (np.max(np.abs(x_sintetico)) + 1e-12)
        
        print(f"     • Señal sintetizada")
        
        # CUMPLIR CON LA CONSIGNA 2B: Comparar periodogramas superpuestos
        comparar_periodogramas_superpuestos(x_original, x_sintetico, fs, nombre_archivo, P, fp)
    
    print(f"\n  ✓ Procesamiento completado para '{nombre_archivo}'")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("EJERCICIO 2B - Síntesis y Comparación de Fonemas con LPC")
    print("="*60)
    
    # Procesar vocal y consonante con múltiples valores de P
    procesar_fonema_completo("a.wav", ordenes_P=[10, 30, 50], alpha=0.3)
    procesar_fonema_completo("s.wav", ordenes_P=[10, 30, 50], alpha=0.3)
    
    print(f"\n{'='*60}")
    print(f"✓ Todas las imágenes guardadas en: {ruta_imagenes}")
    print(f"{'='*60}\n")
