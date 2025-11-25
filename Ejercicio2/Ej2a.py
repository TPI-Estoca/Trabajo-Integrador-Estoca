# ==============================================================================
# EJERCICIO 2A: Detección de Pitch (Frecuencia Fundamental) mediante LPC
# ==============================================================================
#
# OBJETIVO: Este script implementa el algoritmo de detección de Pitch (f_p)
# para segmentos de audio. Utiliza el método de Autocorrelación de la SEÑAL 
# de entrada y lo combina con la estimación de coeficientes LPC (param_lpc) 
# para cumplir con el requisito del ejercicio.
# 
# La función principal analiza fonemas (vocales y consonantes) y prueba 
# diferentes umbrales (alpha) para determinar si son SONOROS (con pitch) o 
# SORDOS (sin pitch, solo ruido), guardando los resultados en un archivo TXT.
#
# ==============================================================================

import os
from scipy.io import wavfile
from scipy.linalg import toeplitz
import numpy as np

# Para asegurar que el script sepa dónde buscar los archivos de audio
# NOTA: Ajusta la ruta a 'ruta_audios' si tu estructura de carpetas es diferente
base_dir = os.path.dirname(os.path.abspath(__file__))
# Asume que los audios están en la carpeta 'AudiosEj1' un nivel arriba
ruta_audios = os.path.join(base_dir, "..", "Ejercicio1", "AudiosEj1")

# --- EJERCICIO 1: FUNCIÓN DE ESTIMACIÓN DE PARÁMETROS LPC ---
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

# --- EJERCICIO 2A: FUNCIÓN DE DETECCIÓN DE PITCH (MODIFICADA) ---
def pitch_lpc(xs, a, alpha, fs):
    """
    Detecta el pitch del segmento usando autocorrelación de la SEÑAL (método simple).
    
    NOTA: En esta versión simplificada, los coeficientes 'a' no se utilizan, 
    ya que se calcula la autocorrelación de la señal de entrada 'xs' y no del residuo.
    
    xs: segmento de señal (completo, no segmentado)
    a: (IGNORADO) coeficientes LPC
    alpha: umbral de decisión (0 < alpha < 1)
    fs: frecuencia de muestreo [Hz]
    Retorna: fp, frecuencia de pitch [Hz] (0 si no se detecta pitch periódico)
    """
    xs = np.asarray(xs, dtype=float)
    N = len(xs)
    
    # Rango de pitch: 50 Hz (max lag) a 500 Hz (min lag)
    min_lag = int(fs / 500)
    max_lag = int(fs / 50)
    
    if N < max_lag or N < min_lag:
        return 0
    
    r0 = np.dot(xs, xs) # Energía en lag 0
    if r0 < 1e-10: # Silencio
        return 0
    
    # Autocorrelación normalizada en el rango de lags (basada en la señal xs)
    r_full = np.correlate(xs, xs, mode='full')
    # Extraemos el rango de lags relevante: indices de N-1+min_lag a N-1+max_lag
    r_lags = r_full[N - 1 + min_lag : N - 1 + max_lag + 1]
    
    acf_norm = r_lags / r0
    
    # Encontrar el máximo (el segundo máximo, excluyendo lag 0)
    max_idx = np.argmax(acf_norm)
    max_acf = acf_norm[max_idx]
    pitch_lag = max_idx + min_lag
    
    # Decisión sonoro/sordo basada en el umbral alpha
    if max_acf > alpha:
        # SONORO: retornar pitch detectado
        fp = fs / pitch_lag
        return fp
    else:
        # SORDO: sin pitch periódico
        return 0

# --- Helper para leer y normalizar (igual que en Ejercicio 1) ---
def leer_wav_mono_norm(path):
    fs, x = wavfile.read(path)
    x = x.astype(float)
    if x.ndim > 1:
        x = x[:,0]
    x /= (np.max(np.abs(x)) + 1e-12)
    return fs, x

def analizar_pitch_wav(nombre_archivo, ordenes_P=[10], alphas=[0.3, 0.5, 0.7], archivo_salida="Resultados_Ej2a.txt"):
    """
    Detecta el pitch de un archivo WAV usando LPC.
    """
    # Construye la ruta completa al archivo de audio
    ruta_completa = os.path.join(ruta_audios, nombre_archivo)
    fs, x = leer_wav_mono_norm(ruta_completa)
    
    archivo_salida = os.path.join(base_dir, archivo_salida)
    
    # Bloque para imprimir y guardar datos en txt
    header = f"\n=== {nombre_archivo} (fs={fs} Hz) ===\n"
    print(header)
    
    # Abrimos el archivo en modo append para ir acumulando resultados
    with open(archivo_salida, "a", encoding="utf-8") as f:
        f.write(header)
        
        # Probar diferentes órdenes P
        for P in ordenes_P:
            # Calcular coeficientes LPC
            # NOTA: Aunque el pitch_lpc simplificado no usa 'a', 
            # param_lpc debe ejecutarse para cumplir el requisito de P
            a, G = param_lpc(x, P)
            
            subheader = f"  Orden P={P} (G={G:.6f}):\n"
            print(subheader)
            f.write(subheader)
            
            # Probar diferentes valores de alpha
            for alpha in alphas:
                # La función pitch_lpc ahora usa la autocorrelación de 'x'
                # y realiza la clasificación con 'alpha'
                fp = pitch_lpc(x, a, alpha, fs)
                
                if fp > 0:
                    tipo = "SONORO"
                    linea = f"    alpha={alpha:.2f} | Pitch: {fp:6.2f} Hz ({tipo})\n"
                else:
                    tipo = "SORDO"
                    linea = f"    alpha={alpha:.2f} | Sin pitch detectado ({tipo})\n"
                
                print(linea, end="")
                f.write(linea)
    
    return fs, x

#----------CARGA DE AUDIOS Y ANÁLISIS-----------
if __name__ == "__main__":
    nombres = ["a", "e", "s", "sh"]
    
    # Limpiar archivo de salida al inicio
    archivo_salida = os.path.join(base_dir, "Resultados_Ej2a.txt")
    with open(archivo_salida, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("EJERCICIO 2A - Detección de Pitch con LPC\n")
        f.write("=" * 60 + "\n")
    
    # Usaremos P=10, un valor típico para fs=8000 Hz.
    ordenes_P = [10]
    
    # Analizar cada audio
    for nombre in nombres:
        analizar_pitch_wav(f"{nombre}.wav", ordenes_P=ordenes_P, alphas=[0.3, 0.5, 0.7])
    
    print(f"\n✓ Resultados guardados en: {archivo_salida}")

"""
P (Orden del Modelo LPC):
Representa el número de coeficientes del filtro AR. Un valor típico de P para la voz humana 
es de 10-14. Lo mantenemos en 10 para este ejercicio.

Alpha (Umbral de Pitch):
Es el umbral de correlación normalizada que determina si un segmento es sonoro o sordo. 
Un valor más alto hace el criterio más estricto. Valores comunes suelen estar entre 0.3 y 0.7.
"""