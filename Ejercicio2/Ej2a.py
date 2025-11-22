# Cargo las librerías necesarias
import os
import sys
from scipy.io import wavfile
from scipy.linalg import toeplitz
import numpy as np

# Ruta a la carpeta del script actual (Ejercicio2)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Carpeta donde están los audios
ruta_audios = os.path.join(base_dir, "..", "Ejercicio1", "AudiosEj1")

# --- EJERCICIO 2A ---
# Empleo la función del Ejercicio 1 para obtener los coeficientes LPC
def param_lpc(x, P):
    x = np.asarray(x, dtype=float)
    N = len(x)
    z = np.array([np.dot(x[:N-k], x[k:]) / (N-k) for k in range(P+1)])
    
    R = toeplitz(z[:P])
    r = z[1:P+1]
    
    try:
        a = np.linalg.solve(R, r)
    except:
        a = np.linalg.lstsq(R, r, rcond=None)[0]
    
    G2 = z[0] - np.dot(a, r)
    G = np.sqrt(max(G2, 1e-12))
    return a, G

# Implemento la función de detección de pitch usando LPC (basándome en la implementada en el Ejercicio 3)
def pitch_lpc(xs, a, alpha, fs):
    """
    Detecta el pitch del segmento usando autocorrelación del residuo.
    xs: segmento de señal (ya ventaneado)
    a: coeficientes LPC [a1, a2, ..., aP]
    alpha: umbral de decisión (0 < alpha < 1)
    fs: frecuencia de muestreo [Hz]
    Retorna: fp, frecuencia de pitch [Hz] (0 si no se detecta pitch periódico)
    """
    xs = np.asarray(xs, dtype=float)
    a = np.asarray(a, dtype=float)
    N = len(xs)
    P = len(a)
    
    # Calcular el residuo e(n) = x(n) - sum(a[k]*x(n-k))
    e = np.zeros(N)
    for n in range(N):
        e[n] = xs[n]
        for k in range(min(P, n)):
            e[n] -= a[k] * xs[n - k - 1]
    
    # Rango de pitch: 50 Hz (max lag) a 500 Hz (min lag)
    min_lag = int(fs / 500)
    max_lag = int(fs / 50)
    
    if N < max_lag or N < min_lag:
        return 0
    
    r0 = np.dot(e, e)  # Energía en lag 0
    if r0 < 1e-10:  # Silencio
        return 0
    
    # Autocorrelación normalizada del residuo en el rango de lags
    r_full = np.correlate(e, e, mode='full')
    # Extraemos el rango de lags relevante: indices de N-1+min_lag a N-1+max_lag
    r_lags = r_full[N - 1 + min_lag : N - 1 + max_lag + 1]
    
    acf_norm = r_lags / r0
    
    # Encontrar el segundo máximo (el primero sería en lag 0)
    max_idx = np.argmax(acf_norm)
    max_acf = acf_norm[max_idx]
    pitch_lag = max_idx + min_lag
    
    # Decisión sonoro/sordo basada en el umbral
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

# Ver el tema del valor del alpha y el orden de P!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def analizar_pitch_wav(nombre_archivo, ordenes_P=[10], alphas=[0.3, 0.5, 0.7], archivo_salida="Resultados_Ej2a.txt"):
    """
    Detecta el pitch de un archivo WAV usando LPC.
    Parámetros
    ----------
    nombre_archivo : str
        Nombre del archivo .wav a analizar.
    ordenes_P : list[int]
        Lista de órdenes LPC a probar (por defecto [10]).
    alphas : list[float]
        Lista de umbrales alpha a probar (por defecto [0.3, 0.5, 0.7]).
    archivo_salida : str
        Nombre del archivo de salida para guardar resultados.
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
            a, G = param_lpc(x, P)
            
            subheader = f"  Orden P={P} (G={G:.6f}):\n"
            print(subheader)
            f.write(subheader)
            
            # Probar diferentes valores de alpha
            for alpha in alphas:
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
    
    # Puedes elegir qué valores de P probar:
    # Opción 1: Solo P=10 (recomendado para el ejercicio 2a)
    ordenes_P = [10]
    
    # Opción 2: Si quieres comparar con los 3 valores del ejercicio 1
    # ordenes_P = [5, 10, 30]
    
    # Analizar cada audio
    for nombre in nombres:
        analizar_pitch_wav(f"{nombre}.wav", ordenes_P=ordenes_P, alphas=[0.3, 0.5, 0.7])
    
    print(f"\n✓ Resultados guardados en: {archivo_salida}")

"""
P corresponde al orden del modelo LPC. Representa el número de coeficientes del filtro AR
utilizados para modelar la señal. Define cuántos polos tendrá la función de transferencia
del filtro LPC. Se debe buscar un equilibrio: un P muy bajo puede no capturar adecuadamente
las características de la señal (poco detalle), mientras que un P muy alto puede llevar
a un sobreajuste (captura ruido innecesario). Un valor típico de P para la voz humana es de
10- 14.
Alpha es un umbral de decisión utilizado en la detección de pitch. Es el umbral de correlación
normalizada que determina si un segmento de señal es considerado sonoro (con pitch periódico)
o sordo (sin pitch). Un valor más alto de alpha hace que el criterio sea más estricto, detectando
unicamente pitches muy claros; mientras que un valor más bajo permite detectar pitches incluso en
señales con menos periodicidad. Valores comunes de alpha suelen estar entre 0.3 y 0.7.
La fs viene determinada directamente en los audios cargados.
"""