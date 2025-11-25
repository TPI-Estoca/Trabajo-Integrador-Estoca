# ==============================================================================
# EJERCICIO 1A: Estimación de Parámetros LPC (a y G)
# ==============================================================================
#
# OBJETIVO: Implementar la función param_lpc(xs, P) que calcula los coeficientes 
# de predicción lineal (a_k) y la ganancia del error (G) para un segmento 
# de señal, utilizando el método de Autocorrelación de Yule-Walker.
# 
# NOTA: Se utiliza la autocorrelación SESGADA (np.correlate sin división por N-k) 
# para mantener la estabilidad numérica.
#
# ==============================================================================


import os
from scipy.io import wavfile
from scipy.linalg import toeplitz
import numpy as np

# Ruta a la carpeta del script actual
base_dir = os.path.dirname(os.path.abspath(__file__))

# Carpeta donde están los audios (por ejemplo: ./AudiosEj1)
ruta_audios = os.path.join(base_dir, "AudiosEj1")

# --- EJERCICIO 1A---
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

# --- Helper para leer y normalizar ---
def leer_wav_mono_norm(path):
    fs, x = wavfile.read(path)
    x = x.astype(float)
    if x.ndim > 1:
        x = x[:,0]
    x /= (np.max(np.abs(x)) + 1e-12)
    return fs, x

def analizar_archivo_wav(nombre_archivo, ordenes=[5, 10, 30], archivo_salida="Resultados_Ej1a.txt"):
    """
    Calcula los coeficientes LPC y la ganancia para un archivo WAV dado.
    Parámetros
    ----------
    nombre_archivo : str
        Nombre o ruta del archivo .wav a analizar.
    ordenes : list[int]
        Lista de órdenes LPC a probar (por defecto [5, 10, 30]).
    """
    # Construye la ruta completa al archivo de audio
    ruta_completa = os.path.join(ruta_audios, nombre_archivo)
    fs, x = leer_wav_mono_norm(ruta_completa)
    
    archivo_salida = os.path.join(base_dir, archivo_salida)
    
    # Bloque para imprimir y guardar datos en txt
    # Creamos el texto a imprimir/guardar
    header = f"\n=== {nombre_archivo} ===\n"
    print(header)

    # Abrimos el archivo en modo append para ir acumulando resultados
    with open(archivo_salida, "a", encoding="utf-8") as f:

        f.write(header)

        for P in ordenes:
            a, G = param_lpc(x, P)
            linea = f"P={P:2d} | G={G:.6f} | a[0:{P}]={a[:P]}\n"

            print(linea, end="")   # sigue imprimiendo en consola
            f.write(linea)         # también lo guarda en el .txt

    return fs, x

#----------CARGA DE AUDIOS-----------
# -> Llamo la función para los diferentes audios e imprimo los paráemtros pedidos para cada una de los
if __name__ == "__main__":
    nombres = ["a", "e", "s", "sh"]

# Limpiar archivo de salida al inicio y agregar encabezado
archivo_salida = os.path.join(base_dir, "Resultados_Ej1a.txt")
with open(archivo_salida, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("EJERCICIO 1A - Estimación de Parámetros LPC\n")
    f.write("=" * 60 + "\n")

for nombre in nombres:
    analizar_archivo_wav(f"{nombre}.wav")

print(f"\n✓ Resultados guardados en: {archivo_salida}")