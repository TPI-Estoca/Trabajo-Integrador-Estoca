# Cargo las librebrerías necesarias
import os
from scipy.io import wavfile
from scipy.linalg import toeplitz
import numpy as np

# Ruta a la carpeta del script actual
base_dir = os.path.dirname(os.path.abspath(__file__))

# Carpeta donde están los audios (por ejemplo: ./AudiosEj1)
ruta_audios = os.path.join(base_dir, "AudiosEj1")

# --- EJERCICIO 1A---
def param_lpc(x, P):
  x = np.asarray(x, dtype=float) # -> Me aseguro que "x" sea un array de flotantes. Si "wavfile.read()" me da enteros los paso a float.
  N = len(x) # -> Cantidad de muestras del audio.
  z = np.array([np.dot(x[:N-k], x[k:]) / (N-k) for k in range(P+1)]) # -> Calculo la autocorrelación no sesgada del audio.
  """
  np.dot(x[:N-k], x[k:]) -> Autocorrelaciona (mide qué tan parecido es el audio a sí mismo desplazado k muestras).
  z[0] es la energía total de la señal (What?). Parece que es E[x^2(n)]
  Con z[1], z[2], ...; comparo la señal con una copia corrida k muestras.
  Nota 1: si la señal es periódica, se supone que para k's parecido al período o sus multiplos, tenemos la misma señal.
  Nota 2: si es ruido, la autocorrelación cae rápido para valores distintos y cercanos a k. Porq está descorrelacionada
  """
  R = toeplitz(z[:P]) # -> Construyo la matriz de autocorrelación R de tamaño P x P.
  r = z[1:P+1] # R*a = r. Donde a son los coeficientes LPC que queremos.

  try:
    a = np.linalg.solve(R, r) # -> resuelve el sistema lineal
  except:
    a = np.linalg.lstsq(R, r, rcond=None)[0] # -> usa la solución en mínimos cuadrados
  G2 = z[0] - np.dot(a, r)
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

    for nombre in nombres:
        analizar_archivo_wav(f"{nombre}.wav")
