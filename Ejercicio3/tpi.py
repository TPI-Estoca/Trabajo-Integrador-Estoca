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
    """
    nperseg = 512 
    noverlap = nperseg // 2

    f, t, Sxx = signal.spectrogram(
        signal_audio, fs, nperseg=nperseg, noverlap=noverlap, 
        window='hann', scaling='spectrum'
    )

    Sxx_dB = 10 * np.log10(Sxx + 1e-12) 

    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t, f, Sxx_dB, shading='gouraud')
    
    plt.ylabel('Frecuencia [Hz]')
    plt.xlabel('Tiempo [s]')
    plt.title(title)
    
    cbar = plt.colorbar(label='Intensidad [dB]')
    plt.ylim(0, fs / 2)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"Espectrograma '{filename}' generado con éxito.")


def process_lpc(signal_audio, fs, alpha, L, P, ventana, hop, modo_pitch, i_audio, output_dir):
    """
    Realiza la codificación y decodificación LPC para un modo de pitch específico.
    output_dir: Carpeta donde se guardarán los resultados.
    """
    
    # --- Codificación LPC ---
    data_lpc = []
    signal_audio_padded = np.pad(signal_audio, (0, L), 'constant')
    num_segmentos = (len(signal_audio_padded) - L) // hop + 1
    
    for i in range(num_segmentos):
        inicio = i * hop
        fin = inicio + L
        
        segmento = signal_audio_padded[inicio:fin] * ventana
        
        # 1. Estimar LPC (a y Ganancia G del error de predicción)
        a, G = param_lpc(segmento, P)
        
        # 2. Estimar Pitch y Clasificar
        fp_est, max_acf = pitch_lpc(segmento, fs)
        
        # CLASIFICACIÓN Y PITCH USADO (fp_used)
        if modo_pitch == 'a':
            fp_used = fp_est if max_acf > alpha else 0
        
        elif modo_pitch == 'b':
            if np.max(np.abs(segmento)) < 0.05: # Umbral de silencio
                fp_used = 0 
            else:
                fp_used = 200.0 
        
        elif modo_pitch == 'c':
            fp_used = 0 
            
        elif modo_pitch == 'd':
            fp_used = pitch_sintetico(i, fs)
        
        else:
            raise ValueError("Modo de pitch no reconocido.")
            
        data_lpc.append((a, G, fp_used))


    # --- Decodificación LPC ---
    signal_recons = np.zeros(len(signal_audio_padded))

    for i, (a, G, fp) in enumerate(data_lpc):
        inicio = i * hop
        
        # Generar excitación E
        if fp > 1.0: # SONORO
            excitacion = gen_pulsos(fp, L, fs)
        else: # SORDO o Silencio
            excitacion = np.random.randn(L)
        
        excitacion *= G # Escalar excitación por la Ganancia G
        
        # Filtrar con modelo LPC
        a_lfilter = np.concatenate(([1], -a)) 
        segmento_sint = signal.lfilter([1], a_lfilter, excitacion)
        
        # Overlap-add
        signal_recons[inicio:inicio+L] += segmento_sint * ventana


    # --- Guardar y Graficar ---
    signal_recons_final = signal_recons[:len(signal_audio)]
    max_abs = np.max(np.abs(signal_recons_final))
    if max_abs > 1e-10:
        signal_recons_norm = signal_recons_final / max_abs
    else:
        signal_recons_norm = signal_recons_final 

    audio_int16 = (signal_recons_norm * 32767).astype(np.int16)

    # Guardar audio
    filename_wav = os.path.join(output_dir, f'audio_{i_audio}_recons_{modo_pitch}.wav')
    wav.write(filename_wav, fs, audio_int16)

    # Generar espectrograma
    filename_spec = os.path.join(output_dir, f'espectrograma_recons_{modo_pitch}.png')
    plot_spectrogram(signal_recons_final, fs, f'Reconstruido (Audio {i_audio}, Modo {modo_pitch.upper()})', filename_spec)

    return signal_recons_final


# --- Código Principal (Iteración Múltiple) ---
if __name__ == '__main__': 
    
    # -- Parámetros generales (usando tus valores optimizados) --
    fs = 8000 
    L = 180
    ventana = np.hamming(L)
    P = 15     
    alpha = 0.2 
    hop = L // 2 
    
    # Lista de audios a procesar
    audio_indices = ['01', '02', '03', '04']
    modos = ['a', 'b', 'c', 'd']
    
    base_input_path = 'AudiosEj3\\' # Ajusta esta ruta si es necesario

    for i_audio in audio_indices:
        
        # 1. Definir y crear la carpeta de resultados para este audio
        output_dir = f'resultados_audio_{i_audio}_lpc'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"\n--- Creada carpeta de resultados: {output_dir} ---")
        
        # 2. Cargar el audio original
        try:
            audio_path = os.path.join(base_input_path, f'audio_{i_audio}.wav')
            fs, signal_audio = wav.read(audio_path)
        except FileNotFoundError:
            # Fallback por si la ruta relativa falla
            audio_path = f'audio_{i_audio}.wav'
            fs, signal_audio = wav.read(audio_path) 

        signal_audio = signal_audio.astype(float) 

        # 3. Generar espectrograma original una sola vez
        original_spec_filename = os.path.join(output_dir, f'espectrograma_original.png')
        plot_spectrogram(signal_audio, fs, f'Espectrograma Original (Audio {i_audio})', original_spec_filename)
        
        
        # 4. Iterar sobre los 4 modos del ejercicio
        for modo in modos:
            print(f"\n--- Procesando Audio {i_audio}, Ejercicio 3({modo}) ---")
            process_lpc(signal_audio, fs, alpha, L, P, ventana, hop, modo, i_audio, output_dir)
            
    print("\n********************************************************")
    print("¡Procesamiento de los 4 audios con los 4 modos completado!")
    print("********************************************************")