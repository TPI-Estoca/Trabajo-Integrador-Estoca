# Descripción general del proyecto
- **Proyecto**: Trabajo-Integrador-Estoca (TPI)  
- **Alcance**: Ejercicios y vocoder final implementando LPC (modelo fuente–filtro) para análisis y síntesis de habla.  
- **Integrantes**:
      - Francisco Javier Moya
      - Ulises Ferrero
      - Emilia Cavalitto

## Estructura del repositorio
- **`Ejercicio1/`**: Estimación de parámetros LPC y comparación de PSD.  
  - `Ej1a.py`: Implementa `param_lpc(xs, P)` y scripts para analizar fonemas aislados y guardar `Resultados_Ej1a.txt`.  
  - `Ej1b.py`: Compara el periodograma vs la PSD por LPC y guarda imágenes en la carpeta `Imagenes_Ej1b/`.  
  - `AudiosEj1/`: Carpeta con archivos WAV de fonemas de ejemplo (por ej. `a.wav`, `e.wav`, `s.wav`, `sh.wav`).  

- **`Ejercicio2/`**: Detección de pitch usando autocorrelación basada en LPC.  
  - `Ej2a.py`: Implementa una rutina de detección de pitch y escribe `Resultados_Ej2a.txt`.  
  - `Ej2b.py`: (archivo del ejercicio 2b — revisar si hay scripts adicionales en esta carpeta).  

- **`Ejercicio3/`**: Vocoder LPC completo (análisis + síntesis).  
  - `tpi.py`: Script principal que implementa el codificador/decodificador, cuatro modos de excitación (a/b/c/d), y genera WAV reconstruidos y espectrogramas PNG en directorios `resultados_audio_<idx>_lpc/`.  
  - `AudiosEj3/`: Archivos de audio de entrada (`audio_01.wav`, `audio_02.wav`, ...).  
  - `resultados_audio_01_lpc/`, ...: Carpetas de salida de ejemplo (generadas por `tpi.py`).  

## Requisitos
- Python 3.8+ (o compatible)  
- Paquetes requeridos: `numpy`, `scipy`, `matplotlib`, `IPython` (opcional para reproducción con `Audio`)  

Podés instalar las dependencias con pip:

```bash
pip install numpy scipy matplotlib ipython
