

def pitch_lpc(xs, a, alpha, fs):
    """
    Detecta el pitch y clasifica el segmento como sonoro o sordo.
    xs: segmento de señal (ya ventaneado)
    a: coeficientes LPC
    alpha: umbral de decisión (0 < alpha < 1)
    fs: frecuencia de muestreo [Hz]
    Retorna: (G, fp) donde G es la ganancia y fp el pitch (0 si es sordo)
    """
    xs = np.asarray(xs, dtype=float)
    N = len(xs)
    
    # Calcular la autocorrelación para detección de pitch
    # Usamos un rango típico para voz: 50-500 Hz
    min_lag = int(fs / 500)  # Lag mínimo (~2ms para pitch máximo 500Hz)
    max_lag = int(fs / 50)   # Lag máximo (~20ms para pitch mínimo 50Hz)
    
    # Autocorrelación normalizada
    r0 = np.dot(xs, xs)  # Energía en lag 0
    
    if r0 < 1e-10:  # Segmento de silencio
        return 0, 0
    
    # Calcular autocorrelación para el rango de lags posibles
    acf = np.array([np.dot(xs[:N-k], xs[k:]) for k in range(min_lag, min(max_lag, N))])
    
    # Normalizar
    acf_norm = acf / r0
    
    # Encontrar el máximo (excluyendo lag 0)
    if len(acf_norm) > 0:
        max_idx = np.argmax(acf_norm)
        max_acf = acf_norm[max_idx]
        pitch_lag = max_idx + min_lag
    else:
        max_acf = 0
        pitch_lag = min_lag
    
    # Decisión sonoro/sordo basada en el máximo de autocorrelación
    if max_acf > alpha:
        # SONORO: calcular pitch
        fp = fs / pitch_lag
        
        # Calcular ganancia del residuo
        # e[n] = x[n] - sum(a[k]*x[n-k])
        residuo = np.copy(xs)
        for k in range(len(a)):
            residuo[k+1:] -= a[k] * xs[:-k-1]
        
        G = np.sqrt(np.mean(residuo**2))
    else:
        # SORDO: sin pitch periódico
        fp = 0
        
        # Ganancia para ruido blanco
        residuo = np.copy(xs)
        for k in range(len(a)):
            residuo[k+1:] -= a[k] * xs[:-k-1]
        
        G = np.sqrt(np.mean(residuo**2))
    
    return G, fp