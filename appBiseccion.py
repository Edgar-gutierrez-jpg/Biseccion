import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import math

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Método de Bisección",
    page_icon="📐",
    layout="wide"
)

# --- Título ---
st.title("📐 Método de Bisección")
st.write("""
Esta aplicación encuentra la raíz de una función en un intervalo dado
utilizando el método de Bisección (corte a la mitad).
""")

# --- Barra Lateral de Entradas ---
st.sidebar.header("Parámetros de Entrada")

# Valores por defecto del script original
funcion_original = "( sqrt(3*x**3 + 2*x**2 + 4) * (x - 3) ) / ( (x**2 + 6*x + 7)**1 * (x**2 + 7)**2 )"
a_original = 0.0
b_original = 5.0
tol_original = 0.01
decimales_original = 4

# Widgets para los datos de entrada
funcion_str = st.sidebar.text_area(
    "Función f(x)", 
    value=funcion_original, 
    height=150,
    help="Escribe la función. Usa 'x' como variable. Ejemplo: sin(x**2) + exp(x) / log(x)"
)
a = st.sidebar.number_input("Límite inferior (a)", value=a_original, format="%.6f")
b = st.sidebar.number_input("Límite superior (b)", value=b_original, format="%.6f")
tol = st.sidebar.number_input(
    "Tolerancia (Error)", 
    value=tol_original, 
    min_value=0.0, 
    format="%.8f",
    help="El cálculo se detiene cuando el error (c = (b-a)/2) sea menor que este valor."
)
decimales = st.sidebar.number_input(
    "Decimales de redondeo", 
    value=decimales_original, 
    min_value=1, 
    step=1
)
max_iter = st.sidebar.number_input("Máximo de Iteraciones", value=100, min_value=1, step=1)

# --- Botón para Calcular ---
if st.sidebar.button("Calcular Raíz"):

    # --- Lógica de SymPy para crear la función ---
    x_sym = sp.symbols('x')  # Define 'x' como un símbolo
    f_lambda = None          # Inicializa la función

    try:
        f_expr = sp.sympify(funcion_str) # Traduce el string
        # Usamos "math" para compatibilidad con valores individuales
        f_lambda = sp.lambdify(x_sym, f_expr, "math") 
        
        st.success(f"Función reconocida:  `f(x) = {f_expr}`")

    except Exception as e:
        st.error(f"Error en la sintaxis de la función: {e}")
        st.stop() # Detiene la ejecución si la función está mal escrita

    # --- Función segura para manejar errores de dominio ---
    def f(x_val):
        try:
            resultado = f_lambda(x_val)
            if isinstance(resultado, complex) or np.isnan(resultado) or np.isinf(resultado):
                return np.nan
            return resultado
        except Exception:
            return np.nan

    # --- Verificación de la Condición Inicial ---
    try:
        f_a = f(a)
        f_b = f(b)
        
        if np.isnan(f_a) or np.isnan(f_b):
            st.error(f"Error: La función no se puede evaluar en los límites del intervalo [a, b]. (f(a)={f_a}, f(b)={f_b})")
            st.stop()
            
        if f_a * f_b > 0:
            st.error(f"Condición Inicial NO CUMPLIDA: f(a) y f(b) deben tener signos opuestos.")
            st.markdown(f"**f(a) = f({a}) = {f_a:.{decimales}f}**")
            st.markdown(f"**f(b) = f({b}) = {f_b:.{decimales}f}**")
            st.stop()
        
        st.info(f"Condición Inicial CUMPLIDA: f(a) y f(b) tienen signos opuestos. (f(a)={f_a:.{decimales}f}, f(b)={f_b:.{decimales}f})")

    except Exception as e:
        st.error(f"Error al evaluar la función en los límites: {e}")
        st.stop()


    # --- Lógica del Método de Bisección ---
    iteraciones = 0
    rows = [] # Lista para guardar los datos de la tabla

    # Copiamos 'a' y 'b' para no modificar los widgets
    a_iter = a
    b_iter = b
    
    # Encabezado de la tabla
    headers = ['n', 'a', 'b', 'c = (b-a)/2', 'x = a+c', 'f(x)']

    try:
        while True:
            iteraciones += 1
            
            c = (b_iter - a_iter) / 2   # Calcular c (el error)
            x_new = a_iter + c          # Calcular x (la nueva aproximación)
            f_x_new = f(x_new)
            
            # Manejar el caso de que f(x_new) falle (ej. división por cero en el punto medio)
            if np.isnan(f_x_new):
                st.error(f"Se encontró un valor indefinido en x = {x_new:.{decimales}f} durante la iteración {iteraciones}. El método no puede continuar.")
                break

            # Guardar datos de la iteración para la tabla
            row_data = [iteraciones, a_iter, b_iter, c, x_new, f_x_new]
            rows.append(row_data)

            # --- Condiciones de Parada ---
            if f_x_new == 0:
                st.success(f"**Raíz exacta encontrada:** `{round(x_new, decimales)}`")
                st.info(f"Iteraciones realizadas: {iteraciones}")
                break
            
            # Condición de parada del método de bisección (el error 'c' es menor que la tolerancia)
            if c < tol:
                st.success(f"**Raíz aproximada encontrada:** `{round(x_new, decimales)}` (con error < {tol})")
                st.info(f"Iteraciones realizadas: {iteraciones}")
                break
            
            if iteraciones >= max_iter:
                st.warning(f"Se alcanzó el límite de {max_iter} iteraciones. No se encontró la raíz.")
                st.info(f"Última aproximación: {round(x_new, decimales)}")
                break

            # --- Decidir nuevo intervalo ---
            if f(a_iter) * f_x_new < 0:
                b_iter = x_new
            else:
                a_iter = x_new
        
        # --- Mostrar Tabla de Resultados ---
        st.subheader("Tabla de Iteraciones")
        
        df = pd.DataFrame(rows, columns=headers)
        
        # Aplicar formato de decimales a las columnas correctas
        format_dict = {col: f"{{:.{decimales}f}}" for col in headers if col != 'n'}
        st.dataframe(df.style.format(format_dict), height=400)

    except Exception as e:
        st.error(f"Ocurrió un error durante el cálculo: {e}")
        st.error("Revise la función, es posible que haya una asíntota o un valor indefinido cerca de la raíz.")