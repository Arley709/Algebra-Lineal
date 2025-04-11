import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.compile import function

def resolver_sistema():
    A = np.array([[2, 1], [1, 3]], dtype=np.float64)
    b = np.array([8, 13], dtype=np.float64)

    A_sym = pt.dmatrix("A")
    b_sym = pt.dvector("b")

    x_sym = pt.solve(A_sym, b_sym)
    solve_fn = function([A_sym, b_sym], x_sym)

    x = solve_fn(A, b)
    print("Solución del sistema Ax = b:")
    print(x)

def matriz_inversa():
    A = np.array([[4, 7], [2, 6]], dtype=np.float64)
    A_sym = pt.dmatrix("A")
    inv_sym = pt.nlinalg.matrix_inverse(A_sym)
    inv_fn = function([A_sym], inv_sym)

    inv = inv_fn(A)
    print("Inversa de la matriz:")
    print(inv)

def determinante():
    A = np.array([[1, 2], [3, 4]], dtype=np.float64)
    A_sym = pt.dmatrix("A")
    det_sym = pt.nlinalg.det(A_sym)
    det_fn = function([A_sym], det_sym)

    det = det_fn(A)
    print("Determinante:")
    print(det)

def multiplicar_matrices():
    A = np.array([[1, 2], [3, 4]], dtype=np.float64)
    B = np.array([[2, 0], [1, 2]], dtype=np.float64)

    A_sym = pt.dmatrix("A")
    B_sym = pt.dmatrix("B")
    C_sym = pt.dot(A_sym, B_sym)
    mult_fn = function([A_sym, B_sym], C_sym)

    C = mult_fn(A, B)
    print("Resultado de A * B:")
    print(C)

def autovalores_autovectores():
    A = np.array([[4, -2], [1, 1]], dtype=np.float64)
    vals, vects = np.linalg.eig(A)
    print("Autovalores:")
    print(vals)
    print("Autovectores:")
    print(vects)

def sumar_matrices():
    A = np.array([[1, 2], [3, 4]], dtype=np.float64)
    B = np.array([[5, 6], [7, 8]], dtype=np.float64)

    A_sym = pt.dmatrix("A")
    B_sym = pt.dmatrix("B")
    suma_sym = A_sym + B_sym
    suma_fn = function([A_sym, B_sym], suma_sym)

    resultado = suma_fn(A, B)
    print("Suma A + B:")
    print(resultado)

def restar_matrices():
    A = np.array([[5, 6], [7, 8]], dtype=np.float64)
    B = np.array([[1, 2], [3, 4]], dtype=np.float64)

    A_sym = pt.dmatrix("A")
    B_sym = pt.dmatrix("B")
    resta_sym = A_sym - B_sym
    resta_fn = function([A_sym, B_sym], resta_sym)

    resultado = resta_fn(A, B)
    print("Resta A - B:")
    print(resultado)

def escalar_por_matriz():
    escalar = 3.0
    A = np.array([[1, 2], [3, 4]], dtype=np.float64)

    A_sym = pt.dmatrix("A")
    esc_sym = pt.dscalar("escalar")

    resultado_sym = esc_sym * A_sym
    mult_fn = function([esc_sym, A_sym], resultado_sym)

    resultado = mult_fn(escalar, A)
    print(f"Resultado de {escalar} * A:")
    print(resultado)

def transpuesta():
    A = np.array([[1, 2], [3, 4]], dtype=np.float64)
    A_sym = pt.dmatrix("A")
    trans_sym = A_sym.T
    trans_fn = function([A_sym], trans_sym)

    resultado = trans_fn(A)
    print("Transpuesta de A:")
    print(resultado)

def menu():
    while True:
        print("\n--- MENÚ ÁLGEBRA LINEAL CON PYTENSOR ---")
        print("1. Resolver sistema Ax = b")
        print("2. Calcular inversa de una matriz")
        print("3. Calcular determinante de una matriz")
        print("4. Multiplicar dos matrices")
        print("5. Autovalores y autovectores (con NumPy)")
        print("6. Sumar dos matrices")
        print("7. Restar dos matrices")
        print("8. Multiplicar matriz por escalar")
        print("9. Calcular transpuesta de una matriz")
        print("0. Salir")

        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            resolver_sistema()
        elif opcion == "2":
            matriz_inversa()
        elif opcion == "3":
            determinante()
        elif opcion == "4":
            multiplicar_matrices()
        elif opcion == "5":
            autovalores_autovectores()
        elif opcion == "6":
            sumar_matrices()
        elif opcion == "7":
            restar_matrices()
        elif opcion == "8":
            escalar_por_matriz()
        elif opcion == "9":
            transpuesta()
        elif opcion == "0":
            break
        else:
            print("Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    menu()
