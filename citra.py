import numpy as np
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Program CLI untuk Perkalian Matriks 3 Dimensi (Batched Matrix Multiplication)"
    )
    
    parser.add_argument('-b', '--batch', type=int, required=True, help="Jumlah batch (kedalaman/depth matriks 3D)")
    parser.add_argument('-m', '--baris_a', type=int, required=True, help="Jumlah baris matriks A")
    parser.add_argument('-n', '--kolom_a_baris_b', type=int, required=True, help="Jumlah kolom matriks A sekaligus baris matriks B")
    parser.add_argument('-p', '--kolom_b', type=int, required=True, help="Jumlah kolom matriks B")
    
    try:
        args = parser.parse_args()
    except SystemExit:
        print("\nHarus memasukan nilai setiap variabel.")
        sys.exit()

    print("="*50)
    print("Matriks Acak (Skala 1-9)")
    print("="*50)

    matrix_a = np.random.randint(1, 10, size=(args.batch, args.baris_a, args.kolom_a_baris_b))
    matrix_b = np.random.randint(1, 10, size=(args.batch, args.kolom_a_baris_b, args.kolom_b))

    print(f"\n[+] Matriks A terbentuk dengan shape: {matrix_a.shape}")
    print(matrix_a)
    print(f"\n[+] Matriks B terbentuk dengan shape: {matrix_b.shape}")
    print(matrix_b)
    print("\n" + "="*50)
    print("perkalian (A * B)")
    print("="*50)
    
    try:
        matrix_c = matrix_a @ matrix_b
        print(f"\n[+] Hasil Matriks C memiliki shape: {matrix_c.shape}")
        print(matrix_c)
    except ValueError as e:
        print(f"\n[!] ERROR Perkalian: {e}")

if __name__ == "__main__":
    main()