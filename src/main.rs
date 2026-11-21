use std::io::{self, Write};
use std::fmt;
use std::f64::consts::PI;
use std::ops::Sub;
use std::ops::Mul;

#[derive(Clone, Debug)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<Vec<f64>>,
}

impl Matrix {
    fn new(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        Self { rows, cols, data }
    }

    fn identity(dimension: usize) -> Self {
        let mut data = vec![vec![0.0; dimension]; dimension];
        for i in 0..dimension {
            data[i][i] = 1.0;
        }
        Self {
            rows: dimension,
            cols: dimension,
            data,
        }
    }

    fn is_square(&self) -> bool {
        self.rows > 0 && self.rows == self.cols
    }

    fn determinant(&self) -> Result<f64, String> {
        if !self.is_square() {
            return Err("No se puede calcular el determinante de una matriz no cuadrada.".to_string());
        } 

        match self.rows {
            1 => Ok(self.data[0][0]),
            2 => Ok(self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]),
            3 => Ok(self.data[0][0] * (self.data[1][1] * self.data[2][2] - self.data[1][2] * self.data[2][1])
               - self.data[0][1] * (self.data[1][0] * self.data[2][2] - self.data[1][2] * self.data[2][0])
               + self.data[0][2] * (self.data[1][0] * self.data[2][1] - self.data[1][1] * self.data[2][0])),
            _ => Err("Tamaño de matriz no soportado.".to_string()),
        }
    }

    fn trace(&self) -> f64 {
        let mut trace: f64 = 0.0;
        for i in 0..self.rows {
            trace += self.data[i][i];
        }

        trace
    }

    fn principal_minors(&self) -> f64 {
        (self.data[1][1] * self.data[2][2] - self.data[1][2] * self.data[2][1])
        + (self.data[0][0] * self.data[2][2] - self.data[0][2] * self.data[2][0])
        + (self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0])
    }

    fn null_space(&self) -> Vec<Vec<f64>> {
        let mut mat = self.data.clone();
        let mut pivot_cols: Vec<usize> = Vec::new();
        let mut pivot_row = 0;
        let tol = 1e-7;

        for col in 0..self.cols {
            if pivot_row >= self.rows {
                break;
            }

            let mut best_row = pivot_row;
            let mut max_val = mat[pivot_row][col].abs();
            for i in (pivot_row + 1)..self.rows {
                if mat[i][col].abs() > max_val {
                    max_val = mat[i][col].abs();
                    best_row = i;
                }
            }

            if max_val < tol {
                continue;
            }

            mat.swap(pivot_row, best_row);

            let pivot = mat[pivot_row][col];
            for j in col..self.cols {
                mat[pivot_row][j] /= pivot;
            }

            for i in 0..self.rows {
                if i != pivot_row {
                    let factor = mat[i][col];
                    if factor.abs() > 1e-12 {
                        for j in col..self.cols {
                            mat[i][j] -= factor * mat[pivot_row][j];
                        }
                    }
                }
            }

            pivot_cols.push(col);
            pivot_row += 1;
        }

        let mut free_cols: Vec<usize> = Vec::new();
        for c in 0..self.cols {
            if !pivot_cols.contains(&c) {
                free_cols.push(c);
            }
        }

        let mut eigenvectors: Vec<Vec<f64>> = Vec::new();
        for &free_col in &free_cols {
            let mut v = vec![0.0; self.cols];
            v[free_col] = 1.0;

            for (row_idx, &piv_col) in pivot_cols.iter().enumerate() {
                v[piv_col] = -mat[row_idx][free_col];
            }

            eigenvectors.push(v);
        }

        eigenvectors
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in &self.data {
            write!(f, "[ ")?;
            for value in row {
                let clean_val = if value.abs() < 1e-10 { 0.0 } else { *value };
                write!(f, "{:8.4} ", clean_val)?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

impl Sub for &Matrix {
    type Output = Matrix;

    fn sub(self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "Las filas no coinciden para restar");
        assert_eq!(self.cols, other.cols, "Las columnas no coinciden para restar");

        let mut result = vec![vec![0.0; self.cols]; self.rows];

        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i][j] = self.data[i][j] - other.data[i][j];
            }
        }

        Matrix::new(result)
    }
}

impl Mul<&Matrix> for f64 {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Matrix {
        let mut result = vec![vec![0.0; rhs.cols]; rhs.rows];

        for i in 0..rhs.rows {
            for j in 0..rhs.cols {
                result[i][j] = self * rhs.data[i][j];
            }
        }

        Matrix::new(result)
    }
}

impl Mul<f64> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: f64) -> Matrix {
        rhs * self
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.cols, rhs.rows, "Dimensiones incompatibles para multiplicar.");

        let mut result = vec![vec![0.0; rhs.cols]; self.rows];

        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * rhs.data[k][j];
                }
                result[i][j] = sum;
            }
        }

        Matrix::new(result)
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                if (self.data[i][j] - other.data[i][j]).abs() > 1e-5 {
                    return false;
                }
            }
        }

        true
    }
}

#[derive(Debug)]
struct Polynomial {
    degree: usize,
    data: Vec<f64>,
}

impl Polynomial {
    fn new(data: Vec<f64>) -> Self {
        Self {
            degree: data.len() - 1,
            data,
        }
    }

    fn solve(&self) -> Result<Vec<f64>, String> {
        if self.degree != 2 && self.degree != 3 {
            return Err("Solo se pueden resolver polinomios de grado 2 o 3.".to_string());
        }

        let a = self.data[0];
        if a.abs() < 1e-12 {
            return Err("El coeficiente principal no puede ser cero.".to_string());
        }

        if self.degree == 2 {
            let b = self.data[1];
            let c = self.data[2];

            let disc = b.powi(2) - 4.0 * a * c;
            if disc < -1e-9 {
                return Err("Raices complejas encontradas.".to_string());
            }

            let clean_disc = disc.max(0.0);
            let sign_b = if b < 0.0 { -1.0 } else { 1.0 };
            let q = -0.5 * (b + sign_b * clean_disc.sqrt());

            if q.abs() < 1e-12 {
                return Ok(vec![0.0, 0.0]);
            }
            Ok(vec![q / a, c / q])
        } else {
            let b = self.data[1] / a;
            let c = self.data[2] / a;
            let d = self.data[3] / a;

            let b2 = b * b;
            let p = c - b2 / 3.0;
            let q = (2.0 * b2 * b - 9.0 * b * c + 27.0 * d) / 27.0;

            let q_val = p / 3.0;
            let r_val = q / 2.0;
            let d_val = q_val * q_val * q_val + r_val * r_val;
            let offset = b / 3.0;

            if d_val.abs() < 1e-7 {
                if r_val.abs() < 1e-12 {
                    // Raíz triple
                    Ok(vec![-offset, -offset, -offset])
                } else {
                    let u = (-r_val).cbrt();
                    Ok(vec![
                        2.0 * u - offset,
                        -u - offset,
                        -u - offset,
                    ])
                }
            } else if d_val < 0.0 {
                // Caso raíces reales distintas.
                let sqrt_neg_q_val = (-q_val).sqrt();
                let cos_theta = (-r_val / (sqrt_neg_q_val * -q_val)).clamp(-1.0, 1.0);
                let theta = cos_theta.acos();

                let two_sqrt_q_val = 2.0 * sqrt_neg_q_val;
                let two_pi = PI * 2.0;

                let x1 = two_sqrt_q_val * (theta / 3.0).cos() - offset;
                let x2 = two_sqrt_q_val * ((theta + two_pi) / 3.0).cos() - offset;
                let x3 = two_sqrt_q_val * ((theta + 2.0 * two_pi) / 3.0).cos() - offset;

                Ok(vec![x1, x2, x3])
            } else {
                // Caso D > 0 (1 raíz real y 2 complejas)
                return Err("Raices complejas encontradas.".to_string());
            }
        }
    }
}

impl fmt::Display for Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut is_first = true;

        for (i, &coeff) in self.data.iter().enumerate() {
            if coeff == 0.0 {
                continue;
            }

            let exp = self.degree - i;
            let abs_coeff = coeff.abs();

            // 2. Manejar el signo correctamente
            if is_first {
                if coeff < 0.0 {
                    write!(f, "-")?;
                }
                is_first = false;
            } else if coeff < 0.0 {
                write!(f, " - ")?;
            } else {
                write!(f, " + ")?;
            }

            if abs_coeff != 1.0 || exp == 0 {
                write!(f, "{}", abs_coeff)?;
            }

            match exp {
                0 => {}
                1 => write!(f, "λ")?,
                _ => write!(f, "λ^{}", exp)?,
            }
        }

        if is_first {
            write!(f, "0")?;
        }

        Ok(())
    }
}

fn read_line(message: &str) -> String {
    print!("{}", message);
    io::stdout().flush().expect("Error al limpiar stdout");

    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("Error al leer desde el teclado");

    input.trim().to_string()
}

fn read_matrix() -> Result<Matrix, String> {
    let rows: usize = read_line("Número de filas: ")
        .parse()
        .map_err(|_| "Las filas deben ser un entero positivo".to_string())?;
    let cols: usize = read_line("Número de columnas: ")
        .parse()
        .map_err(|_| "Las columnas deben ser un entero positivo".to_string())?;

    if rows <= 1 || cols <= 1 {
        return Err("El tamaño mínimo es 2x2".to_string());
    }

    let mut data: Vec<Vec<f64>> = Vec::with_capacity(rows);

    println!("Ingresa cada fila separando los {} valores con espacios:", cols);
    for i in 0..rows {
        let line = read_line(&format!("Fila {}: ", i + 1));

        let row: Result<Vec<f64>, _> = line
            .split_whitespace()
            .map(|token| token.parse::<f64>())
            .collect();

        let row = row.map_err(|_| format!("Valor no númerico en la fila {}.", i + 1))?;

        if row.len() != cols {
            return Err(format!(
                "Se esperaban {} columnas en la fila {}, pero ingresaste {}.",
                cols,
                i + 1,
                row.len()
            ));
        }

        data.push(row);
    }

    Ok(Matrix::new(data))
}

fn main() {
    let a = match read_matrix() {
        Ok(m) => m,
        Err(e) => {
            println!("Error de entrada: {}", e);
            return;
        }
    };

    println!("\nMatriz A ingresada ({}x{}):", a.rows, a.cols);
    println!("{}", a);

    if !a.is_square() {
        println!("No se pueden calcular eigenvalores y eigenvectores.");
        return;
    }

    let trace = a.trace();
    let det = match a.determinant() {
        Ok(d) => d,
        Err(e) => {
            println!("{}", e);
            return;
        }
    };

    let polynomial = if a.rows == 2 {
        Polynomial::new(vec![1.0, -trace, det])
    } else {
        // La única forma de llegar aquí es si m es 3x3.
        let principal_minors = a.principal_minors();
        Polynomial::new(vec![1.0, -trace, principal_minors, -det])
    };

    println!("Polinomio: {}", polynomial);
    let roots = match polynomial.solve() {
        Ok(v) => v,
        Err(e) => {
            println!("Error al resolver polinomio: {}", e);
            return;
        }
    };
    println!("Las raíces son: {:?}", roots);

    let mut unique_roots = roots.clone();
    unique_roots.sort_by(|a, b| a.total_cmp(b));
    unique_roots.dedup_by(|a, b| (*a - *b).abs() < 1e-7);

    let identity_m = Matrix::identity(a.rows);
    let mut eigen_pairs: Vec<(f64, Vec<f64>)> = Vec::new();

    for &root in &unique_roots {
        let system = &a - &(root * &identity_m);
        let vectors = system.null_space();
        for v in vectors {
            eigen_pairs.push((root, v));
        }
    }

    println!("\nPares eigen: ");
    for t in &eigen_pairs {
        println!("Raiz {}: {:?}", t.0, t.1);
    }

    if eigen_pairs.len() < a.rows {
        println!("No se puede calcular la matriz Q que diagonaliza a A.");
        return;
    }

    let n = a.rows;
    let mut q_data = vec![vec![0.0; n]; n];
    let mut d_data = vec![vec![0.0; n]; n];

    for (col, (eigenval, eigenvec)) in eigen_pairs.iter().enumerate() {
        d_data[col][col] = *eigenval;
        for row in 0..n {
            q_data[row][col] = eigenvec[row];
        }
    }

    let q = Matrix::new(q_data);
    let d = Matrix::new(d_data);

    println!("\nMatriz Q (eigenvectores en columnas):");
    println!("{}", q);

    println!("Matriz D (eigenvalores en diagonal):");
    println!("{}", d);

    let qd = &q * &d;
    let aq = &a * &q;

    println!("Producto Q * D:");
    println!("{}", qd);

    println!("Producto A * Q:");
    println!("{}", aq);

    if qd == aq {
        println!("Ya que se cumple QD = AQ, la matriz A es diagonalizable.")
    } else {
        println!("No se cumple QD = AQ.")
    }
}


// TESTS DE MIAMOR GEMINI
#[cfg(test)]
mod tests {
    use super::*;

    /// Función auxiliar para comparar flotantes con una tolerancia
    fn assert_approx_eq(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "Se esperaba aproximadamente {}, pero se obtuvo {} (diferencia: {})",
            b,
            a,
            (a - b).abs()
        );
    }

    // ==========================================
    // TESTS DE MATRICES
    // ==========================================

    #[test]
    fn test_matrix_creation_and_identity() {
        let m = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 2);
        assert!(m.is_square());

        let rect = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert!(!rect.is_square());

        let id = Matrix::identity(3);
        let expected_id = Matrix::new(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        assert_eq!(id, expected_id);
    }

    #[test]
    fn test_matrix_determinant() {
        // 1x1
        let m1 = Matrix::new(vec![vec![5.0]]);
        assert_eq!(m1.determinant().unwrap(), 5.0);

        // 2x2
        let m2 = Matrix::new(vec![vec![4.0, 6.0], vec![3.0, 8.0]]);
        assert_approx_eq(m2.determinant().unwrap(), 14.0, 1e-9);

        // 3x3
        let m3 = Matrix::new(vec![
            vec![6.0, 1.0, 1.0],
            vec![4.0, -2.0, 5.0],
            vec![2.0, 8.0, 7.0],
        ]);
        assert_approx_eq(m3.determinant().unwrap(), -306.0, 1e-9);

        // Errores esperados
        let non_square = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert!(non_square.determinant().is_err());

        let m4 = Matrix::identity(4);
        assert!(m4.determinant().is_err());
    }

    #[test]
    fn test_matrix_trace_and_principal_minors() {
        let m = Matrix::new(vec![
            vec![2.0, 1.0, 0.0],
            vec![0.0, 3.0, 4.0],
            vec![1.0, 0.0, 5.0],
        ]);
        // Traza = 2 + 3 + 5 = 10
        assert_approx_eq(m.trace(), 10.0, 1e-9);

        // Menores principales: (3*5 - 4*0) + (2*5 - 0*1) + (2*3 - 1*0) = 15 + 10 + 6 = 31
        assert_approx_eq(m.principal_minors(), 31.0, 1e-9);
    }

    #[test]
    fn test_matrix_operations() {
        let a = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = Matrix::new(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);

        // Resta
        let sub_res = &b - &a;
        assert_eq!(sub_res, Matrix::new(vec![vec![4.0, 4.0], vec![4.0, 4.0]]));

        // Multiplicación por escalar (izquierda y derecha)
        let scalar_left = 2.0 * &a;
        let scalar_right = &a * 2.0;
        let expected_scalar = Matrix::new(vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
        assert_eq!(scalar_left, expected_scalar);
        assert_eq!(scalar_right, expected_scalar);

        // Multiplicación de matrices
        let mul_res = &a * &b;
        let expected_mul = Matrix::new(vec![vec![19.0, 22.0], vec![43.0, 50.0]]);
        assert_eq!(mul_res, expected_mul);
    }

    #[test]
    #[should_panic(expected = "Dimensiones incompatibles para multiplicar.")]
    fn test_matrix_mul_incompatible_panics() {
        let a = Matrix::new(vec![vec![1.0, 2.0]]); // 1x2
        let b = Matrix::new(vec![vec![1.0, 2.0]]); // 1x2
        let _ = &a * &b;
    }

    #[test]
    fn test_null_space() {
        // Matriz con rango 1 (2x2), dimensión del espacio nulo debe ser 1
        // Fila 2 es 2 * Fila 1 -> x + 2y = 0 -> vector base [-2, 1]
        let m = Matrix::new(vec![vec![1.0, 2.0], vec![2.0, 4.0]]);
        let ns = m.null_space();
        assert_eq!(ns.len(), 1);
        assert_approx_eq(ns[0][0], -2.0, 1e-6);
        assert_approx_eq(ns[0][1], 1.0, 1e-6);

        // Matriz invertible -> espacio nulo vacío
        let inv = Matrix::identity(3);
        assert!(inv.null_space().is_empty());
    }

    #[test]
    fn test_matrix_display() {
        let m = Matrix::new(vec![vec![1.0, -1e-12], vec![0.0, 2.5]]);
        let output = format!("{}", m);
        assert!(output.contains("1.0000"));
        assert!(output.contains("0.0000")); // Verifica limpieza de -0.0000
        assert!(output.contains("2.5000"));
    }

    // ==========================================
    // TESTS DE POLINOMIOS
    // ==========================================

    #[test]
    fn test_polynomial_quadratic_real_roots() {
        // λ^2 - 5λ + 6 = 0 -> raíces: 2, 3
        let p = Polynomial::new(vec![1.0, -5.0, 6.0]);
        let mut roots = p.solve().unwrap();
        roots.sort_by(|a, b| a.total_cmp(b));

        assert_eq!(roots.len(), 2);
        assert_approx_eq(roots[0], 2.0, 1e-7);
        assert_approx_eq(roots[1], 3.0, 1e-7);
    }

    #[test]
    fn test_polynomial_quadratic_double_root_and_zero() {
        // λ^2 - 4λ + 4 = 0 -> raíz doble: 2
        let p_double = Polynomial::new(vec![1.0, -4.0, 4.0]);
        let roots_double = p_double.solve().unwrap();
        assert_approx_eq(roots_double[0], 2.0, 1e-7);
        assert_approx_eq(roots_double[1], 2.0, 1e-7);

        // λ^2 = 0 -> raíces: 0, 0
        let p_zero = Polynomial::new(vec![1.0, 0.0, 0.0]);
        let roots_zero = p_zero.solve().unwrap();
        assert_eq!(roots_zero, vec![0.0, 0.0]);
    }

    #[test]
    fn test_polynomial_cubic_three_distinct_roots() {
        // (λ - 1)(λ - 2)(λ - 3) = λ^3 - 6λ^2 + 11λ - 6 = 0
        let p = Polynomial::new(vec![1.0, -6.0, 11.0, -6.0]);
        let mut roots = p.solve().unwrap();
        roots.sort_by(|a, b| a.total_cmp(b));

        assert_eq!(roots.len(), 3);
        assert_approx_eq(roots[0], 1.0, 1e-6);
        assert_approx_eq(roots[1], 2.0, 1e-6);
        assert_approx_eq(roots[2], 3.0, 1e-6);
    }

    #[test]
    fn test_polynomial_cubic_multiple_roots() {
        // Raíz triple: (λ - 2)^3 = λ^3 - 6λ^2 + 12λ - 8 = 0
        let p_triple = Polynomial::new(vec![1.0, -6.0, 12.0, -8.0]);
        let roots_triple = p_triple.solve().unwrap();
        for r in roots_triple {
            assert_approx_eq(r, 2.0, 1e-6);
        }

        // Raíz doble y una simple: (λ - 1)^2 * (λ - 4) = λ^3 - 6λ^2 + 9λ - 4 = 0
        let p_double = Polynomial::new(vec![1.0, -6.0, 9.0, -4.0]);
        let mut roots_double = p_double.solve().unwrap();
        roots_double.sort_by(|a, b| a.total_cmp(b));
        assert_approx_eq(roots_double[0], 1.0, 1e-6);
        assert_approx_eq(roots_double[1], 1.0, 1e-6);
        assert_approx_eq(roots_double[2], 4.0, 1e-6);
    }

    #[test]
    fn test_polynomial_errors() {
        // Raíces complejas (λ^2 + 1 = 0)
        let complex_quad = Polynomial::new(vec![1.0, 0.0, 1.0]);
        assert!(complex_quad.solve().is_err());

        // Cúbica con raíces complejas (λ^3 - 1 = 0 -> 1 real, 2 complejas)
        let complex_cubic = Polynomial::new(vec![1.0, 0.0, 0.0, -1.0]);
        assert!(complex_cubic.solve().is_err());

        // Coeficiente principal cero
        let zero_lead = Polynomial::new(vec![0.0, 2.0, 1.0]);
        assert!(zero_lead.solve().is_err());

        // Grado no soportado (grado 1 o grado 4)
        let deg1 = Polynomial::new(vec![1.0, 2.0]);
        assert!(deg1.solve().is_err());
    }

    #[test]
    fn test_polynomial_display() {
        let p = Polynomial::new(vec![1.0, -5.0, 0.0, 6.0]);
        assert_eq!(format!("{}", p), "λ^3 - 5λ^2 + 6");

        let zero_poly = Polynomial::new(vec![0.0, 0.0, 0.0]);
        assert_eq!(format!("{}", zero_poly), "0");
    }

    // ==========================================
    // TESTS DE INTEGRACIÓN (DIAGONALIZACIÓN)
    // ==========================================

    #[test]
    fn test_full_diagonalization_3x3() {
        // Matriz simétrica 3x3 (garantizada diagonalizable con eigenvalores reales)
        let a = Matrix::new(vec![
            vec![4.0, 1.0, 1.0],
            vec![1.0, 4.0, 1.0],
            vec![1.0, 1.0, 4.0],
        ]);
        // Eigenvalores esperados: 6, 3, 3

        let poly = Polynomial::new(vec![1.0, -a.trace(), a.principal_minors(), -a.determinant().unwrap()]);
        let mut roots = poly.solve().unwrap();
        roots.sort_by(|x, y| x.total_cmp(y));
        roots.dedup_by(|x, y| (*x - *y).abs() < 1e-7);

        let id = Matrix::identity(a.rows);
        let mut eigen_pairs = Vec::new();
        for &r in &roots {
            let sys = &a - &(r * &id);
            for v in sys.null_space() {
                eigen_pairs.push((r, v));
            }
        }

        // Debe encontrar 3 eigenvectores linealmente independientes (2 para λ=3, 1 para λ=6)
        assert_eq!(eigen_pairs.len(), 3);

        let mut q_data = vec![vec![0.0; 3]; 3];
        let mut d_data = vec![vec![0.0; 3]; 3];
        for (col, (val, vec)) in eigen_pairs.iter().enumerate() {
            d_data[col][col] = *val;
            for row in 0..3 {
                q_data[row][col] = vec[row];
            }
        }

        let q = Matrix::new(q_data);
        let d = Matrix::new(d_data);

        // Verificar propiedad fundamental: A * Q == Q * D
        assert_eq!(&a * &q, &q * &d);
    }

    #[test]
    fn test_non_diagonalizable_matrix() {
        // Matriz de Jordan 2x2: [[2, 1], [0, 2]] -> Eigenvalor 2 con multiplicidad algebraica 2,
        // pero multiplicidad geométrica 1 (no es diagonalizable).
        let a = Matrix::new(vec![vec![2.0, 1.0], vec![0.0, 2.0]]);
        let poly = Polynomial::new(vec![1.0, -a.trace(), a.determinant().unwrap()]);
        
        let mut roots = poly.solve().unwrap();
        roots.dedup_by(|x, y| (*x - *y).abs() < 1e-7);

        let id = Matrix::identity(2);
        let mut eigen_pairs = Vec::new();
        for &r in &roots {
            let sys = &a - &(r * &id);
            for v in sys.null_space() {
                eigen_pairs.push((r, v));
            }
        }

        // Solo obtiene 1 eigenvector para una matriz 2x2
        assert_eq!(eigen_pairs.len(), 1);
    }
}