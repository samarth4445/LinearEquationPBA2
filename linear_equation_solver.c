#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void printEquations(double** A, double* B, int n) {
    printf("\nThe system of linear equations is:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%.2lf", A[i][j]);
            if (j < n - 1) {
                printf("x%d + ", j + 1);
            } else {
                printf("x%d ", j + 1);
            }
        }
        printf("= %.2lf\n", B[i]);
    }
}

void gaussianElimination(double** A, double* B, double* X, int n) {
    for (int k = 0; k < n; ++k) {
        double pivot = A[k][k];
        #pragma omp parallel for
        for (int j = k; j < n; ++j) {
            A[k][j] /= pivot;
        }
        B[k] /= pivot;

        #pragma omp parallel for
        for (int i = k + 1; i < n; ++i) {
            double factor = A[i][k];
            for (int j = k; j < n; ++j) {
                A[i][j] -= factor * A[k][j];
            }
            B[i] -= factor * B[k];
        }
    }

    for (int i = n - 1; i >= 0; --i) {
        X[i] = B[i];
        #pragma omp parallel for
        for (int j = i + 1; j < n; ++j) {
            X[i] -= A[i][j] * X[j];
        }
    }
}

int main() {
    int n;

    printf("Enter the number of variables: ");
    scanf("%d", &n);

    double** A = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; ++i) {
        A[i] = (double*)malloc(n * sizeof(double));
    }
    double* B = (double*)malloc(n * sizeof(double));
    double* X = (double*)malloc(n * sizeof(double));

    printf("Enter the coefficients of the augmented matrix (A and B):\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            scanf("%lf", &A[i][j]);
        }
        scanf("%lf", &B[i]);
    }

    printEquations(A, B, n);
    gaussianElimination(A, B, X, n);

    printf("\nSolution:\n");
    for (int i = 0; i < n; ++i) {
        printf("x[%d] = %.6lf\n", i + 1, X[i]);
    }

    for (int i = 0; i < n; ++i) {
        free(A[i]);
    }
    free(A);
    free(B);
    free(X);

    return 0;
}
