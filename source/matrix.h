#pragma once

#include <initializer_list>
#include <cstddef>
#include <iostream>

namespace linalg {
    class Matrix {
    public:
        Matrix() = default;
        explicit Matrix(size_t rows);
        Matrix(size_t rows, size_t columns);
        Matrix(const Matrix& matrix);
        Matrix(Matrix&& matrix) noexcept;
        Matrix(std::initializer_list<std::initializer_list<double>> list);
        Matrix(std::initializer_list<double> list);
        Matrix& operator=(const Matrix& matrix);
        Matrix& operator=(Matrix&& matrix) noexcept;
        ~Matrix() noexcept { delete[] m_ptr; }

        double& operator()(size_t row, size_t column) noexcept {
            return m_ptr[m_columns * row + column];
        }
        double operator()(size_t row, size_t column) const noexcept {
            return m_ptr[m_columns * row + column];
        }

        [[nodiscard]] size_t rows() const noexcept { return m_rows; }
        [[nodiscard]] size_t columns() const noexcept { return m_columns; }
        [[nodiscard]] bool empty() const noexcept { return m_ptr == nullptr; }
        void reshape(size_t rows, size_t columns);

        Matrix& operator+=(const Matrix& matrix);
        Matrix& operator-=(const Matrix& matrix);
        Matrix& operator*=(const Matrix& matrix);
        Matrix& operator*=(double number) noexcept;

        [[nodiscard]] double norm() const noexcept;
        [[nodiscard]] double trace() const;
        [[nodiscard]] double det() const;
        Matrix& gauss_forward() noexcept;
        Matrix& gauss_backward();
        [[nodiscard]] size_t rank() const noexcept;
    private:
        [[nodiscard]]int find_first_not_zero_row_in_col(size_t col, size_t row_start) const noexcept;
        void swap_rows(size_t row1, size_t row2) noexcept;
        void divide_row_by_a_number(size_t row, double number, size_t col_start) noexcept;
        void combine_rows(size_t take, size_t change, double ratio, size_t col_start) noexcept;
        void make_zero_elements_under(size_t row_start, size_t col) noexcept;
        void make_zero_elements_above(size_t row_start, size_t col) noexcept;
    private:
        double* m_ptr = nullptr;
        size_t m_rows = 0;
        size_t m_columns = 0;
    };

    std::ostream& operator<<(std::ostream& out, const Matrix& matrix);
    const Matrix operator+(const Matrix& m1, const Matrix& m2);
    const Matrix operator-(const Matrix& m1, const Matrix& m2);
    const Matrix operator*(const Matrix& m1, const Matrix& m2);
    const Matrix operator*(const Matrix& m, double number) noexcept;
    const Matrix operator*(double number, const Matrix& m) noexcept;
    bool operator==(const Matrix& m1, const Matrix& m2) noexcept;
    bool operator!=(const Matrix& m1, const Matrix& m2) noexcept;

    Matrix concatenate(const Matrix& m1, const Matrix& m2);
    Matrix transpose(const Matrix& m);
    Matrix invert(const Matrix& m);
    Matrix power(const Matrix& m, int power);
    Matrix solve(const Matrix& A, const Matrix& b);
}