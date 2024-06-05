#include "matrix.h"

#include <stdexcept>
#include <utility>
#include <sstream>
#include <iomanip>
#include <cmath>

static bool is_equal(double n1, double n2) {
    return std::fabs(n1 - n2) < std::numeric_limits<double>::epsilon() * 100;
}

linalg::Matrix::Matrix(size_t rows) {
    if (rows == 0) { return; }
    m_ptr = new double[rows];
    for (size_t i = 0; i < rows; ++i) {
        m_ptr[i] = 0;
    }
    m_rows = rows;
    m_columns = 1;
}

linalg::Matrix::Matrix(size_t rows, size_t columns) {
    if (rows == 0 && columns == 0) { return; }
    if (rows == 0) {
        throw std::runtime_error{"Impossible to create matrix with 0 rows and not 0 columns"};
    }
    if (columns == 0) {
        throw std::runtime_error{"Impossible to create matrix with 0 columns and not 0 rows"};
    }
    m_ptr = new double[rows * columns];
    for (size_t i = 0; i < rows * columns; ++i) {
        m_ptr[i] = 0;
    }
    m_rows = rows;
    m_columns = columns;
}

linalg::Matrix::Matrix(const linalg::Matrix& matrix) {
    if (matrix.empty()) { return; }
    m_ptr = new double[matrix.m_rows * matrix.m_columns];
    for (size_t i = 0; i < matrix.m_rows * matrix.m_columns; ++i) {
        m_ptr[i] = matrix.m_ptr[i];
    }
    m_rows = matrix.m_rows;
    m_columns = matrix.m_columns;
}

linalg::Matrix::Matrix(linalg::Matrix&& matrix) noexcept {
    std::swap(m_ptr, matrix.m_ptr);
    std::swap(m_rows, matrix.m_rows);
    std::swap(m_columns, matrix.m_columns);
}

linalg::Matrix::Matrix(std::initializer_list<std::initializer_list<double>> list) {
    if (list.size() == 0) { return; }
    if (list.begin()->size() == 0) { throw std::runtime_error{"Incorrect input"}; }
    for (std::initializer_list<std::initializer_list<double>>::iterator it_line = list.begin();
         it_line != list.end(); ++it_line) {
        if (it_line->size() != list.begin()->size()) {
            throw std::runtime_error{"Incorrect input"};
        }
    }
    m_ptr = new double[list.size() * list.begin()->size()];
    size_t i = 0;
    for (std::initializer_list<std::initializer_list<double>>::iterator it_line = list.begin();
         it_line != list.end(); ++it_line) {
        for (std::initializer_list<double>::iterator it = it_line->begin();
             it != it_line->end(); ++it, ++i) {
            m_ptr[i] = *it;
        }
    }
    m_rows = list.size();
    m_columns = list.begin()->size();
}

linalg::Matrix::Matrix(std::initializer_list<double> list) {
    if (list.size() == 0) { return; }
    m_ptr = new double[list.size()];
    std::initializer_list<double>::iterator it = list.begin();
    for (size_t i = 0; i < list.size(); ++i) {
        m_ptr[i] = it[i];
    }
    m_rows = list.size();
    m_columns = 1;
}


linalg::Matrix& linalg::Matrix::operator=(const linalg::Matrix& matrix) {
    if (this == &matrix) {
        return *this;
    }
    if (m_ptr == nullptr && matrix.m_ptr == nullptr) {
        return *this;
    }
    if (matrix.empty()) {
        delete[] m_ptr;
        m_ptr = nullptr;
        m_rows = 0;
        m_columns = 0;
        return *this;
    }
    if (matrix.m_rows * matrix.m_columns != m_rows * m_columns) {
        double* ptr = new double[matrix.m_rows * matrix.m_columns];
        delete[] m_ptr;
        m_ptr = ptr;
    }
    for (size_t i = 0; i < matrix.m_rows * matrix.m_columns; ++i) {
        m_ptr[i] = matrix.m_ptr[i];
    }
    m_rows = matrix.m_rows;
    m_columns = matrix.m_columns;
    return *this;
}

linalg::Matrix& linalg::Matrix::operator=(linalg::Matrix&& matrix) noexcept {
    std::swap(m_ptr, matrix.m_ptr);
    std::swap(m_rows, matrix.m_rows);
    std::swap(m_columns, matrix.m_columns);
    return *this;
}

void linalg::Matrix::reshape(size_t rows, size_t columns) {
    if (m_rows == rows && m_columns == columns) { return; }
    if (m_rows * m_columns != rows * columns || rows == 0 || columns == 0) {
        throw std::runtime_error{"Incorrect shape"};
    }
    m_rows = rows;
    m_columns = columns;
}
static void update_width_in_row(const linalg::Matrix& matrix,
                                size_t i,
                                size_t& first_w, size_t& other_w,
                                const std::ios_base::fmtflags& flags) {
    std::stringstream s_str;
    s_str.flags(flags);
    s_str << matrix(i, 0);
    first_w = std::max(first_w, s_str.str().size());
    s_str.str("");

    for (size_t j = 1; j < matrix.columns(); ++j) {
        s_str << matrix(i, j);
        other_w = std::max(other_w, s_str.str().size());
        s_str.str("");
    }
}

static std::pair<size_t, size_t> compute_width(const linalg::Matrix& matrix,
                                               const std::ios_base::fmtflags& flags) {
    size_t first_w = 0, other_w = 0;
    for (size_t i = 0; i < matrix.rows(); ++i) {
        update_width_in_row(matrix, i, first_w, other_w, flags);
    }
    return {first_w, other_w};
}

std::ostream& linalg::operator<<(std::ostream& out, const linalg::Matrix& matrix) {
    if (matrix.empty()) { return out << "Matrix is empty"; }
    //*
    auto [first_w, other_w] = compute_width(matrix, out.flags());
    /*/
    std::pair<size_t, size_t> width_pair = compute_width(matrix, out.flags()); // without auto
    size_t first_w = width_pair.first, other_w = width_pair.second;
    //*/
    for (size_t i = 0; i < matrix.rows(); ++i) {
        out << '|' << std::setw(first_w) << matrix(i, 0);
        for (size_t j = 1; j < matrix.columns(); ++j) {
            out << std::setw(other_w + 1) << matrix(i, j);
        }
        out << '|';
        if (i != matrix.rows() - 1) { out << '\n'; }
    }
    return out;
}
linalg::Matrix& linalg::Matrix::operator+=(const linalg::Matrix& matrix) {
    if (m_rows != matrix.m_rows || m_columns != matrix.m_columns) {
        throw std::runtime_error{"Impossible to + these matrices"};
    }
    if (matrix.empty() && empty()) { return *this; }
    for (size_t i = 0; i < m_rows * m_columns; ++i) {
        m_ptr[i] += matrix.m_ptr[i];
    }
    return *this;
}

linalg::Matrix& linalg::Matrix::operator-=(const linalg::Matrix& matrix) {
    if (m_rows != matrix.m_rows || m_columns != matrix.m_columns) {
        throw std::runtime_error{"Impossible to - these matrices"};
    }
    if (matrix.empty() && empty()) { return *this; }
    for (size_t i = 0; i < m_rows * m_columns; ++i) {
        m_ptr[i] -= matrix.m_ptr[i];
    }
    return *this;
}
const linalg::Matrix linalg::operator+(const linalg::Matrix& m1, const linalg::Matrix& m2) {
    Matrix res = m1;
    return res += m2;
}
const linalg::Matrix linalg::operator-(const linalg::Matrix& m1, const linalg::Matrix& m2) {
    Matrix res = m1;
    return res -= m2;
}

linalg::Matrix& linalg::Matrix::operator*=(const linalg::Matrix& matrix) {
    if (m_columns != matrix.m_rows) { throw std::runtime_error{"Impossible to * these matrices"}; }
    Matrix res(m_rows, matrix.m_columns);
    for (size_t i = 0; i < m_rows; ++i) {
        for (size_t j = 0; j < matrix.m_columns; ++j) {
            for (size_t k = 0; k < m_columns; ++k) {
                res(i, j) += operator()(i, k) * matrix(k, j); // Общий вид произведения матриц.
            }
        }
    }
    *this = std::move(res);
    return *this;
}
const linalg::Matrix linalg::operator*(const linalg::Matrix& m1, const linalg::Matrix& m2) {
    Matrix res = m1;
    return res *= m2;
}
linalg::Matrix& linalg::Matrix::operator*=(double number) noexcept {
    for (size_t i = 0; i < m_rows * m_columns; ++i) {
        m_ptr[i] *= number;
    }
    return *this;
}
const linalg::Matrix linalg::operator*(const linalg::Matrix& m, double number) noexcept {
    Matrix res = m;
    return res *= number;
}

const linalg::Matrix linalg::operator*(double number, const linalg::Matrix& m) noexcept {
    Matrix res = m;
    return res *= number;
}
bool linalg::operator==(const linalg::Matrix& m1, const linalg::Matrix& m2) noexcept {
    if (m1.rows() != m2.rows() || m1.columns() != m2.columns()) {
        return false;
    }
    for (size_t i = 0; i < m1.rows(); ++i) {
        for (size_t j = 0; j < m1.columns(); ++j) {
            if (!is_equal(m1(i, j), m2(i, j))) { return false; }
        }
    }
    return true;
}

bool linalg::operator!=(const linalg::Matrix& m1, const linalg::Matrix& m2) noexcept {
    return !(m1 == m2);
}

int linalg::Matrix::find_first_not_zero_row_in_col(size_t col, size_t row_start) const noexcept {
    for (size_t row = row_start; row < m_rows; ++row) {
        if (!is_equal(operator()(row, col), 0.0)) { return row; }
    }
    return -1;
}
void linalg::Matrix::swap_rows(size_t row1, size_t row2) noexcept {
    for (size_t col = 0; col < m_columns; ++col) {
        std::swap(operator()(row1, col), operator()(row2, col));
    }
}
void linalg::Matrix::divide_row_by_a_number(size_t row, double number, size_t col_start) noexcept {
    for (size_t col = col_start; col < m_columns; ++col) {
        operator()(row, col) /= number;
    }
}
void linalg::Matrix::combine_rows(size_t take, size_t change, double ratio, size_t col_start) noexcept {
    for (size_t col = col_start; col < m_columns; ++col) {
        operator()(change, col) += ratio * operator()(take, col);
    }
}

void linalg::Matrix::make_zero_elements_under(size_t row_start, size_t col) noexcept {
    for (size_t row_under = row_start + 1; row_under < m_rows; ++row_under) {
        if (is_equal(operator()(row_under, col), 0.0)) {
            continue;
        }
        combine_rows(row_start, row_under, -operator()(row_under, col), col);
    }
}
void linalg::Matrix::make_zero_elements_above(size_t row_start, size_t col) noexcept {
    for (size_t row_above = 0; row_above < row_start; ++row_above) {
        if (is_equal(operator()(row_above, col), 0.0)) {
            continue;
        }
        combine_rows(row_start, row_above, -operator()(row_above, col), col);
    }
}
double linalg::Matrix::norm() const noexcept {
    double res = 0.0;
    for (size_t i = 0; i < m_rows * m_columns; ++i) {
        res += m_ptr[i] * m_ptr[i];
    }
    return sqrt(res);
}
double linalg::Matrix::trace() const {
    if (m_rows != m_columns) { throw std::runtime_error{"Impossible to compute trace"}; }
    double res = 0.0;
    for (size_t i = 0; i < m_rows; ++i) {
        res += operator()(i, i);
    }
    return res;
}
double linalg::Matrix::det() const {
    if (empty()) { throw std::runtime_error{"Impossible to compute det of empty matrix"}; }
    if (m_rows != m_columns) {
        throw std::runtime_error{"Impossible to compute det of not squared matrix"};
    }
    if (m_rows == 1) { return operator()(0, 0); }
    if (m_rows == 2) {
        return operator()(0, 0) * operator()(1, 1) - operator()(1, 0) * operator()(0, 1);
    }
    Matrix tmp = *this;
    bool sign = true;
    double res = 1;
    for (size_t row = 0; row < tmp.rows(); ++row) {
        int row_not_zero = tmp.find_first_not_zero_row_in_col(row, row); // ищем ненулевую строку
        if (row_not_zero == -1) {
            return 0; // найден столбец из нулей
        }
        if (row != row_not_zero) {
            tmp.swap_rows(row, row_not_zero); // ставим ненулевую строку на хорошее место
            sign = !sign;
        }
        res *= tmp(row, row);

        tmp.divide_row_by_a_number(row, tmp(row, row), row);
        tmp.make_zero_elements_under(row, row); // обнуляем всё снизу
    }
    res *= tmp(0, 0);
    for (size_t i = 1; i < tmp.rows(); ++i) {
        res *= tmp(i, i);
    }
    return sign ? res : -res;
}
linalg::Matrix& linalg::Matrix::gauss_forward() noexcept {
    for (size_t col = 0, row = 0; col < m_columns && row < m_rows; ++col) { // перебор столбцов
        size_t row_not_zero = find_first_not_zero_row_in_col(col, row); // строка с ненулевым элементом
        if (row_not_zero == -1) {
            continue; // если такой нет, то столбец плохой, идем дальше
        }
        if (row_not_zero != row) {
            swap_rows(row, row_not_zero); // ставим хорошую строку на место
        }
        divide_row_by_a_number(row, operator()(row, col), col); // нормируем хорошую строку
        make_zero_elements_under(row, col); // обнуляем всё снизу
        ++row;
    }
    return *this;
}
linalg::Matrix& linalg::Matrix::gauss_backward() {
    for (size_t col = 0, row = 0; col < m_columns && row < m_rows; ++col) { // перебор столбцов
        size_t row_not_zero = find_first_not_zero_row_in_col(col, row); // строка с ненулевым элементом
        if (row_not_zero == -1) {
            continue;
        }
        if (row_not_zero != row) {
            throw std::runtime_error{"Matrix is not triangular, so make gauss_forward first"};
        }
        if (!is_equal(operator()(row, col), 1.0)) {
            throw std::runtime_error{"Top element is not 1, so make gauss_forward first"};
        }
        make_zero_elements_above(row, col); // обнуляем всё сверху
        ++row;
    }
    return *this;
}

size_t linalg::Matrix::rank() const noexcept {
    Matrix tmp = *this;
    tmp.gauss_forward();
    size_t r = 0, row = 0;
    for (size_t col = 0; col < m_columns; ++col) {
        for (; row < m_rows; ++row) {
            if (is_equal(tmp(row, col), 1.0)) {
                ++r;
                ++row;
                break;
            }
        }
    }
    return r;
}

linalg::Matrix linalg::concatenate(const linalg::Matrix& m1, const linalg::Matrix& m2) {
    if (m1.rows() != m2.rows()) {
        throw std::runtime_error{"Invalid shape to concatenate"};
    }
    Matrix res(m1.rows(), m1.columns() + m2.columns());
    for (size_t i = 0; i < res.rows(); ++i) {
        for (size_t j = 0; j < m1.columns(); ++j) {
            res(i, j) = m1(i, j);
        }
        for (size_t j = m1.columns(); j < res.columns(); ++j) {
            res(i, j) = m2(i, j - m1.columns());
        }
    }
    return res;
}
linalg::Matrix linalg::transpose(const linalg::Matrix& m) {
    Matrix res(m.columns(), m.rows());
    for (size_t i = 0; i < res.rows(); ++i) {
        for (size_t j = 0; j < res.columns(); ++j) {
            res(i, j) = m(j, i);
        }
    }
    return res;
}
linalg::Matrix linalg::invert(const linalg::Matrix& m) {
    if (m.rows() != m.columns()) {
        throw std::runtime_error{"Impossible to invert not squared matrix"};
    }
    if (is_equal(m.det(), 0.0)) {
        throw std::runtime_error{"Impossible to invert matrix with det = 0"};
    }
    Matrix inverted(m.rows(), m.rows());
    for (size_t i = 0; i < inverted.rows(); ++i) { inverted(i, i) = 1; }
    Matrix gauss = concatenate(m, inverted);
    gauss.gauss_forward();
    gauss.gauss_backward();
    for (size_t row = 0; row < m.rows(); ++row) {
        for (size_t col = 0; col < m.rows(); ++col) {
            inverted(row, col) = gauss(row, col + m.rows());
        }
    }
    return inverted;
}
linalg::Matrix linalg::power(const linalg::Matrix& m, int power) {
    if (m.rows() != m.columns()) {
        throw std::runtime_error{"Invalid shape to power matrix"};
    }
    Matrix res(m.rows(), m.rows());
    if (power == 0) {
        for (size_t i = 0; i < res.rows(); ++i) { res(i, i) = 1; }
        return res;
    }
    res = m;
    for (size_t i = 1; i < std::abs(power); ++i) {
        res *= m;
    }
    return power > 0 ? res : invert(res);
}

static int row_zero_check(size_t row, const linalg::Matrix& system) noexcept {
    if (!is_equal(system(row, system.columns() - 1), 0.0)) {
        for (size_t col = 0; col < system.columns() - 1; ++col) {
            if (!is_equal(system(row, col), 0.0)) {
                return 1; // нашли ненулевой с края и в матрице
            }
        }
        return -1; // нашли ненулевой с края, но в матрице нули
    }
    return 0; // вся строка нулевая
}

linalg::Matrix linalg::solve(const linalg::Matrix& A, const linalg::Matrix& b) {
    if (b.columns() != 1) {
        throw std::runtime_error{"Invalid vector provided (should be 1 column)!"};
    }
    if (A.rows() < A.columns()) {
        throw std::runtime_error{"Invalid shape of A: there are inf or zero solutions"};
    }
    Matrix system = concatenate(A, b);
    system.gauss_forward();
    system.gauss_backward();
    for (int row = system.rows() - 1; row >= 0; --row) {
        int row_info = row_zero_check(row, system);
        if (row_info == -1) { throw std::runtime_error{"No solution"}; }
        if (row_info == 1) { break; }
    }
    Matrix x(A.columns());
    for (size_t row = 0; row < x.rows(); ++row) {
        x(row, 0) = system(row, system.columns() - 1);
    }
    return x;
}
