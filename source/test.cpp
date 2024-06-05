#include "test.h"
#include "matrix.h"

using namespace linalg;

void test() {
    Matrix m;
    Matrix m1;
    m1 = m;

    Matrix A{{1, 2, 1},
             {2, 1, 2},
             {3, 3, 1},
             {1, 2, 1}};

    Matrix f{8, 10, 12, 8};
    Matrix result{1, 2, 3};

    std::cout << std::showpoint << A;
}

void run_all_tests() {
    test();
}