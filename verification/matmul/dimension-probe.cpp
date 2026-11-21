#include "eshkol/types/dependent.h"
#include <iostream>

using namespace eshkol::hott;

static DependentType matrix(CTValue rows, CTValue cols) {
    DependentType operand{};
    operand.base = BuiltinTypes::Tensor;
    operand.value_indices = {rows, cols};
    return operand;
}

static void observe(const DependentType& left, const DependentType& right) {
    auto result = DimensionChecker::checkMatMulDimensions(left, right, "matmul");
    std::cout << (result.valid ? "compatible" : result.error_message) << '\n';
}

int main() {
    unsigned rows, leftCols, rightRows, cols;
    while (std::cin >> rows >> leftCols >> rightRows >> cols) {
        observe(matrix(CTValue::makeNat(rows), CTValue::makeNat(leftCols)),
                matrix(CTValue::makeNat(rightRows), CTValue::makeNat(cols)));
    }
    observe(matrix(CTValue::makeNat(2), CTValue::makeNat(3)),
            matrix(CTValue::makeUnknown(), CTValue::makeNat(4)));
}
