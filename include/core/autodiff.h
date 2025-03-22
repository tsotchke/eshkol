/**
 * @file autodiff.h
 * @brief Automatic differentiation system for Eshkol
 * 
 * This file defines the automatic differentiation interface for Eshkol,
 * which provides forward-mode and reverse-mode automatic differentiation for scalar and vector functions.
 */

#ifndef ESHKOL_AUTODIFF_H
#define ESHKOL_AUTODIFF_H

#include "core/memory.h"
#include "core/vector.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Dual number for forward-mode automatic differentiation
 * 
 * A dual number consists of a value and a derivative part.
 * It is used to compute derivatives of functions automatically.
 */
typedef struct {
    float value;      /**< Value part */
    float derivative; /**< Derivative part */
} DualNumber;

/**
 * @brief Create a dual number
 * 
 * @param value Value part
 * @param derivative Derivative part
 * @return Dual number
 */
DualNumber dual_number_create(float value, float derivative);

/**
 * @brief Create a dual number from a constant
 * 
 * The derivative part is set to 0.
 * 
 * @param value Value part
 * @return Dual number
 */
DualNumber dual_number_from_constant(float value);

/**
 * @brief Create a dual number from a variable
 * 
 * The derivative part is set to 1.
 * 
 * @param value Value part
 * @return Dual number
 */
DualNumber dual_number_from_variable(float value);

/**
 * @brief Add two dual numbers
 * 
 * @param a First dual number
 * @param b Second dual number
 * @return Result of addition
 */
DualNumber dual_number_add(DualNumber a, DualNumber b);

/**
 * @brief Subtract two dual numbers
 * 
 * @param a First dual number
 * @param b Second dual number
 * @return Result of subtraction
 */
DualNumber dual_number_sub(DualNumber a, DualNumber b);

/**
 * @brief Multiply two dual numbers
 * 
 * @param a First dual number
 * @param b Second dual number
 * @return Result of multiplication
 */
DualNumber dual_number_mul(DualNumber a, DualNumber b);

/**
 * @brief Divide two dual numbers
 * 
 * @param a First dual number
 * @param b Second dual number
 * @return Result of division
 */
DualNumber dual_number_div(DualNumber a, DualNumber b);

/**
 * @brief Compute the sine of a dual number
 * 
 * @param a Dual number
 * @return Sine of the dual number
 */
DualNumber dual_number_sin(DualNumber a);

/**
 * @brief Compute the cosine of a dual number
 * 
 * @param a Dual number
 * @return Cosine of the dual number
 */
DualNumber dual_number_cos(DualNumber a);

/**
 * @brief Compute the exponential of a dual number
 * 
 * @param a Dual number
 * @return Exponential of the dual number
 */
DualNumber dual_number_exp(DualNumber a);

/**
 * @brief Compute the natural logarithm of a dual number
 * 
 * @param a Dual number
 * @return Natural logarithm of the dual number
 */
DualNumber dual_number_log(DualNumber a);

/**
 * @brief Compute the power of a dual number
 * 
 * @param a Base dual number
 * @param b Exponent (constant)
 * @return Power of the dual number
 */
DualNumber dual_number_pow(DualNumber a, float b);

/**
 * @brief Dual vector for forward-mode automatic differentiation
 * 
 * A dual vector consists of a value vector and a derivative vector.
 * It is used to compute derivatives of vector functions automatically.
 */
typedef struct {
    VectorF* value;      /**< Value vector */
    VectorF* derivative; /**< Derivative vector */
} DualVector;

/**
 * @brief Create a dual vector
 * 
 * @param arena Arena for allocations
 * @param value Value vector
 * @param derivative Derivative vector
 * @return Dual vector
 */
DualVector* dual_vector_create(Arena* arena, VectorF* value, VectorF* derivative);

/**
 * @brief Create a dual vector from a constant vector
 * 
 * The derivative vector is set to zero.
 * 
 * @param arena Arena for allocations
 * @param value Value vector
 * @return Dual vector
 */
DualVector* dual_vector_from_constant(Arena* arena, VectorF* value);

/**
 * @brief Create a dual vector from a variable vector
 * 
 * The derivative vector is set to the identity matrix.
 * 
 * @param arena Arena for allocations
 * @param value Value vector
 * @return Dual vector
 */
DualVector* dual_vector_from_variable(Arena* arena, VectorF* value);

/**
 * @brief Add two dual vectors
 * 
 * @param arena Arena for allocations
 * @param a First dual vector
 * @param b Second dual vector
 * @return Result of addition
 */
DualVector* dual_vector_add(Arena* arena, DualVector* a, DualVector* b);

/**
 * @brief Subtract two dual vectors
 * 
 * @param arena Arena for allocations
 * @param a First dual vector
 * @param b Second dual vector
 * @return Result of subtraction
 */
DualVector* dual_vector_sub(Arena* arena, DualVector* a, DualVector* b);

/**
 * @brief Multiply a dual vector by a scalar
 * 
 * @param arena Arena for allocations
 * @param a Dual vector
 * @param b Scalar
 * @return Result of multiplication
 */
DualVector* dual_vector_mul_scalar(Arena* arena, DualVector* a, float b);

/**
 * @brief Compute the dot product of two dual vectors
 * 
 * @param a First dual vector
 * @param b Second dual vector
 * @return Dot product as a dual number
 */
DualNumber dual_vector_dot(DualVector* a, DualVector* b);

/**
 * @brief Compute the cross product of two dual vectors
 * 
 * @param arena Arena for allocations
 * @param a First dual vector
 * @param b Second dual vector
 * @return Cross product as a dual vector
 */
DualVector* dual_vector_cross(Arena* arena, DualVector* a, DualVector* b);

/**
 * @brief Compute the magnitude of a dual vector
 * 
 * @param a Dual vector
 * @return Magnitude as a dual number
 */
DualNumber dual_vector_magnitude(DualVector* a);

/**
 * @brief Compute the gradient of a scalar function
 * 
 * @param arena Arena for allocations
 * @param f Function to differentiate (takes a VectorF* and returns a float)
 * @param x Point at which to compute the gradient
 * @return Gradient vector
 */
VectorF* compute_gradient_autodiff(Arena* arena, float (*f)(VectorF*), VectorF* x);

/**
 * @brief Compute the Jacobian matrix of a vector function
 * 
 * @param arena Arena for allocations
 * @param f Function to differentiate (takes a VectorF* and returns a VectorF*)
 * @param x Point at which to compute the Jacobian
 * @return Jacobian matrix as an array of VectorF*
 */
VectorF** compute_jacobian(Arena* arena, VectorF* (*f)(Arena*, VectorF*), VectorF* x);

/**
 * @brief Compute the Hessian matrix of a scalar function
 * 
 * @param arena Arena for allocations
 * @param f Function to differentiate (takes a VectorF* and returns a float)
 * @param x Point at which to compute the Hessian
 * @return Hessian matrix as an array of VectorF*
 */
VectorF** compute_hessian(Arena* arena, float (*f)(VectorF*), VectorF* x);

/**
 * @brief Compute the nth derivative of a scalar function of one variable
 * 
 * @param arena Arena for allocations
 * @param f Function to differentiate (takes a VectorF* and returns a float)
 * @param x Point at which to compute the derivative
 * @param order Order of the derivative (1 for first derivative, 2 for second, etc.)
 * @return nth derivative value
 */
float compute_nth_derivative(Arena* arena, float (*f)(VectorF*), float x, size_t order);

/**
 * @brief Compute a tensor of higher-order derivatives of a scalar function
 * 
 * @param arena Arena for allocations
 * @param f Function to differentiate (takes a VectorF* and returns a float)
 * @param x Point at which to compute the derivatives
 * @param order Order of the derivatives (1 for gradient, 2 for Hessian, etc.)
 * @return Tensor of higher-order derivatives
 */
void* compute_derivative_tensor(Arena* arena, float (*f)(VectorF*), VectorF* x, size_t order);

/**
 * @brief Computational graph node types for reverse-mode automatic differentiation
 */
typedef enum {
    NODE_CONSTANT,    /**< Constant value */
    NODE_VARIABLE,    /**< Input variable */
    NODE_ADD,         /**< Addition operation */
    NODE_SUB,         /**< Subtraction operation */
    NODE_MUL,         /**< Multiplication operation */
    NODE_DIV,         /**< Division operation */
    NODE_SIN,         /**< Sine operation */
    NODE_COS,         /**< Cosine operation */
    NODE_EXP,         /**< Exponential operation */
    NODE_LOG,         /**< Natural logarithm operation */
    NODE_POW          /**< Power operation */
} ComputationalNodeType;

/**
 * @brief Computational graph node for reverse-mode automatic differentiation
 */
typedef struct ComputationalNode {
    ComputationalNodeType type;     /**< Node type */
    float value;                    /**< Forward pass value */
    float gradient;                 /**< Backward pass gradient */
    struct ComputationalNode* left; /**< Left child (first operand) */
    struct ComputationalNode* right;/**< Right child (second operand) */
    float constant;                 /**< Constant value (for NODE_CONSTANT and NODE_POW) */
} ComputationalNode;

/**
 * @brief Computational graph for reverse-mode automatic differentiation
 */
typedef struct {
    Arena* arena;                   /**< Arena for allocations */
    ComputationalNode* root;        /**< Root node of the graph */
    ComputationalNode** variables;  /**< Array of variable nodes */
    size_t num_variables;           /**< Number of variable nodes */
} ComputationalGraph;

/**
 * @brief Create a computational graph
 * 
 * @param arena Arena for allocations
 * @param num_variables Number of input variables
 * @return Computational graph
 */
ComputationalGraph* computational_graph_create(Arena* arena, size_t num_variables);

/**
 * @brief Create a constant node in the computational graph
 * 
 * @param graph Computational graph
 * @param value Constant value
 * @return Constant node
 */
ComputationalNode* computational_node_constant(ComputationalGraph* graph, float value);

/**
 * @brief Create a variable node in the computational graph
 * 
 * @param graph Computational graph
 * @param index Variable index
 * @param value Initial value
 * @return Variable node
 */
ComputationalNode* computational_node_variable(ComputationalGraph* graph, size_t index, float value);

/**
 * @brief Create an addition node in the computational graph
 * 
 * @param graph Computational graph
 * @param left Left operand
 * @param right Right operand
 * @return Addition node
 */
ComputationalNode* computational_node_add(ComputationalGraph* graph, ComputationalNode* left, ComputationalNode* right);

/**
 * @brief Create a subtraction node in the computational graph
 * 
 * @param graph Computational graph
 * @param left Left operand
 * @param right Right operand
 * @return Subtraction node
 */
ComputationalNode* computational_node_sub(ComputationalGraph* graph, ComputationalNode* left, ComputationalNode* right);

/**
 * @brief Create a multiplication node in the computational graph
 * 
 * @param graph Computational graph
 * @param left Left operand
 * @param right Right operand
 * @return Multiplication node
 */
ComputationalNode* computational_node_mul(ComputationalGraph* graph, ComputationalNode* left, ComputationalNode* right);

/**
 * @brief Create a division node in the computational graph
 * 
 * @param graph Computational graph
 * @param left Left operand
 * @param right Right operand
 * @return Division node
 */
ComputationalNode* computational_node_div(ComputationalGraph* graph, ComputationalNode* left, ComputationalNode* right);

/**
 * @brief Create a sine node in the computational graph
 * 
 * @param graph Computational graph
 * @param input Input node
 * @return Sine node
 */
ComputationalNode* computational_node_sin(ComputationalGraph* graph, ComputationalNode* input);

/**
 * @brief Create a cosine node in the computational graph
 * 
 * @param graph Computational graph
 * @param input Input node
 * @return Cosine node
 */
ComputationalNode* computational_node_cos(ComputationalGraph* graph, ComputationalNode* input);

/**
 * @brief Create an exponential node in the computational graph
 * 
 * @param graph Computational graph
 * @param input Input node
 * @return Exponential node
 */
ComputationalNode* computational_node_exp(ComputationalGraph* graph, ComputationalNode* input);

/**
 * @brief Create a natural logarithm node in the computational graph
 * 
 * @param graph Computational graph
 * @param input Input node
 * @return Natural logarithm node
 */
ComputationalNode* computational_node_log(ComputationalGraph* graph, ComputationalNode* input);

/**
 * @brief Create a power node in the computational graph
 * 
 * @param graph Computational graph
 * @param input Input node
 * @param exponent Exponent value
 * @return Power node
 */
ComputationalNode* computational_node_pow(ComputationalGraph* graph, ComputationalNode* input, float exponent);

/**
 * @brief Perform forward pass on the computational graph
 * 
 * @param graph Computational graph
 * @param values Input values
 * @return Output value
 */
float computational_graph_forward(ComputationalGraph* graph, float* values);

/**
 * @brief Perform backward pass on the computational graph
 * 
 * @param graph Computational graph
 * @return Gradient vector
 */
VectorF* computational_graph_backward(ComputationalGraph* graph);

/**
 * @brief Compute the gradient of a scalar function using reverse-mode automatic differentiation
 * 
 * @param arena Arena for allocations
 * @param f Function to differentiate (takes a VectorF* and returns a float)
 * @param x Point at which to compute the gradient
 * @return Gradient vector
 */
VectorF* compute_gradient_reverse_mode(Arena* arena, float (*f)(VectorF*), VectorF* x);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_AUTODIFF_H */
