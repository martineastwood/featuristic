//! Built-in symbolic operations

use crate::tree::SymbolicOp;

/// Get the default set of built-in operations
pub fn default_builtins() -> Vec<SymbolicOp> {
    vec![
        // Binary operations (id 0-5)
        SymbolicOp {
            name: "add".to_string(),
            arity: 2,
            format_str: "({} + {})".to_string(),
            op_id: 0,
        },
        SymbolicOp {
            name: "subtract".to_string(),
            arity: 2,
            format_str: "({} - {})".to_string(),
            op_id: 1,
        },
        SymbolicOp {
            name: "multiply".to_string(),
            arity: 2,
            format_str: "({} * {})".to_string(),
            op_id: 2,
        },
        SymbolicOp {
            name: "divide".to_string(),
            arity: 2,
            format_str: "({} / {})".to_string(),
            op_id: 3,
        },
        SymbolicOp {
            name: "min".to_string(),
            arity: 2,
            format_str: "min({}, {})".to_string(),
            op_id: 4,
        },
        SymbolicOp {
            name: "max".to_string(),
            arity: 2,
            format_str: "max({}, {})".to_string(),
            op_id: 5,
        },

        // Unary operations (id 6-15)
        SymbolicOp {
            name: "sin".to_string(),
            arity: 1,
            format_str: "sin({})".to_string(),
            op_id: 6,
        },
        SymbolicOp {
            name: "cos".to_string(),
            arity: 1,
            format_str: "cos({})".to_string(),
            op_id: 7,
        },
        SymbolicOp {
            name: "tan".to_string(),
            arity: 1,
            format_str: "tan({})".to_string(),
            op_id: 8,
        },
        SymbolicOp {
            name: "exp".to_string(),
            arity: 1,
            format_str: "exp({})".to_string(),
            op_id: 9,
        },
        SymbolicOp {
            name: "log".to_string(),
            arity: 1,
            format_str: "log({})".to_string(),
            op_id: 10,
        },
        SymbolicOp {
            name: "sqrt".to_string(),
            arity: 1,
            format_str: "sqrt({})".to_string(),
            op_id: 11,
        },
        SymbolicOp {
            name: "abs".to_string(),
            arity: 1,
            format_str: "abs({})".to_string(),
            op_id: 12,
        },
        SymbolicOp {
            name: "neg".to_string(),
            arity: 1,
            format_str: "(-{})".to_string(),
            op_id: 13,
        },
        SymbolicOp {
            name: "square".to_string(),
            arity: 1,
            format_str: "({}^2)".to_string(),
            op_id: 14,
        },
        SymbolicOp {
            name: "cube".to_string(),
            arity: 1,
            format_str: "({}^3)".to_string(),
            op_id: 15,
        },

        // Ternary operations (id 16)
        SymbolicOp {
            name: "clip".to_string(),
            arity: 3,
            format_str: "clip({}, {}, {})".to_string(),
            op_id: 16,
        },
    ]
}
