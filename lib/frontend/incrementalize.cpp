/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#include <eshkol/frontend/incrementalize.h>
#include <algorithm>
#include <sstream>

namespace eshkol::frontend {
std::string IncrementalizePass::source(const DifferentiationGraph& graph) {
    using Op = DifferentiationOperation;
    const size_t count = graph.nodes.size(), width = graph.parameters.size();
    if (!width || width > DifferentiationGraph::max_inputs || !count ||
        count > DifferentiationGraph::max_nodes || graph.output >= count)
        throw std::logic_error("invalid graph passed to IncrementalizePass");
    std::vector<std::vector<size_t>> consumers(count);
    size_t edges = 0, kernels = 0, capacity = 1, height = 0;
    while (capacity < count) { capacity *= 2; ++height; }
    for (size_t i = 0; i < count; ++i) {
        const auto& node = graph.nodes[i];
        if ((i < width && (node.operation != Op::Input || node.input_index != i)) ||
            (i >= width && node.operation == Op::Input) || count < width)
            throw std::logic_error("incremental graph inputs are not a parameter prefix");
        if (!differentiation_rule(node.operation).incremental)
            throw std::logic_error("missing incremental graph rule");
        if (node.inputs.size() != differentiation_rule(node.operation).arity)
            throw std::logic_error("invalid incremental graph arity");
        if (!node.inputs.empty()) ++kernels;
        for (auto parent : node.inputs) {
            if (parent >= i) throw std::logic_error("incremental graph is not topological");
            ++edges;
            auto& list = consumers[parent];
            if (list.empty() || list.back() != i) list.push_back(i);
        }
    }
    // Context: baseline, kernels, consumers, capacity, width, output, info,
    //          read, write, minimum, check, factory, enqueue, refresh.
    // Tick: context, persistent values, queue, owned work receipt.
    // Work: kernels, changed values, trie visits, copied trie slots,
    //       enqueue attempts, inspected input slots.
    std::ostringstream out;
    out << "(lambda (__ig_point) "
        << "(if (not (and (vector? __ig_point) (= (vector-length __ig_point) " << width
        << "))) (error \"differentiate incremental point width mismatch\")) "
        << "(let* ((__ig_base (make-vector " << count << " 0)) (__ig_kernels (vector ";
    for (const auto& node : graph.nodes) {
        if (node.inputs.empty()) out << "#f ";
        else out << "(lambda (__ig_get) (" << differentiation_rule(node.operation).spelling
                 << " (__ig_get " << node.inputs[0] << ") (__ig_get " << node.inputs[1] << "))) ";
    }
    out << ")) (__ig_children (vector ";
    for (const auto& list : consumers) {
        out << "(vector ";
        for (size_t child : list) out << child << ' ';
        out << ") ";
    }
    out << ")) (__ig_context (vector __ig_base __ig_kernels __ig_children " << capacity << ' '
        << width << ' ' << graph.output << " (vector " << count << ' ' << edges << ' ' << width << ' '
        << height << ") #f #f #f #f #f #f #f))) ";
    out << R"ESK(
      (letrec (
        (__ig_read (lambda (tree lo hi index base work)
          (vector-set! work 2 (+ (vector-ref work 2) 1))
          (if (not tree) (vector-ref base index)
            (if (= (- hi lo) 1) tree
              (let ((mid (quotient (+ lo hi) 2)))
                (if (< index mid) (__ig_read (vector-ref tree 0) lo mid index base work)
                    (__ig_read (vector-ref tree 1) mid hi index base work)))))))
        (__ig_write (lambda (tree lo hi index value work)
          (vector-set! work 2 (+ (vector-ref work 2) 1))
          (if (= (- hi lo) 1) value
            (let* ((left (if tree (vector-ref tree 0) #f))
                   (right (if tree (vector-ref tree 1) #f))
                   (mid (quotient (+ lo hi) 2))
                   (next-left (if (< index mid) (__ig_write left lo mid index value work) left))
                   (next-right (if (< index mid) right (__ig_write right mid hi index value work))))
              (if (and (eq? left next-left) (eq? right next-right)) tree
                (if (and (not next-left) (not next-right)) #f
                  (begin (vector-set! work 3 (+ (vector-ref work 3) 2))
                         (vector next-left next-right))))))))
        (__ig_minimum (lambda (tree lo hi work)
          (vector-set! work 2 (+ (vector-ref work 2) 1))
          (if (= (- hi lo) 1) lo
            (let ((mid (quotient (+ lo hi) 2)))
              (if (vector-ref tree 0) (__ig_minimum (vector-ref tree 0) lo mid work)
                  (__ig_minimum (vector-ref tree 1) mid hi work))))))
        (__ig_check (lambda (value)
          (let ((kind (type-name value)))
            (if (and (or (eq? kind 'integer) (eq? kind 'rational) (eq? kind 'real))
                     (or (exact? value) (finite? value))) value
                (error "differentiate incremental requires ordinary finite real values")))))
        (__ig_enqueue (lambda (tick index)
          (let* ((c (vector-ref tick 0)) (work (vector-ref tick 3))
                 (children (vector-ref (vector-ref c 2) index)))
            (let loop ((i 0)) (if (< i (vector-length children))
              (begin (vector-set! work 4 (+ (vector-ref work 4) 1))
                (vector-set! tick 2 ((vector-ref c 8) (vector-ref tick 2) 0 (vector-ref c 3)
                                    (vector-ref children i) #t work))
                (loop (+ i 1))))))))
        (__ig_refresh (lambda (tick index fresh)
          (let* ((c (vector-ref tick 0)) (work (vector-ref tick 3))
                 (old ((vector-ref c 7) (vector-ref tick 1) 0 (vector-ref c 3) index (vector-ref c 0) work))
                 (value ((vector-ref c 10) fresh)))
            (if (not (eqv? old value))
              (begin (vector-set! tick 1 ((vector-ref c 8) (vector-ref tick 1) 0 (vector-ref c 3) index value work))
                (vector-set! work 1 (+ (vector-ref work 1) 1))
                ((vector-ref c 12) tick index))))))
        (__ig_factory (lambda (c tree receipt)
          (lambda (action payload)
            (cond
              ((eq? action 'output) ((vector-ref c 7) tree 0 (vector-ref c 3) (vector-ref c 5)
                                    (vector-ref c 0) (vector 0 0 0 0 0 0)))
              ((eq? action 'work) (vector (vector-ref receipt 0) (vector-ref receipt 1)
                 (vector-ref receipt 2) (vector-ref receipt 3) (vector-ref receipt 4) (vector-ref receipt 5)))
              ((eq? action 'info) (let ((info (vector-ref c 6)))
                (vector (vector-ref info 0) (vector-ref info 1) (vector-ref info 2) (vector-ref info 3))))
              ((eq? action 'step)
                (if (not (and (vector? payload) (= (vector-length payload) (vector-ref c 4))))
                    (error "differentiate incremental delta width mismatch"))
                (let* ((work (vector 0 0 0 0 0 0)) (tick (vector c tree #f work))
                       (before ((vector-ref c 7) tree 0 (vector-ref c 3) (vector-ref c 5) (vector-ref c 0) work)))
                  (let inputs ((i 0)) (if (< i (vector-ref c 4))
                    (let* ((delta ((vector-ref c 10) (vector-ref payload i)))
                           (old ((vector-ref c 7) (vector-ref tick 1) 0 (vector-ref c 3) i (vector-ref c 0) work)))
                      (vector-set! work 5 (+ (vector-ref work 5) 1))
                      ((vector-ref c 13) tick i (+ old delta)) (inputs (+ i 1)))))
                  (let drain () (if (vector-ref tick 2)
                    (let ((index ((vector-ref c 9) (vector-ref tick 2) 0 (vector-ref c 3) work)))
                      (vector-set! tick 2 ((vector-ref c 8) (vector-ref tick 2) 0 (vector-ref c 3) index #f work))
                      (vector-set! work 0 (+ (vector-ref work 0) 1))
                      ((vector-ref c 13) tick index
                        ((vector-ref (vector-ref c 1) index)
                          (lambda (parent) ((vector-ref c 7) (vector-ref tick 1) 0 (vector-ref c 3)
                                             parent (vector-ref c 0) work))))
                      (drain))))
                  (let* ((after ((vector-ref c 7) (vector-ref tick 1) 0 (vector-ref c 3)
                                                (vector-ref c 5) (vector-ref c 0) work))
                         (delta ((vector-ref c 10) (- after before))))
                    (vector ((vector-ref c 11) c (vector-ref tick 1) work) delta
                      (vector (vector-ref work 0) (vector-ref work 1) (vector-ref work 2)
                              (vector-ref work 3) (vector-ref work 4) (vector-ref work 5))))))
              (else (error "unknown differentiate incremental state action")))))))
        (vector-set! __ig_context 7 __ig_read)
        (vector-set! __ig_context 8 __ig_write)
        (vector-set! __ig_context 9 __ig_minimum)
        (vector-set! __ig_context 10 __ig_check)
        (vector-set! __ig_context 11 __ig_factory)
        (vector-set! __ig_context 12 __ig_enqueue)
        (vector-set! __ig_context 13 __ig_refresh)
    )ESK";
    // Initial evaluation is a compiled let spine. Only initialization touches
    // every node; ticks path-copy trie branches and execute queued kernels.
    out << "(let* (";
    for (size_t i = 0; i < count; ++i) {
        const auto& node = graph.nodes[i];
        out << "(__ig_" << i << " (__ig_check ";
        if (node.operation == Op::Input) out << "(vector-ref __ig_point " << node.input_index << ')';
        else if (node.operation == Op::Constant) out << node.literal;
        else out << '(' << differentiation_rule(node.operation).spelling
                 << " __ig_" << node.inputs[0] << " __ig_" << node.inputs[1] << ')';
        out << "))";
    }
    out << ") ";
    for (size_t i = 0; i < count; ++i) out << "(vector-set! __ig_base " << i << " __ig_" << i << ") ";
    out << "(__ig_factory __ig_context #f (vector " << kernels << ' ' << count << " 0 0 0 " << width << "))))))";
    return out.str();
}
}
