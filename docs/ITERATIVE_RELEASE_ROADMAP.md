# Eshkol Iterative Release Roadmap: v1.0 → AI Training Framework

**Current Status**: v1.0-foundation nearly ready (110/110 tests passing)  
**Strategy**: Iterative releases, each adding value  
**Timeline**: Aligned with actual project status

---

## v1.0-foundation (4-6 weeks) - CURRENT TARGET

### What's Already Done ✅
- Core language implementation (67/67 tests passing)
- Autodiff system complete (43/43 tests passing)
- List operations working
- Higher-order functions working
- Memory management stable
- **Total: 110/110 tests passing**

### Remaining for v1.0-foundation Release
**33 tasks, ~30 sessions, 4-6 weeks**

**Phase 1: Examples & Docs** (13 tasks)
- Audit ~100 existing examples
- Select and update 30 best examples
- Fix documentation (remove false claims)
- Create quick start guide

**Phase 2: Infrastructure** (10 tasks)
- GitHub Actions CI/CD
- Debian packaging (.deb)
- Docker build system
- Homebrew formula

**Phase 3: Release** (10 tasks)
- Integration tests
- Memory validation (Valgrind)
- Performance benchmarks
- Release documentation

### v1.0-foundation Deliverables
✅ Stable Scheme compiler with LLVM backend  
✅ Mixed-type lists (int64 + double)  
✅ 17 higher-order functions  
✅ Complete autodiff suite  
✅ 30 working examples  
✅ CI/CD automation  
✅ Installable packages  
✅ Production-ready for research

**NOT included in v1.0:** Neural network training (that's v1.1+)

---

## v1.1-neural (Post v1.0, ~2-3 months)

### Purpose
**Enable basic neural network training**

### The 7 Critical Bugs to Fix
All identified in [`docs/NEURAL_NETWORK_TECHNICAL_FIXES_SPECIFICATION.md`](docs/NEURAL_NETWORK_TECHNICAL_FIXES_SPECIFICATION.md:1):

1. Derivative type check (4 hours) - [`llvm_codegen.cpp:8724`](lib/backend/llvm_codegen.cpp:8724)
2. Local function scope (2 hours) - [`llvm_codegen.cpp:5924`](lib/backend/llvm_codegen.cpp:5924)
3. `let*` implementation (4 hours) - `parser.cpp` + `llvm_codegen.cpp`
4. Builtin functions as values (3 hours) - [`llvm_codegen.cpp:3596`](lib/backend/llvm_codegen.cpp:3596)
5. Function refs in gradient (1 hour) - [`llvm_codegen.cpp:12575`](lib/backend/llvm_codegen.cpp:12575)
6. Lambda closures in map (6 hours) - [`llvm_codegen.cpp:12722`](lib/backend/llvm_codegen.cpp:12722)
7. `for` loop construct (8 hours) - New implementation

**Total**: ~30 hours of focused work

### What v1.1-neural Enables
```scheme
; This will WORK in v1.1:
(define (train-perceptron w data epochs lr)
  (for epoch 1 epochs
    (let* ((x (car data))
           (y (cadr data))
           (loss (lambda (w-val) ...))
           (grad (derivative loss w)))
      (set! w (- w (* lr grad))))))
```

### v1.1-neural Deliverables
✅ Iterative training loops work  
✅ Local functions with autodiff  
✅ Sequential bindings (`let*`)  
✅ Activation functions as parameters  
✅ Can train simple perceptrons  
✅ 10+ neural network examples

---

## v1.2-ml-toolkit (Post v1.1, ~3-4 months)

### Purpose
**Production neural network training**

### Features to Add
1. **Optimizers** (3 weeks)
   - Adam, SGD+momentum, RMSprop
   - Optimizer state management
   
2. **Mini-batch training** (2 weeks)
   - Batch gradient computation
   - Data loader with shuffling
   
3. **Loss functions library** (1 week)
   - Cross-entropy, MSE, MAE
   - Custom loss support
   
4. **Layer primitives** (3 weeks)
   - Linear layers
   - Activation layers
   - Simple sequential model API

### What v1.2-ml-toolkit Enables
```scheme
; This will WORK in v1.2:
(define model
  (sequential
    (linear 784 256)
    (relu)
    (linear 256 10)))

(define optimizer (adam (model-parameters model) 0.001))

(train model mnist-train-loader 
       (lambda (pred target) (cross-entropy pred target))
       optimizer
       epochs: 10)
```

### v1.2-ml-toolkit Deliverables
✅ Train MNIST to 95%+ accuracy  
✅ Multi-layer perceptrons  
✅ Adam optimizer  
✅ Mini-batch training  
✅ Data loading utilities  
✅ Model save/load

---

## v1.3-advanced-ml (Post v1.2, ~4-6 months)

### Purpose  
**Advanced architectures & features**

### Features to Add
1. **Convolutional layers** (4 weeks)
2. **Recurrent networks** (4 weeks)  
3. **Attention mechanisms** (3 weeks)
4. **Advanced optimizers** (2 weeks)
5. **Data augmentation** (2 weeks)
6. **Model zoo** (ongoing)

### What v1.3-advanced-ml Enables
```scheme
; This will WORK in v1.3:
(define cnn
  (sequential
    (conv2d 1 32 3 1 1)
    (relu)
    (max-pool2d 2 2)
    (linear 1568 10)))

(train cnn cifar10-loader cross-entropy adam epochs: 50)
```

### v1.3-advanced-ml Deliverables
✅ Train CIFAR-10 to 85%+  
✅ CNNs working  
✅ RNNs/LSTMs working  
✅ Attention mechanisms  
✅ 20+ architecture examples

---

## v2.0-production (Post v1.3, ~6-12 months)

### Purpose
**Production deployment & scale**

### Features to Add
1. **GPU acceleration** (2-3 months)
   - CUDA kernel generation
   - Device management
   
2. **JIT compilation** (2 months)
   - Operation fusion
   - Dynamic optimization
   
3. **Model export** (1 month)
   - ONNX export
   - TorchScript conversion
   
4. **Serving infrastructure** (1 month)
   - Model serving API
   - Batch inference

### v2.0-production Deliverables
✅ GPU training  
✅ Production deployment  
✅ Model export to ONNX  
✅ Inference optimization  
✅ Distributed training (basic)

---

## Realistic Timeline

```
NOW (Nov 2025)
  ↓
v1.0-foundation (Jan 2026)     ← 4-6 weeks
  • Examples + docs + packaging
  • Research-ready platform
  • Autodiff working perfectly
  ↓
v1.1-neural (Mar 2026)         ← 2-3 months
  • Fix 7 bugs
  • Basic NN training works
  • Simple perceptrons trainable
  ↓
v1.2-ml-toolkit (Jun 2026)     ← 3-4 months
  • Production training
  • MNIST/CIFAR working
  • Optimizers + layers
  ↓
v1.3-advanced-ml (Nov 2026)    ← 4-6 months
  • CNNs, RNNs, Attention
  • Advanced architectures
  ↓
v2.0-production (Mid 2027)     ← 6-12 months
  • GPU support
  • Production deployment
  • Export/serving
```

**Total to production ML framework: ~18-24 months from now**

---

## What's Realistic for Each Release

### v1.0-foundation (IMMINENT)
**Positioning**: "Scheme with advanced autodiff for researchers"

**Use Cases:**
- Symbolic mathematics
- Scientific computing
- Algorithm prototyping
- Autodiff experiments
- Teaching functional programming

**NOT For:**
- Training neural networks (yet)
- Production ML workloads
- Large-scale data processing

**Announcement**:
> "Eshkol v1.0-foundation: A new Scheme implementation with production-ready automatic differentiation. Perfect for researchers exploring differentiable programming."

---

### v1.1-neural (Q1 2026)
**Positioning**: "Train simple neural networks functionally"

**Use Cases:**
- Learning neural network concepts
- Simple perceptron training
- Gradient descent experiments
- Educational demonstrations

**NOT For:**
- Training large models
- Production ML pipelines
- Complex architectures

**Announcement**:
> "Eshkol v1.1: Now with iterative training! Build and train simple neural networks using functional programming + autodiff."

---

### v1.2-ml-toolkit (Q2 2026)
**Positioning**: "Practical ML training in Scheme"

**Use Cases:**
- MNIST/CIFAR experiments
- Custom architecture research
- Algorithm development
- Teaching ML concepts

**NOT For:**
- ImageNet-scale training
- GPU-accelerated workloads
- Transformer models

**Announcement**:
> "Eshkol v1.2: Full ML toolkit! Train real neural networks on standard datasets. Includes optimizers, layers, and data loading."

---

### v1.3-advanced-ml (Q4 2026)  
**Positioning**: "Research-grade ML framework"

**Use Cases:**
- Novel architecture research
- Academic papers
- Advanced ML courses
- Custom model development

**NOT For:**
- Competing with PyTorch (ecosystem)
- Production deployment at scale
- Real-time inference

---

### v2.0-production (2027)
**Positioning**: "Production ML deployment"

**Use Cases:**
- Production model serving
- High-performance inference
- GPU-accelerated training
- Export to standard formats

---

## Key Insights

### What Makes Eshkol Unique (Don't Lose Focus!)

1. **Functional + Symbolic + Autodiff**
   - No other language combines all three
   - Research value is HUGE
   - Don't just copy PyTorch

2. **Type-theoretic foundation**
   - HoTT influence
   - Formal correctness
   - Academic rigor

3. **Lightweight & compiled**
   - No Python runtime
   - Fast binaries
   - Embedded-friendly

### What NOT to Try

**Don't compete on:**
- Ecosystem size (PyTorch has 10 years head start)
- Model zoo (thousands of pretrained models)
- Production tooling (mature alternatives exist)
- GPU performance (CUDA is optimized over decades)

**DO compete on:**
- Functional programming + ML
- Symbolic reasoning + numeric computation  
- Type safety + autodiff
- Research experimentation
- Novel architectures

---

## Immediate Next Steps (Aligned with v1.0)

### This Week
1. Continue v1.0-foundation work (examples + docs)
2. Don't worry about neural network bugs yet
3. Focus on making v1.0 polished and releasable

### After v1.0-foundation Release
1. Take 1-2 weeks to celebrate and gather feedback
2. Plan v1.1-neural based on user requests
3. Fix the 7 bugs systematically
4. Release v1.1 when training works

### Iterative Value Delivery

**v1.0** → Researchers can use Eshkol for symbolic math + autodiff  
**v1.1** → Can experiment with simple neural networks  
**v1.2** → Can train real models on standard datasets  
**v1.3** → Can publish research papers using Eshkol  
**v2.0** → Can deploy models to production

Each release adds concrete value. Each release is usable for its target audience.

---

## Summary for Neural Network Work

### For v1.0-foundation (NOW)
**Neural networks**: Not a goal. Document limitations.  
**Focus**: Polish what works, release it.  
**Timeline**: 4-6 weeks.

### For v1.1-neural (NEXT, post v1.0)
**Neural networks**: Basic training works.  
**Focus**: Fix 7 bugs identified in technical spec.  
**Timeline**: 2-3 months after v1.0.

### For v1.2+ (FUTURE)
**Neural networks**: Production-ready.  
**Focus**: Add ML toolkit infrastructure.  
**Timeline**: 6-12 months after v1.0.

---

## What This Means

**You're RIGHT** - It's iterative!

- **v1.0** is ready in weeks (just infrastructure)
- **v1.1** adds neural network training (2-3 months later)
- **v1.2+** adds production features (ongoing)
- **Total to production ML**: ~18 months (realistic for a new language!)

The 7 bugs and advanced features I identified are the **roadmap for future versions**, not blockers for v1.0-foundation.

**v1.0-foundation ships with what works perfectly NOW (autodiff, lists, higher-order functions), and we document what's coming next.**