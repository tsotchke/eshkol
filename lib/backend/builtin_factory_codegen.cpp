#include <eshkol/backend/llvm_codegen.h>
#include <eshkol/platform_runtime.h>
#include <eshkol/runtime_exports.h>
#include <eshkol/core/runtime.h>
#include <eshkol/backend/tensorcore_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

void EshkolLLVMCodeGen::createBuiltinFunctions() {
        // printf function declaration
        std::vector<Type*> printf_args;
        printf_args.push_back(PointerType::getUnqual(*context)); // const char* format

        FunctionType* printf_type = FunctionType::get(
            int32_type, // return int
            printf_args,
            true // varargs
        );

        Function* printf_func = Function::Create(
            printf_type,
            Function::ExternalLinkage,
            "printf",
            module.get()
        );

        function_table["printf"] = printf_func;

        // sin function declaration (from libm)
        std::vector<Type*> sin_args;
        sin_args.push_back(double_type); // double x

        FunctionType* sin_type = FunctionType::get(
            double_type, // return double
            sin_args,
            false // not varargs
        );

        Function* sin_func = Function::Create(
            sin_type,
            Function::ExternalLinkage,
            "sin",
            module.get()
        );

        function_table["sin"] = sin_func;

        // cos function declaration (from libm)
        std::vector<Type*> cos_args;
        cos_args.push_back(double_type); // double x

        FunctionType* cos_type = FunctionType::get(
            double_type, // return double
            cos_args,
            false // not varargs
        );

        Function* cos_func = Function::Create(
            cos_type,
            Function::ExternalLinkage,
            "cos",
            module.get()
        );

        function_table["cos"] = cos_func;

        // sqrt function declaration (from libm)
        std::vector<Type*> sqrt_args;
        sqrt_args.push_back(double_type); // double x

        FunctionType* sqrt_type = FunctionType::get(
            double_type, // return double
            sqrt_args,
            false // not varargs
        );

        Function* sqrt_func = Function::Create(
            sqrt_type,
            Function::ExternalLinkage,
            "sqrt",
            module.get()
        );

        function_table["sqrt"] = sqrt_func;

        // pow function declaration (from libm)
        std::vector<Type*> pow_args;
        pow_args.push_back(double_type); // double base
        pow_args.push_back(double_type); // double exponent

        FunctionType* pow_type = FunctionType::get(
            double_type, // return double
            pow_args,
            false // not varargs
        );

        Function* pow_func = Function::Create(
            pow_type,
            Function::ExternalLinkage,
            "pow",
            module.get()
        );

        function_table["pow"] = pow_func;

        // exit function declaration (from stdlib.h)
        std::vector<Type*> exit_args;
        exit_args.push_back(int32_type); // int status

        FunctionType* exit_type = FunctionType::get(
            void_type, // returns void (actually noreturn)
            exit_args,
            false // not varargs
        );

        Function* exit_func = Function::Create(
            exit_type,
            Function::ExternalLinkage,
            "exit",
            module.get()
        );
        exit_func->addFnAttr(Attribute::NoReturn);

        function_table["exit"] = exit_func;

        // ============================================================================
        // FILE I/O FUNCTIONS (from stdio.h)
        // ============================================================================

        // fopen: FILE* eshkol_fopen(const char* filename, const char* mode)
        std::vector<Type*> fopen_args;
        fopen_args.push_back(PointerType::get(*context, 0));  // filename
        fopen_args.push_back(PointerType::get(*context, 0));  // mode
        FunctionType* fopen_type = FunctionType::get(
            PointerType::get(*context, 0), fopen_args, false);
        Function* fopen_func = Function::Create(
            fopen_type, Function::ExternalLinkage, eshkol::runtime::fopen_symbol, module.get());
        function_table["fopen"] = fopen_func;

        // fclose: int fclose(FILE* stream)
        std::vector<Type*> fclose_args;
        fclose_args.push_back(PointerType::get(*context, 0));  // stream
        FunctionType* fclose_type = FunctionType::get(
            int32_type, fclose_args, false);
        Function* fclose_func = Function::Create(
            fclose_type, Function::ExternalLinkage, "fclose", module.get());
        function_table["fclose"] = fclose_func;

        // fgets: char* fgets(char* str, int n, FILE* stream)
        std::vector<Type*> fgets_args;
        fgets_args.push_back(PointerType::get(*context, 0));  // str
        fgets_args.push_back(int32_type);      // n
        fgets_args.push_back(PointerType::get(*context, 0));  // stream
        FunctionType* fgets_type = FunctionType::get(
            PointerType::get(*context, 0), fgets_args, false);
        Function* fgets_func = Function::Create(
            fgets_type, Function::ExternalLinkage, "fgets", module.get());
        function_table["fgets"] = fgets_func;

        // feof: int feof(FILE* stream)
        std::vector<Type*> feof_args;
        feof_args.push_back(PointerType::get(*context, 0));  // stream
        FunctionType* feof_type = FunctionType::get(
            int32_type, feof_args, false);
        Function* feof_func = Function::Create(
            feof_type, Function::ExternalLinkage, "feof", module.get());
        function_table["feof"] = feof_func;

        // fputs: int fputs(const char* str, FILE* stream)
        std::vector<Type*> fputs_args;
        fputs_args.push_back(PointerType::get(*context, 0));  // str
        fputs_args.push_back(PointerType::get(*context, 0));  // stream
        FunctionType* fputs_type = FunctionType::get(
            int32_type, fputs_args, false);
        Function* fputs_func = Function::Create(
            fputs_type, Function::ExternalLinkage, "fputs", module.get());
        function_table["fputs"] = fputs_func;

        // fputc: int fputc(int c, FILE* stream)
        std::vector<Type*> fputc_args;
        fputc_args.push_back(int32_type);      // c
        fputc_args.push_back(PointerType::get(*context, 0));  // stream
        FunctionType* fputc_type = FunctionType::get(
            int32_type, fputc_args, false);
        Function* fputc_func = Function::Create(
            fputc_type, Function::ExternalLinkage, "fputc", module.get());
        function_table["fputc"] = fputc_func;

        // strlen: size_t strlen(const char* str)
        std::vector<Type*> strlen_args;
        strlen_args.push_back(PointerType::get(*context, 0));  // str
        FunctionType* strlen_type = FunctionType::get(
            int64_type, strlen_args, false);
        Function* strlen_func = Function::Create(
            strlen_type, Function::ExternalLinkage, "strlen", module.get());
        function_table["strlen"] = strlen_func;

        // ============================================================================
        // RANDOM NUMBER FUNCTIONS (from stdlib.h)
        // ============================================================================

        // drand48: double drand48(void) - returns random double in [0.0, 1.0)
        FunctionType* drand48_type = FunctionType::get(
            double_type, {}, false);
        Function* drand48_func = Function::Create(
            drand48_type, Function::ExternalLinkage, "drand48", module.get());
        function_table["drand48"] = drand48_func;

        // srand48: void srand48(long seed) - seeds the random number generator
        std::vector<Type*> srand48_args;
        srand48_args.push_back(int64_type);  // seed
        FunctionType* srand48_type = FunctionType::get(
            void_type, srand48_args, false);
        Function* srand48_func = Function::Create(
            srand48_type, Function::ExternalLinkage, "srand48", module.get());
        function_table["srand48"] = srand48_func;

        // ============================================================================
        // QUANTUM RANDOM NUMBER GENERATOR (from lib/quantum/quantum_rng_wrapper.h)
        // ============================================================================

        // eshkol_qrng_double: double eshkol_qrng_double(void) - quantum random in [0,1)
        FunctionType* qrng_double_type = FunctionType::get(double_type, {}, false);
        Function* qrng_double_func = Function::Create(
            qrng_double_type, Function::ExternalLinkage, "eshkol_qrng_double", module.get());
        function_table["eshkol_qrng_double"] = qrng_double_func;

        // eshkol_qrng_uint64: uint64_t eshkol_qrng_uint64(void) - quantum random uint64
        FunctionType* qrng_uint64_type = FunctionType::get(int64_type, {}, false);
        Function* qrng_uint64_func = Function::Create(
            qrng_uint64_type, Function::ExternalLinkage, "eshkol_qrng_uint64", module.get());
        function_table["eshkol_qrng_uint64"] = qrng_uint64_func;

        // eshkol_qrng_range: int64_t eshkol_qrng_range(int64_t min, int64_t max) - quantum random in range
        std::vector<Type*> qrng_range_args = {int64_type, int64_type};
        FunctionType* qrng_range_type = FunctionType::get(int64_type, qrng_range_args, false);
        Function* qrng_range_func = Function::Create(
            qrng_range_type, Function::ExternalLinkage, "eshkol_qrng_range", module.get());
        function_table["eshkol_qrng_range"] = qrng_range_func;

        // time: time_t time(time_t* timer) - for seeding random
        std::vector<Type*> time_args;
        time_args.push_back(PointerType::get(*context, 0));  // timer (can be NULL)
        FunctionType* time_type = FunctionType::get(
            int64_type, time_args, false);
        Function* time_func = Function::Create(
            time_type, Function::ExternalLinkage, "time", module.get());
        function_table["time"] = time_func;

        // gettimeofday: int gettimeofday(struct timeval* tv, struct timezone* tz)
        // struct timeval { time_t tv_sec; suseconds_t tv_usec; }
        // We'll use i64 for both fields for simplicity
        std::vector<Type*> gettimeofday_args;
        gettimeofday_args.push_back(PointerType::get(*context, 0));  // timeval*
        gettimeofday_args.push_back(PointerType::get(*context, 0));  // timezone* (can be NULL)
        FunctionType* gettimeofday_type = FunctionType::get(
            int32_type, gettimeofday_args, false);
        Function* gettimeofday_func = Function::Create(
            gettimeofday_type, Function::ExternalLinkage, "gettimeofday", module.get());
        function_table["gettimeofday"] = gettimeofday_func;

        // ============================================================================
        // SYSTEM & ENVIRONMENT FUNCTIONS (from stdlib.h, unistd.h)
        // ============================================================================

        // getenv: char* eshkol_getenv(const char* name)
        std::vector<Type*> getenv_args;
        getenv_args.push_back(PointerType::get(*context, 0));  // name
        FunctionType* getenv_type = FunctionType::get(
            PointerType::get(*context, 0), getenv_args, false);
        Function* getenv_func = Function::Create(
            getenv_type, Function::ExternalLinkage, eshkol::runtime::getenv_symbol, module.get());
        function_table["getenv"] = getenv_func;

        // setenv: int eshkol_setenv(const char* name, const char* value, int overwrite)
        std::vector<Type*> setenv_args;
        setenv_args.push_back(PointerType::get(*context, 0));  // name
        setenv_args.push_back(PointerType::get(*context, 0));  // value
        setenv_args.push_back(int32_type);                     // overwrite
        FunctionType* setenv_type = FunctionType::get(
            int32_type, setenv_args, false);
        Function* setenv_func = Function::Create(
            setenv_type, Function::ExternalLinkage, eshkol::runtime::setenv_symbol, module.get());
        function_table["setenv"] = setenv_func;

        // unsetenv: int eshkol_unsetenv(const char* name)
        std::vector<Type*> unsetenv_args;
        unsetenv_args.push_back(PointerType::get(*context, 0));  // name
        FunctionType* unsetenv_type = FunctionType::get(
            int32_type, unsetenv_args, false);
        Function* unsetenv_func = Function::Create(
            unsetenv_type, Function::ExternalLinkage, eshkol::runtime::unsetenv_symbol, module.get());
        function_table["unsetenv"] = unsetenv_func;

        // system: int system(const char* command)
        std::vector<Type*> system_args;
        system_args.push_back(PointerType::get(*context, 0));  // command
        FunctionType* system_type = FunctionType::get(
            int32_type, system_args, false);
        Function* system_func = Function::Create(
            system_type, Function::ExternalLinkage, "system", module.get());
        function_table["system"] = system_func;

        // usleep: int usleep(useconds_t usec) - sleep for microseconds
        std::vector<Type*> usleep_args;
        usleep_args.push_back(int32_type);  // usec (microseconds)
        FunctionType* usleep_type = FunctionType::get(
            int32_type, usleep_args, false);
        Function* usleep_func = Function::Create(
            usleep_type, Function::ExternalLinkage, "usleep", module.get());
        function_table["usleep"] = usleep_func;

        // access: int eshkol_access(const char* path, int mode) - check file access
        std::vector<Type*> access_args;
        access_args.push_back(PointerType::get(*context, 0));  // path
        access_args.push_back(int32_type);                     // mode
        FunctionType* access_type = FunctionType::get(
            int32_type, access_args, false);
        Function* access_func = Function::Create(
            access_type, Function::ExternalLinkage, eshkol::runtime::access_symbol, module.get());
        function_table["access"] = access_func;

        // remove: int eshkol_remove(const char* path) - delete file
        std::vector<Type*> remove_args;
        remove_args.push_back(PointerType::get(*context, 0));  // path
        FunctionType* remove_type = FunctionType::get(
            int32_type, remove_args, false);
        Function* remove_func = Function::Create(
            remove_type, Function::ExternalLinkage, eshkol::runtime::remove_symbol, module.get());
        function_table["remove"] = remove_func;

        // rename: int eshkol_rename(const char* old, const char* new)
        std::vector<Type*> rename_args;
        rename_args.push_back(PointerType::get(*context, 0));  // old path
        rename_args.push_back(PointerType::get(*context, 0));  // new path
        FunctionType* rename_type = FunctionType::get(
            int32_type, rename_args, false);
        Function* rename_func = Function::Create(
            rename_type, Function::ExternalLinkage, eshkol::runtime::rename_symbol, module.get());
        function_table["rename"] = rename_func;

        // mkdir: int eshkol_mkdir(const char* path, mode_t mode)
        std::vector<Type*> mkdir_args;
        mkdir_args.push_back(PointerType::get(*context, 0));  // path
        mkdir_args.push_back(int32_type);                     // mode
        FunctionType* mkdir_type = FunctionType::get(
            int32_type, mkdir_args, false);
        Function* mkdir_func = Function::Create(
            mkdir_type, Function::ExternalLinkage, eshkol::runtime::mkdir_symbol, module.get());
        function_table["mkdir"] = mkdir_func;

        // rmdir: int eshkol_rmdir(const char* path)
        std::vector<Type*> rmdir_args;
        rmdir_args.push_back(PointerType::get(*context, 0));  // path
        FunctionType* rmdir_type = FunctionType::get(
            int32_type, rmdir_args, false);
        Function* rmdir_func = Function::Create(
            rmdir_type, Function::ExternalLinkage, eshkol::runtime::rmdir_symbol, module.get());
        function_table["rmdir"] = rmdir_func;

        // getcwd: char* getcwd(char* buf, size_t size)
        std::vector<Type*> getcwd_args;
        getcwd_args.push_back(PointerType::get(*context, 0));  // buf
        getcwd_args.push_back(int64_type);                     // size
        FunctionType* getcwd_type = FunctionType::get(
            PointerType::get(*context, 0), getcwd_args, false);
        Function* getcwd_func = Function::Create(
            getcwd_type, Function::ExternalLinkage, "getcwd", module.get());
        function_table["getcwd"] = getcwd_func;

        // chdir: int eshkol_chdir(const char* path)
        std::vector<Type*> chdir_args;
        chdir_args.push_back(PointerType::get(*context, 0));  // path
        FunctionType* chdir_type = FunctionType::get(
            int32_type, chdir_args, false);
        Function* chdir_func = Function::Create(
            chdir_type, Function::ExternalLinkage, eshkol::runtime::chdir_symbol, module.get());
        function_table["chdir"] = chdir_func;

        // stat: int eshkol_stat(const char* path, struct stat* buf)
        std::vector<Type*> stat_args;
        stat_args.push_back(PointerType::get(*context, 0));  // path
        stat_args.push_back(PointerType::get(*context, 0));  // stat buf
        FunctionType* stat_type = FunctionType::get(
            int32_type, stat_args, false);
        Function* stat_func = Function::Create(
            stat_type, Function::ExternalLinkage, eshkol::runtime::stat_symbol, module.get());
        function_table["stat"] = stat_func;

        // opendir: DIR* eshkol_opendir(const char* name)
        std::vector<Type*> opendir_args;
        opendir_args.push_back(PointerType::get(*context, 0));  // name
        FunctionType* opendir_type = FunctionType::get(
            PointerType::get(*context, 0), opendir_args, false);
        Function* opendir_func = Function::Create(
            opendir_type, Function::ExternalLinkage, eshkol::runtime::opendir_symbol, module.get());
        function_table["opendir"] = opendir_func;

        // readdir: struct dirent* readdir(DIR* dirp)
        std::vector<Type*> readdir_args;
        readdir_args.push_back(PointerType::get(*context, 0));  // dirp
        FunctionType* readdir_type = FunctionType::get(
            PointerType::get(*context, 0), readdir_args, false);
        Function* readdir_func = Function::Create(
            readdir_type, Function::ExternalLinkage, "readdir", module.get());
        function_table["readdir"] = readdir_func;

        // closedir: int closedir(DIR* dirp)
        std::vector<Type*> closedir_args;
        closedir_args.push_back(PointerType::get(*context, 0));  // dirp
        FunctionType* closedir_type = FunctionType::get(
            int32_type, closedir_args, false);
        Function* closedir_func = Function::Create(
            closedir_type, Function::ExternalLinkage, "closedir", module.get());
        function_table["closedir"] = closedir_func;

        // fseek: int fseek(FILE* stream, long offset, int whence)
        std::vector<Type*> fseek_args;
        fseek_args.push_back(PointerType::get(*context, 0));  // stream
        fseek_args.push_back(int64_type);                     // offset
        fseek_args.push_back(int32_type);                     // whence
        FunctionType* fseek_type = FunctionType::get(
            int32_type, fseek_args, false);
        Function* fseek_func = Function::Create(
            fseek_type, Function::ExternalLinkage, "fseek", module.get());
        function_table["fseek"] = fseek_func;

        // ftell: long ftell(FILE* stream)
        std::vector<Type*> ftell_args;
        ftell_args.push_back(PointerType::get(*context, 0));  // stream
        FunctionType* ftell_type = FunctionType::get(
            int64_type, ftell_args, false);
        Function* ftell_func = Function::Create(
            ftell_type, Function::ExternalLinkage, "ftell", module.get());
        function_table["ftell"] = ftell_func;

        // fread: size_t fread(void* ptr, size_t size, size_t nmemb, FILE* stream)
        std::vector<Type*> fread_args;
        fread_args.push_back(PointerType::get(*context, 0));  // ptr
        fread_args.push_back(int64_type);                     // size
        fread_args.push_back(int64_type);                     // nmemb
        fread_args.push_back(PointerType::get(*context, 0));  // stream
        FunctionType* fread_type = FunctionType::get(
            int64_type, fread_args, false);
        Function* fread_func = Function::Create(
            fread_type, Function::ExternalLinkage, "fread", module.get());
        function_table["fread"] = fread_func;

        // fwrite: size_t fwrite(const void* ptr, size_t size, size_t nmemb, FILE* stream)
        std::vector<Type*> fwrite_args;
        fwrite_args.push_back(PointerType::get(*context, 0));  // ptr
        fwrite_args.push_back(int64_type);                     // size
        fwrite_args.push_back(int64_type);                     // nmemb
        fwrite_args.push_back(PointerType::get(*context, 0));  // stream
        FunctionType* fwrite_type = FunctionType::get(
            int64_type, fwrite_args, false);
        Function* fwrite_func = Function::Create(
            fwrite_type, Function::ExternalLinkage, "fwrite", module.get());
        function_table["fwrite"] = fwrite_func;

        // ============================================================================
        // COMPREHENSIVE C STANDARD MATH FUNCTIONS
        // ============================================================================

        // Helper to declare a single-arg math function (double -> double)
        auto declareUnaryMathFunc = [this](const char* name) {
            std::vector<Type*> args = {double_type};
            FunctionType* type = FunctionType::get(double_type, args, false);
            Function* func = Function::Create(type, Function::ExternalLinkage, name, module.get());
            function_table[name] = func;
        };

        // Helper to declare a two-arg math function (double, double -> double)
        auto declareBinaryMathFunc = [this](const char* name) {
            std::vector<Type*> args = {double_type, double_type};
            FunctionType* type = FunctionType::get(double_type, args, false);
            Function* func = Function::Create(type, Function::ExternalLinkage, name, module.get());
            function_table[name] = func;
        };

        // Trigonometric functions
        declareUnaryMathFunc("tan");
        declareUnaryMathFunc("asin");
        declareUnaryMathFunc("acos");
        declareUnaryMathFunc("atan");
        declareBinaryMathFunc("atan2");

        // Hyperbolic functions
        declareUnaryMathFunc("sinh");
        declareUnaryMathFunc("cosh");
        declareUnaryMathFunc("tanh");
        declareUnaryMathFunc("asinh");
        declareUnaryMathFunc("acosh");
        declareUnaryMathFunc("atanh");

        // Exponential
        declareUnaryMathFunc("exp2");

        // Logarithmic
        declareUnaryMathFunc("log");  // natural log
        declareUnaryMathFunc("log10");
        declareUnaryMathFunc("log2");

        // Numeric/rounding functions
        declareUnaryMathFunc("fabs");   // absolute value
        declareUnaryMathFunc("floor");
        declareUnaryMathFunc("ceil");
        declareUnaryMathFunc("round");
        declareUnaryMathFunc("trunc");
        declareBinaryMathFunc("fmod");  // modulo for floats
        declareBinaryMathFunc("remainder");  // IEEE remainder
        declareBinaryMathFunc("fmin");
        declareBinaryMathFunc("fmax");
        declareUnaryMathFunc("cbrt");   // cube root

        // Note: Builtin runtime function declarations (eshkol_deep_equal, eshkol_display_value,
        // eshkol_lambda_registry_*) are now created via BuiltinDeclarations after CodegenContext
        // is initialized. See the builtins_ initialization below.

        // Get struct types from TypeSystem (types are created once in constructor)
        dual_number_type = types->getDualNumberType();
        ad_node_type = types->getAdNodeType();
        tensor_type = types->getTensorType();

        // Initialize tape state
        current_tape_ptr = nullptr;
        next_node_id = 0;

        eshkol_debug("Using TypeSystem-managed struct types (dual_number, ad_node, tensor)");

        // Initialize memory codegen (creates arena function declarations)
        mem = std::make_unique<eshkol::MemoryCodegen>(*module, *types);

        // Initialize CodegenContext - shared state for extracted modules
        // This must happen after types, funcs, and mem are all initialized
        ctx_ = std::make_unique<eshkol::CodegenContext>(
            *context, *module, *builder, *types, *funcs, *mem
        );
        ctx_->setLibraryMode(library_mode);
        ctx_->setModulePrefix(module_prefix);
        ctx_->setGlobalArena(global_arena);
        ctx_->setAdModeActive(ad_mode_active);
        ctx_->setCurrentAdTape(current_ad_tape);
        ctx_->setAdTapeStack(ad_tape_stack);
        ctx_->setAdTapeDepth(ad_tape_depth);
        ctx_->setAdPertLevel(ad_pert_level);
        ctx_->setAdTowerActive(ad_tower_active);
        ctx_->setAdTowerOrder(ad_tower_order);
        ctx_->setOuterAdNodeStorage(outer_ad_node_storage);
        ctx_->setOuterAdNodeToInner(outer_ad_node_to_inner);
        ctx_->setOuterGradAccumulator(outer_grad_accumulator);
        ctx_->setInnerVarNodePtr(inner_var_node_ptr);
        ctx_->setGradientXDegree(gradient_x_degree);
        ctx_->setOuterAdNodeStack(outer_ad_node_stack);
        ctx_->setOuterAdNodeDepth(outer_ad_node_depth);
        if (eshkol::llvm_codegen_detail::replModeEnabled()) {
            ctx_->setReplMode(true);
        }
        eshkol_debug("Created CodegenContext for module '%s'", module_prefix.c_str());

        // Initialize TaggedValueCodegen - pack/unpack operations for tagged values
        tagged_ = std::make_unique<eshkol::TaggedValueCodegen>(*ctx_);
        eshkol_debug("Created TaggedValueCodegen");

        // Initialize BuiltinDeclarations - runtime function declarations
        // Creates: eshkol_deep_equal, eshkol_display_value, eshkol_lambda_registry_*
        builtins_ = std::make_unique<eshkol::BuiltinDeclarations>(*ctx_);
        // Update member pointers for backward compatibility
        eshkol_deep_equal_func = builtins_->getDeepEqual();
        eshkol_display_value_func = builtins_->getDisplayValue();
        eshkol_lambda_registry_init_func = builtins_->getLambdaRegistryInit();
        eshkol_lambda_registry_add_func = builtins_->getLambdaRegistryAdd();
        eshkol_lambda_registry_lookup_func = builtins_->getLambdaRegistryLookup();
        eshkol_debug("Created BuiltinDeclarations");

        // Every frontend shares the canonical Eshkol-owned adapter ABI. Builds
        // without an installed TensorCore package resolve these declarations to
        // explicit-unavailable runtime stubs; no ambient environment toggle can
        // silently change compiler lowering.
        if (eshkol_register_tensorcore_builtins(ctx_.get()) < 0) {
            eshkol_error("tensorcore: canonical adapter registration failed");
        }

        // Initialize TensorCodegen - tensor operations (needed by ArithmeticCodegen)
        tensor_ = std::make_unique<eshkol::TensorCodegen>(*ctx_, *tagged_, *mem);
        // Set up callbacks for AST evaluation (uses same pattern as other modules)
        tensor_->setCodegenCallbacks(
            ControlFlowCallbacks::codegenASTWrapper,
            ControlFlowCallbacks::codegenTypedASTWrapper,
            ControlFlowCallbacks::typedToTaggedWrapper,
            this
        );
        eshkol_debug("Created TensorCodegen with callbacks");

        // Initialize AutodiffCodegen - automatic differentiation operations (needed by ArithmeticCodegen)
        autodiff_ = std::make_unique<eshkol::AutodiffCodegen>(*ctx_, *tagged_, *mem);
        // Set up function table reference for math operations (sin, cos, exp, etc.)
        autodiff_->setFunctionTable(&function_table);
        // Set up symbol tables for variable/capture lookup
        autodiff_->setSymbolTables(&symbol_table, &global_symbol_table);
        // Set up REPL mode flag
        autodiff_->setReplMode(&eshkol::llvm_codegen_detail::replModeEnabled());
        // Set up REPL state for cross-evaluation function resolution
        autodiff_->setReplState(&eshkol::llvm_codegen_detail::replMutex(), &eshkol::llvm_codegen_detail::replLambdaCaptures(), &eshkol::llvm_codegen_detail::replSymbolAddresses());
        // Set up AST codegen callback
        autodiff_->setCodegenASTCallback(ControlFlowCallbacks::codegenASTTypedWrapper, this);
        // Set up lambda resolution callback
        autodiff_->setResolveLambdaCallback(ControlFlowCallbacks::resolveLambdaWrapper);
        // Calculus extraction: wire closure call, arity table, captures, closure alloc
        autodiff_->setClosureCallCallback(ControlFlowCallbacks::closureCallWithInfoWrapper);
        autodiff_->setGradientSpreadCallCallback(ControlFlowCallbacks::gradientSpreadCallWrapper);
        autodiff_->setFunctionArityTable(&function_arity_table);
        autodiff_->setFunctionBodyAstTable(&function_body_ast);
        autodiff_->setFunctionDefAstTable(&function_def_ast);
        autodiff_->setNestedFunctionCaptures(&nested_function_captures);
        autodiff_->setGetClosureAllocFunc(ControlFlowCallbacks::getClosureAllocWrapper);
        tensor_->setAutodiffCodegen(autodiff_.get());
        eshkol_debug("Created AutodiffCodegen with function table and callbacks");

        // Initialize ComplexCodegen - complex number arithmetic
        complex_ = std::make_unique<eshkol::ComplexCodegen>(*ctx_, *tagged_, *mem);
        eshkol_debug("Created ComplexCodegen");

        // Initialize ArithmeticCodegen - polymorphic arithmetic operations
        // Now fully functional with tensor, autodiff, and complex support
        arith_ = std::make_unique<eshkol::ArithmeticCodegen>(*ctx_, *tagged_, *tensor_, *autodiff_, *complex_);
        eshkol_debug("Created ArithmeticCodegen");

        // Initialize CallApplyCodegen - function call and apply operations
        call_apply_ = std::make_unique<eshkol::CallApplyCodegen>(*ctx_, *tagged_, *arith_);
        call_apply_->setSymbolTables(&symbol_table, &global_symbol_table);
        call_apply_->setVariadicFunctionInfo(&variadic_function_info);
        call_apply_->setFunctionTable(&function_table);
        call_apply_->setCodegenASTCallback(ControlFlowCallbacks::codegenASTTypedWrapper, this);
        call_apply_->setExtractConsCarCallback(ControlFlowCallbacks::extractConsCarWrapper);
        call_apply_->setGetConsAccessorCallback(ControlFlowCallbacks::getConsAccessorWrapper);
        call_apply_->setCreateConsCallback(ControlFlowCallbacks::consCreateWrapper);
        call_apply_->setGetBuiltinArithmeticCallback(ControlFlowCallbacks::getBuiltinArithmeticWrapper);
        call_apply_->setGetBuiltinPredicateCallback(ControlFlowCallbacks::getBuiltinPredicateWrapper);
        call_apply_->setApplyBuiltinCallback(ControlFlowCallbacks::applyBuiltinWrapper);
        call_apply_->setApplyForwardRefCallback(ControlFlowCallbacks::applyForwardRefWrapper);
        eshkol_debug("Created CallApplyCodegen with callbacks");

        // Initialize MapCodegen - higher-order list mapping operations
        map_ = std::make_unique<eshkol::MapCodegen>(*ctx_, *tagged_);
        map_->setSymbolTables(&symbol_table, &global_symbol_table);
        map_->setFunctionTable(&function_table);
        map_->setNestedFunctionCaptures(&nested_function_captures);
        map_->setLastGeneratedLambdaName(&eshkol::llvm_codegen_detail::lastGeneratedLambdaName());
        map_->setCurrentFunction(&current_function);
        map_->setCodegenASTCallback(ControlFlowCallbacks::codegenASTTypedWrapper, this);
        map_->setCodegenLambdaCallback(ControlFlowCallbacks::codegenLambdaWrapper);
        map_->setClosureCallCallback(ControlFlowCallbacks::closureCallWrapper);
        map_->setExtractCarCallback(ControlFlowCallbacks::extractConsCarWrapper);
        map_->setCreateConsCallback(ControlFlowCallbacks::consCreateWrapper);
        map_->setGetConsGetPtrCallback(ControlFlowCallbacks::getConsAccessorWrapper);
        map_->setGetConsSetPtrCallback(ControlFlowCallbacks::getConsSetPtrWrapper);
        map_->setResolveLambdaCallback(ControlFlowCallbacks::resolveLambdaWrapper);
        map_->setIndirectCallCallback(ControlFlowCallbacks::indirectCallWrapper);
        map_->setFunctionContextCallbacks(ControlFlowCallbacks::pushFunctionContextWrapper,
                                          ControlFlowCallbacks::popFunctionContextWrapper);
        eshkol_debug("Created MapCodegen with callbacks");

        // Initialize ControlFlowCodegen - control flow operations
        // Now fully functional with callback-based AST evaluation
        flow_ = std::make_unique<eshkol::ControlFlowCodegen>(*ctx_, *tagged_);
        // Set up callbacks for AST evaluation
        flow_->setCodegenCallbacks(
            ControlFlowCallbacks::codegenASTWrapper,
            ControlFlowCallbacks::codegenTypedASTWrapper,
            ControlFlowCallbacks::typedToTaggedWrapper,
            ControlFlowCallbacks::codegenFuncDefineWrapper,
            ControlFlowCallbacks::codegenVarDefineWrapper,
            ControlFlowCallbacks::eqvCompareWrapper,
            ControlFlowCallbacks::detectAndPackWrapper,
            this
        );
        flow_->setClosureCallCallback(ControlFlowCallbacks::closureCallWrapper);
        eshkol_debug("Created ControlFlowCodegen with callbacks");

        // Initialize StringIOCodegen - string and I/O operations
        strio_ = std::make_unique<eshkol::StringIOCodegen>(*ctx_, *tagged_);
        // Set up callbacks for AST evaluation (reuse ControlFlowCallbacks wrappers)
        strio_->setCodegenCallbacks(
            ControlFlowCallbacks::codegenASTWrapper,
            ControlFlowCallbacks::codegenTypedASTWrapper,
            ControlFlowCallbacks::typedToTaggedWrapper,
            ControlFlowCallbacks::consCreateWrapper,
            this
        );
        strio_->setDisplayValueFunc(eshkol_display_value_func);
        // R7RS §5.3.1: display must not short-circuit a redefined name to the
        // name-keyed `<name>_sexpr` side table (see setRedefinedTopLevelNames).
        strio_->setRedefinedTopLevelNames(&redefined_toplevel_names);
        eshkol_debug("Created StringIOCodegen with callbacks");

        // Initialize BindingCodegen - variable binding operations
        binding_ = std::make_unique<eshkol::BindingCodegen>(*ctx_, *tagged_);
        binding_->setCodegenCallbacks(
            ControlFlowCallbacks::codegenASTWrapper,
            ControlFlowCallbacks::codegenTypedASTWrapper,
            ControlFlowCallbacks::typedToTaggedWrapper,
            ControlFlowCallbacks::getTypedValueTypeWrapper,
            ControlFlowCallbacks::registerFuncBindingWrapper,
            this
        );
        binding_->setSymbolTables(&symbol_table, &global_symbol_table);
        binding_->setCurrentFunction(&current_function);
        binding_->setReplMode(&eshkol::llvm_codegen_detail::replModeEnabled());
        binding_->setLambdaTracking(&eshkol::llvm_codegen_detail::lastGeneratedLambdaName(), &function_table);
        binding_->setLetrecExcludedCaptureNames(&letrec_excluded_capture_names);
        // Set up TCO callbacks for tail call optimization
        binding_->setTCOCallbacks(ControlFlowCallbacks::isSelfTailRecursiveWrapper);
        eshkol_debug("Created BindingCodegen with callbacks and TCO support");

        // Wire binding codegen to autodiff (for TCO context save/restore in higher-order gradient)
        autodiff_->setBindingCodegen(binding_.get());

        // Initialize CollectionCodegen - list and vector operations
        coll_ = std::make_unique<eshkol::CollectionCodegen>(*ctx_, *tagged_, *mem);
        // Set up callbacks for AST evaluation (reuse ControlFlowCallbacks wrappers)
        coll_->setCodegenCallbacks(
            ControlFlowCallbacks::codegenASTWrapper,
            ControlFlowCallbacks::codegenTypedASTWrapper,
            ControlFlowCallbacks::typedToTaggedWrapper,
            this
        );
        eshkol_debug("Created CollectionCodegen with callbacks");

        // Initialize HomoiconicCodegen - quote and S-expression operations
        homoiconic_ = std::make_unique<eshkol::HomoiconicCodegen>(*ctx_, *tagged_, *coll_, *strio_);
        eshkol_debug("Created HomoiconicCodegen");

        // Initialize TailCallCodegen - tail call optimization support
        tailcall_ = std::make_unique<eshkol::TailCallCodegen>(*ctx_, *tagged_, *mem);
        tailcall_->generateTrampolineRuntime();
        eshkol_debug("Created TailCallCodegen with trampoline runtime");

        // Initialize SystemCodegen - system, environment, and file operations
        system_ = std::make_unique<eshkol::SystemCodegen>(*ctx_, *tagged_, *mem, function_table);
        system_->setCodegenCallbacks(
            ControlFlowCallbacks::codegenASTWrapper,
            ControlFlowCallbacks::codegenTypedASTWrapper,
            this
        );
        eshkol_debug("Created SystemCodegen");

        // Initialize HashCodegen - hash table operations
        hash_ = std::make_unique<eshkol::HashCodegen>(*ctx_, *tagged_, *mem, function_table, *arith_);
        hash_->setCodegenCallbacks(
            ControlFlowCallbacks::codegenASTWrapper,
            ControlFlowCallbacks::codegenTypedASTWrapper,
            this
        );
        eshkol_debug("Created HashCodegen");

        // Initialize LogicWorkspaceCodegen - consciousness engine primitives
        // (logic vars, KB, factor graphs, active inference, workspace, model_io)
        logic_workspace_ = std::make_unique<eshkol::LogicWorkspaceCodegen>(*ctx_, *tagged_);
        logic_workspace_->setCodegenASTCallback(ControlFlowCallbacks::codegenASTWrapper, this);
        logic_workspace_->setCodegenClosureCallCallback(ControlFlowCallbacks::closureCallWithInfoWrapper);
        eshkol_debug("Created LogicWorkspaceCodegen");

        // Initialize FunctionCodegen - lambda and closure operations
        // Note: The main implementations remain in this file for now
        func_ = std::make_unique<eshkol::FunctionCodegen>(*ctx_, *tagged_, *mem);
        eshkol_debug("Created FunctionCodegen");

        // Initialize ParallelCodegen - parallel execution primitives.
        // Its constructor emits worker dispatchers and a registration ctor, so
        // keep it out of freestanding object mode unless a hosted runtime is
        // available to satisfy those symbols.
        if (!freestanding_codegen_) {
            parallel_ = std::make_unique<eshkol::ParallelCodegen>(*ctx_);
            parallel_->setCodegenASTCallback(
                ControlFlowCallbacks::codegenASTWrapper,
                this
            );
            eshkol_debug("Created ParallelCodegen");
        } else {
            eshkol_debug("Skipped ParallelCodegen for freestanding object mode");
        }

        // Populate function_table for backward compatibility
        registerArenaFunctions();
    }

#endif
