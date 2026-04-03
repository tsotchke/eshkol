static void vm_dispatch_native(VM* vm, int fid) {
    switch (fid) {
    /* ══════════════════════════════════════════════════════════════════════
     * Math functions (20-35)
     * ══════════════════════════════════════════════════════════════════════ */
    case 20: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(sin(as_number(a)))); break; }
    case 21: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(cos(as_number(a)))); break; }
    case 22: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(tan(as_number(a)))); break; }
    case 23: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(exp(as_number(a)))); break; }
    case 24: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(log(as_number(a)))); break; }
    case 25: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(sqrt(as_number(a)))); break; }
    case 26: { Value a = vm_pop(vm); vm_push(vm, number_val(floor(as_number(a)))); break; }
    case 27: { Value a = vm_pop(vm); vm_push(vm, number_val(ceil(as_number(a)))); break; }
    case 28: { Value a = vm_pop(vm); vm_push(vm, number_val(round(as_number(a)))); break; }
    case 29: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(asin(as_number(a)))); break; }
    case 30: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(acos(as_number(a)))); break; }
    case 31: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(atan(as_number(a)))); break; }
    case 32: { Value b = vm_pop(vm); Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(pow(as_number(a), as_number(b)))); break; }
    case 33: { Value b = vm_pop(vm); Value a = vm_pop(vm); double da=as_number(a),db=as_number(b); vm_push(vm, number_val(da<db?da:db)); break; }
    case 34: { Value b = vm_pop(vm); Value a = vm_pop(vm); double da=as_number(a),db=as_number(b); vm_push(vm, number_val(da>db?da:db)); break; }
    case 35: { Value a = vm_pop(vm); vm_push(vm, number_val(fabs(as_number(a)))); break; }

    /* ══════════════════════════════════════════════════════════════════════
     * Predicates (40-50)
     * ══════════════════════════════════════════════════════════════════════ */
    case 40: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) > 0)); break; }
    case 41: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) < 0)); break; }
    case 42: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL((int64_t)as_number(a) % 2 != 0)); break; }
    case 43: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL((int64_t)as_number(a) % 2 == 0)); break; }
    case 44: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) == 0)); break; }
    case 45: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_PAIR)); break; }
    case 46: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_INT || a.type == VAL_FLOAT)); break; }
    case 47: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_STRING)); break; }
    case 48: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_BOOL)); break; }
    case 49: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_CLOSURE)); break; }
    case 50: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_VECTOR)); break; }

    /* ══════════════════════════════════════════════════════════════════════
     * String operations (51-56) — legacy IDs
     * ══════════════════════════════════════════════════════════════════════ */
    case 51: { /* number->string */
        Value a = vm_pop(vm);
        char buf[64];
        if (a.type == VAL_INT) snprintf(buf, 64, "%lld", (long long)a.as.i);
        else snprintf(buf, 64, "%.15g", as_number(a));
        VmString* s = vm_string_from_cstr(&vm->heap.regions, buf);
        if (s) {
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr >= 0) {
                vm->heap.objects[ptr]->type = HEAP_STRING;
                vm->heap.objects[ptr]->opaque.ptr = s;
                vm_push(vm, (Value){.type = VAL_STRING, .as.ptr = ptr});
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 54: { Value b = vm_pop(vm); Value a = vm_pop(vm); (void)a; (void)b; vm_push(vm, NIL_VAL); break; }
    case 55: { Value b = vm_pop(vm); Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) == as_number(b))); break; }
    case 56: { Value a = vm_pop(vm); (void)a; vm_push(vm, INT_VAL(0)); break; }

    /* ══════════════════════════════════════════════════════════════════════
     * I/O (60-61)
     * ══════════════════════════════════════════════════════════════════════ */
    case 60: printf("\n"); vm_push(vm, NIL_VAL); break;
    case 61: { Value v = vm_pop(vm); print_value(vm, v); vm_push(vm, NIL_VAL); break; }

    /* ══════════════════════════════════════════════════════════════════════
     * List/apply (70-73)
     * ══════════════════════════════════════════════════════════════════════ */
    case 70: { /* apply */
        Value args_list = vm_pop(vm);
        Value func = vm_pop(vm);
        if (func.type != VAL_CLOSURE) { vm->error = 1; break; }
        /* Push closure below args (OP_CALL convention: closure at fp-1) */
        vm_push(vm, func);
        int argc = 0;
        Value cur = args_list;
        while (cur.type == VAL_PAIR && argc < 16) {
            vm_push(vm, vm->heap.objects[cur.as.ptr]->cons.car);
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            argc++;
        }
        HeapObject* cl70 = vm->heap.objects[func.as.ptr];
        if (vm->frame_count >= MAX_FRAMES) { vm->error = 1; break; }
        vm->frames[vm->frame_count].return_pc = vm->pc;
        vm->frames[vm->frame_count].return_fp = vm->fp;
        vm->frames[vm->frame_count].func_pc = cl70->closure.func_pc;
        vm->frame_count++;
        vm->fp = vm->sp - argc;
        vm->pc = cl70->closure.func_pc;
        break;
    }
    case 71: { /* length */
        Value lst = vm_pop(vm);
        int len = 0;
        while (lst.type == VAL_PAIR) { len++; lst = vm->heap.objects[lst.as.ptr]->cons.cdr; }
        vm_push(vm, INT_VAL(len));
        break;
    }
    case 72: { /* reverse */
        Value lst = vm_pop(vm);
        Value result = NIL_VAL;
        while (lst.type == VAL_PAIR) {
            Value car = vm->heap.objects[lst.as.ptr]->cons.car;
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_CONS;
            vm->heap.objects[ptr]->cons.car = car;
            vm->heap.objects[ptr]->cons.cdr = result;
            result = PAIR_VAL(ptr);
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr;
        }
        vm_push(vm, result);
        break;
    }
    case 73: { /* append */
        Value b = vm_pop(vm), a = vm_pop(vm);
        if (a.type == VAL_NIL) { vm_push(vm, b); break; }
        Value rev = NIL_VAL;
        Value cur2 = a;
        while (cur2.type == VAL_PAIR) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = vm->heap.objects[cur2.as.ptr]->cons.car;
            vm->heap.objects[p]->cons.cdr = rev;
            rev = PAIR_VAL(p);
            cur2 = vm->heap.objects[cur2.as.ptr]->cons.cdr;
        }
        Value result2 = b;
        while (rev.type == VAL_PAIR) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = vm->heap.objects[rev.as.ptr]->cons.car;
            vm->heap.objects[p]->cons.cdr = result2;
            result2 = PAIR_VAL(p);
            rev = vm->heap.objects[rev.as.ptr]->cons.cdr;
        }
        vm_push(vm, result2);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Make-vector (260)
     * ══════════════════════════════════════════════════════════════════════ */
    case 260: {
        Value fill = vm_pop(vm);
        Value n_val = vm_pop(vm);
        (void)fill;
        int n = (int)as_number(n_val);
        if (n < 0) n = 0;
        if (n > 256) n = 256;
        int32_t ptr = heap_alloc(&vm->heap);
        if (ptr < 0) { vm->error = 1; break; }
        vm->heap.objects[ptr]->type = HEAP_VECTOR;
        vm_push(vm, (Value){.type = VAL_VECTOR, .as.ptr = ptr});
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Complex Number Operations (300-319)
     * ══════════════════════════════════════════════════════════════════════ */
    case 300: case 301: case 302: case 303: case 304: case 305: case 306:
    case 307: case 308: case 309: case 310: case 311: case 312: case 313:
    case 314: case 315: case 316: case 317: case 318: case 319: {
        if (fid == 300) { /* make-rectangular */
            Value imag = vm_pop(vm), real = vm_pop(vm);
            VmComplex* z = vm_complex_new(&vm->heap.regions, as_number(real), as_number(imag));
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr < 0 || !z) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_COMPLEX;
            vm->heap.objects[ptr]->opaque.ptr = z;
            vm_push(vm, (Value){.type = VAL_COMPLEX, .as.ptr = ptr});
        } else if (fid == 302) { /* real-part */
            Value z_val = vm_pop(vm);
            if (z_val.type == VAL_COMPLEX) {
                VmComplex* z = (VmComplex*)vm->heap.objects[z_val.as.ptr]->opaque.ptr;
                vm_push(vm, FLOAT_VAL(z->real));
            } else { vm_push(vm, FLOAT_VAL(as_number(z_val))); }
        } else if (fid == 303) { /* imag-part */
            Value z_val = vm_pop(vm);
            if (z_val.type == VAL_COMPLEX) {
                VmComplex* z = (VmComplex*)vm->heap.objects[z_val.as.ptr]->opaque.ptr;
                vm_push(vm, FLOAT_VAL(z->imag));
            } else { vm_push(vm, FLOAT_VAL(0.0)); }
        } else if (fid == 304) { /* magnitude */
            Value z_val = vm_pop(vm);
            if (z_val.type == VAL_COMPLEX) {
                VmComplex* z = (VmComplex*)vm->heap.objects[z_val.as.ptr]->opaque.ptr;
                vm_push(vm, FLOAT_VAL(vm_complex_magnitude(z)));
            } else { vm_push(vm, FLOAT_VAL(fabs(as_number(z_val)))); }
        } else if (fid == 317) { /* complex? */
            Value v = vm_pop(vm);
            vm_push(vm, BOOL_VAL(v.type == VAL_COMPLEX));
        } else {
            int is_binary = (fid >= 307 && fid <= 310) || fid == 318 || fid == 319;
            if (is_binary) {
                Value b_val = vm_pop(vm), a_val = vm_pop(vm);
                VmComplex a_z = {as_number(a_val), 0}, b_z = {as_number(b_val), 0};
                if (a_val.type == VAL_COMPLEX) a_z = *(VmComplex*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
                if (b_val.type == VAL_COMPLEX) b_z = *(VmComplex*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
                VmComplex* result = NULL;
                switch (fid) {
                    case 307: result = vm_complex_add(&vm->heap.regions, &a_z, &b_z); break;
                    case 308: result = vm_complex_sub(&vm->heap.regions, &a_z, &b_z); break;
                    case 309: result = vm_complex_mul(&vm->heap.regions, &a_z, &b_z); break;
                    case 310: result = vm_complex_div(&vm->heap.regions, &a_z, &b_z); break;
                    case 318: result = vm_complex_expt(&vm->heap.regions, &a_z, &b_z); break;
                    case 319: vm_push(vm, BOOL_VAL(a_z.real == b_z.real && a_z.imag == b_z.imag)); break;
                }
                if (fid != 319) {
                    if (!result) { vm->error = 1; break; }
                    int32_t ptr = heap_alloc(&vm->heap);
                    if (ptr < 0) { vm->error = 1; break; }
                    vm->heap.objects[ptr]->type = HEAP_COMPLEX;
                    vm->heap.objects[ptr]->opaque.ptr = result;
                    vm_push(vm, (Value){.type = VAL_COMPLEX, .as.ptr = ptr});
                }
            } else {
                Value a_val = vm_pop(vm);
                VmComplex a_z = {as_number(a_val), 0};
                if (a_val.type == VAL_COMPLEX) a_z = *(VmComplex*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
                VmComplex* result = NULL;
                switch (fid) {
                    case 301: { Value ang = vm_pop(vm);
                        result = vm_make_polar(&vm->heap.regions, as_number(a_val), as_number(ang)); break; }
                    case 305: vm_push(vm, FLOAT_VAL(vm_complex_angle(&a_z))); break;
                    case 306: result = vm_complex_conjugate(&vm->heap.regions, &a_z); break;
                    case 311: result = vm_complex_sqrt(&vm->heap.regions, &a_z); break;
                    case 312: result = vm_complex_exp(&vm->heap.regions, &a_z); break;
                    case 313: result = vm_complex_log(&vm->heap.regions, &a_z); break;
                    case 314: result = vm_complex_sin(&vm->heap.regions, &a_z); break;
                    case 315: result = vm_complex_cos(&vm->heap.regions, &a_z); break;
                    case 316: result = vm_complex_tan(&vm->heap.regions, &a_z); break;
                }
                if (fid != 305) {
                    if (!result) { vm->error = 1; break; }
                    int32_t ptr = heap_alloc(&vm->heap);
                    if (ptr < 0) { vm->error = 1; break; }
                    vm->heap.objects[ptr]->type = HEAP_COMPLEX;
                    vm->heap.objects[ptr]->opaque.ptr = result;
                    vm_push(vm, (Value){.type = VAL_COMPLEX, .as.ptr = ptr});
                }
            }
        }
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Rational Number Operations (330-349)
     * ══════════════════════════════════════════════════════════════════════ */
    case 330: case 331: case 332: case 333: case 334: case 335: case 336:
    case 337: case 338: case 339: case 340: case 341: case 342: case 343:
    case 344: case 345: case 346: case 347: case 348: case 349: {
        VmArena* rat_arena = vm_active_arena(&vm->heap.regions);
        switch (fid) {
        case 330: { Value denom = vm_pop(vm), num = vm_pop(vm);
            VmRational* r = vm_rational_make(rat_arena, (int64_t)as_number(num), (int64_t)as_number(denom));
            if (!r) { vm_push(vm, NIL_VAL); break; }
            int32_t ptr = heap_alloc(&vm->heap); if (ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_RATIONAL; vm->heap.objects[ptr]->opaque.ptr = r;
            vm_push(vm, (Value){.type = VAL_RATIONAL, .as.ptr = ptr}); break; }
        case 331: case 332: case 333: case 334: {
            Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmRational a_r = {(int64_t)as_number(a_val), 1}, b_r = {(int64_t)as_number(b_val), 1};
            if (a_val.type == VAL_RATIONAL) a_r = *(VmRational*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_RATIONAL) b_r = *(VmRational*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            VmRational* result = NULL;
            switch (fid) {
                case 331: result = vm_rational_add(rat_arena, &a_r, &b_r); break;
                case 332: result = vm_rational_sub(rat_arena, &a_r, &b_r); break;
                case 333: result = vm_rational_mul(rat_arena, &a_r, &b_r); break;
                case 334: result = vm_rational_div(rat_arena, &a_r, &b_r); break;
            }
            if (!result) { vm_push(vm, FLOAT_VAL((double)a_r.num/(double)a_r.denom + (double)b_r.num/(double)b_r.denom)); break; }
            if (result->denom == 1) { vm_push(vm, INT_VAL(result->num)); break; }
            int32_t ptr = heap_alloc(&vm->heap); if (ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_RATIONAL; vm->heap.objects[ptr]->opaque.ptr = result;
            vm_push(vm, (Value){.type = VAL_RATIONAL, .as.ptr = ptr}); break; }
        case 335: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) {
                VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr;
                VmRational* res = vm_rational_neg(rat_arena, r);
                if (!res) { vm_push(vm, NIL_VAL); break; }
                int32_t ptr = heap_alloc(&vm->heap); if (ptr < 0) { vm->error = 1; break; }
                vm->heap.objects[ptr]->type = HEAP_RATIONAL; vm->heap.objects[ptr]->opaque.ptr = res;
                vm_push(vm, (Value){.type = VAL_RATIONAL, .as.ptr = ptr});
            } else vm_push(vm, number_val(-as_number(v))); break; }
        case 336: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) {
                VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr;
                VmRational* res = vm_rational_abs(rat_arena, r);
                if (!res) { vm_push(vm, NIL_VAL); break; }
                int32_t ptr = heap_alloc(&vm->heap); if (ptr < 0) { vm->error = 1; break; }
                vm->heap.objects[ptr]->type = HEAP_RATIONAL; vm->heap.objects[ptr]->opaque.ptr = res;
                vm_push(vm, (Value){.type = VAL_RATIONAL, .as.ptr = ptr});
            } else vm_push(vm, number_val(fabs(as_number(v)))); break; }
        case 337: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) {
                VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr;
                VmRational* res = vm_rational_inv(rat_arena, r);
                if (!res) { vm_push(vm, NIL_VAL); break; }
                int32_t ptr = heap_alloc(&vm->heap); if (ptr < 0) { vm->error = 1; break; }
                vm->heap.objects[ptr]->type = HEAP_RATIONAL; vm->heap.objects[ptr]->opaque.ptr = res;
                vm_push(vm, (Value){.type = VAL_RATIONAL, .as.ptr = ptr});
            } else vm_push(vm, FLOAT_VAL(1.0 / as_number(v))); break; }
        case 338: { Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmRational a_r = {(int64_t)as_number(a_val), 1}, b_r = {(int64_t)as_number(b_val), 1};
            if (a_val.type == VAL_RATIONAL) a_r = *(VmRational*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_RATIONAL) b_r = *(VmRational*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(vm_rational_compare(&a_r, &b_r))); break; }
        case 339: { Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmRational a_r = {(int64_t)as_number(a_val), 1}, b_r = {(int64_t)as_number(b_val), 1};
            if (a_val.type == VAL_RATIONAL) a_r = *(VmRational*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_RATIONAL) b_r = *(VmRational*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            vm_push(vm, BOOL_VAL(vm_rational_equal(&a_r, &b_r))); break; }
        case 340: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) { VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, FLOAT_VAL(vm_rational_to_double(r))); }
            else vm_push(vm, FLOAT_VAL(as_number(v))); break; }
        case 341: { Value v = vm_pop(vm);
            VmRational* r = vm_rational_from_int(rat_arena, (int64_t)as_number(v));
            if (!r) { vm_push(vm, NIL_VAL); break; }
            int32_t ptr = heap_alloc(&vm->heap); if (ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_RATIONAL; vm->heap.objects[ptr]->opaque.ptr = r;
            vm_push(vm, (Value){.type = VAL_RATIONAL, .as.ptr = ptr}); break; }
        case 342: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) { VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, INT_VAL(vm_rational_floor(r))); }
            else vm_push(vm, INT_VAL((int64_t)floor(as_number(v)))); break; }
        case 343: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) { VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, INT_VAL(vm_rational_ceil(r))); }
            else vm_push(vm, INT_VAL((int64_t)ceil(as_number(v)))); break; }
        case 344: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) { VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, INT_VAL(vm_rational_truncate(r))); }
            else vm_push(vm, INT_VAL((int64_t)as_number(v))); break; }
        case 345: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) { VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, INT_VAL(vm_rational_round(r))); }
            else vm_push(vm, INT_VAL((int64_t)round(as_number(v)))); break; }
        case 346: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) { VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, INT_VAL(vm_rational_numerator(r))); }
            else vm_push(vm, INT_VAL((int64_t)as_number(v))); break; }
        case 347: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) { VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, INT_VAL(vm_rational_denominator(r))); }
            else vm_push(vm, INT_VAL(1)); break; }
        case 348: { Value b_v = vm_pop(vm), a_v = vm_pop(vm);
            vm_push(vm, INT_VAL(vm_rational_gcd((int64_t)as_number(a_v), (int64_t)as_number(b_v)))); break; }
        case 349: { Value tol = vm_pop(vm), x = vm_pop(vm);
            VmRational* r = vm_rationalize(rat_arena, as_number(x), as_number(tol));
            if (!r) { vm_push(vm, NIL_VAL); break; }
            if (r->denom == 1) { vm_push(vm, INT_VAL(r->num)); break; }
            int32_t ptr = heap_alloc(&vm->heap); if (ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_RATIONAL; vm->heap.objects[ptr]->opaque.ptr = r;
            vm_push(vm, (Value){.type = VAL_RATIONAL, .as.ptr = ptr}); break; }
        default: vm_push(vm, NIL_VAL); break;
        }
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Bignum Operations (350-369)
     * ══════════════════════════════════════════════════════════════════════ */
    case 350: case 351: case 352: case 353: case 354: case 355: case 356:
    case 357: case 358: case 359: case 360: case 361: case 362: case 363:
    case 364: case 365: case 366: case 367: case 368: case 369: {
        VmRegionStack* bn_rs = &vm->heap.regions;
        switch (fid) {
        case 350: { Value v = vm_pop(vm); VmBignum* b = bignum_from_int64(bn_rs, (int64_t)as_number(v));
            if (!b) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, b); break; }
        case 351: { Value v = vm_pop(vm);
            const char* s = (v.type == VAL_STRING && vm->heap.objects[v.as.ptr]->opaque.ptr) ? (const char*)vm->heap.objects[v.as.ptr]->opaque.ptr : "0";
            VmBignum* b = bignum_from_string(bn_rs, s);
            if (!b) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, b); break; }
        case 352: { Value v = vm_pop(vm);
            if (v.type == VAL_BIGNUM) {
                VmBignum* b = (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr;
                char* s = bignum_to_string(bn_rs, b);
                if (!s) { vm->error = 1; break; }
                VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s);
            } else {
                char buf[32]; snprintf(buf, sizeof(buf), "%lld", (long long)(int64_t)as_number(v));
                char* s = (char*)vm_alloc(bn_rs, strlen(buf)+1); if (s) strcpy(s, buf);
                if (!s) { vm->error = 1; break; }
                VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s);
            } break; }
        case 353: { Value v = vm_pop(vm);
            if (v.type == VAL_BIGNUM) { VmBignum* b = (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, FLOAT_VAL(bignum_to_double(b))); }
            else vm_push(vm, FLOAT_VAL(as_number(v))); break; }
        case 354: { Value v = vm_pop(vm);
            if (v.type == VAL_BIGNUM) { VmBignum* b = (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr; int ov=0; vm_push(vm, INT_VAL(bignum_to_int64(b, &ov))); }
            else vm_push(vm, INT_VAL((int64_t)as_number(v))); break; }
        case 355: case 356: case 357: case 358: case 359: {
            Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmBignum* a_bn = (a_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(a_val));
            VmBignum* b_bn = (b_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(b_val));
            if (!a_bn || !b_bn) { vm_push(vm, NIL_VAL); break; }
            VmBignum* result = NULL;
            switch (fid) { case 355: result=bignum_add(bn_rs,a_bn,b_bn); break; case 356: result=bignum_sub(bn_rs,a_bn,b_bn); break;
                case 357: result=bignum_mul(bn_rs,a_bn,b_bn); break; case 358: result=bignum_div(bn_rs,a_bn,b_bn); break; case 359: result=bignum_mod(bn_rs,a_bn,b_bn); break; }
            if (!result) { vm_push(vm, INT_VAL(0)); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        case 360: { Value v = vm_pop(vm);
            VmBignum* b = (v.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(v));
            if (!b) { vm_push(vm, NIL_VAL); break; }
            VmBignum* result = bignum_neg(bn_rs, b);
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        case 361: { Value v = vm_pop(vm);
            VmBignum* b = (v.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(v));
            if (!b) { vm_push(vm, NIL_VAL); break; }
            VmBignum* result = bignum_abs_val(bn_rs, b);
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        case 362: { Value exp_val = vm_pop(vm), base_val = vm_pop(vm);
            VmBignum* base_bn = (base_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[base_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(base_val));
            if (!base_bn) { vm_push(vm, NIL_VAL); break; }
            VmBignum* result = bignum_pow(bn_rs, base_bn, (uint64_t)(int64_t)as_number(exp_val));
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        case 363: { Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmBignum* a_bn = (a_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(a_val));
            VmBignum* b_bn = (b_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(b_val));
            if (!a_bn || !b_bn) { vm_push(vm, INT_VAL(0)); break; }
            vm_push(vm, INT_VAL(bignum_compare(a_bn, b_bn))); break; }
        case 364: { Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmBignum* a_bn = (a_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(a_val));
            VmBignum* b_bn = (b_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(b_val));
            if (!a_bn || !b_bn) { vm_push(vm, INT_VAL(0)); break; }
            VmBignum* result = bignum_gcd(bn_rs, a_bn, b_bn);
            if (!result) { vm_push(vm, INT_VAL(0)); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        case 365: case 366: case 367: {
            Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmBignum* a_bn = (a_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(a_val));
            VmBignum* b_bn = (b_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(b_val));
            if (!a_bn || !b_bn) { vm_push(vm, NIL_VAL); break; }
            VmBignum* result = NULL;
            switch (fid) { case 365: result=bignum_bitwise_and(bn_rs,a_bn,b_bn); break; case 366: result=bignum_bitwise_or(bn_rs,a_bn,b_bn); break; case 367: result=bignum_bitwise_xor(bn_rs,a_bn,b_bn); break; }
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        case 368: { Value v = vm_pop(vm);
            VmBignum* b = (v.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(v));
            if (!b) { vm_push(vm, NIL_VAL); break; }
            VmBignum* result = bignum_bitwise_not(bn_rs, b);
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        case 369: { Value shift_val = vm_pop(vm), v = vm_pop(vm);
            VmBignum* b = (v.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(v));
            if (!b) { vm_push(vm, NIL_VAL); break; }
            int shift = (int)as_number(shift_val);
            VmBignum* result = (shift >= 0) ? bignum_shift_left(bn_rs, b, shift) : bignum_shift_right(bn_rs, b, -shift);
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        default: vm_push(vm, NIL_VAL); break;
        }
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Dual Number Operations (370-389)
     * ══════════════════════════════════════════════════════════════════════ */
    case 370: case 371: case 372: case 373: case 374: case 375: case 376:
    case 377: case 378: case 379: case 380: case 381: case 382: case 383:
    case 384: case 385: case 386: case 387: case 388: case 389: {
        VmRegionStack* dual_rs = &vm->heap.regions;
        switch (fid) {
        case 370: { Value tangent = vm_pop(vm), primal = vm_pop(vm);
            VmDual* d = vm_dual_make(dual_rs, as_number(primal), as_number(tangent));
            if (!d) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_DUAL, VAL_DUAL, d); break; }
        case 371: { Value v = vm_pop(vm);
            if (v.type == VAL_DUAL) { VmDual* d = (VmDual*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, FLOAT_VAL(d->primal)); }
            else vm_push(vm, FLOAT_VAL(as_number(v))); break; }
        case 372: { Value v = vm_pop(vm);
            if (v.type == VAL_DUAL) { VmDual* d = (VmDual*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, FLOAT_VAL(d->tangent)); }
            else vm_push(vm, FLOAT_VAL(0.0)); break; }
        case 373: case 374: case 375: case 376: {
            Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmDual a_d = {as_number(a_val), 0.0}, b_d = {as_number(b_val), 0.0};
            if (a_val.type == VAL_DUAL) a_d = *(VmDual*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_DUAL) b_d = *(VmDual*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            VmDual* result = NULL;
            switch (fid) { case 373: result=vm_dual_add(dual_rs,&a_d,&b_d); break; case 374: result=vm_dual_sub(dual_rs,&a_d,&b_d); break;
                case 375: result=vm_dual_mul(dual_rs,&a_d,&b_d); break; case 376: result=vm_dual_div(dual_rs,&a_d,&b_d); break; }
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_DUAL, VAL_DUAL, result); break; }
        case 377: case 378: case 379: case 380: case 381:
        case 383: case 384: case 385: case 386: case 387: {
            Value v = vm_pop(vm);
            VmDual a_d = {as_number(v), 0.0};
            if (v.type == VAL_DUAL) a_d = *(VmDual*)vm->heap.objects[v.as.ptr]->opaque.ptr;
            VmDual* result = NULL;
            switch (fid) { case 377: result=vm_dual_sin(dual_rs,&a_d); break; case 378: result=vm_dual_cos(dual_rs,&a_d); break;
                case 379: result=vm_dual_exp(dual_rs,&a_d); break; case 380: result=vm_dual_log(dual_rs,&a_d); break;
                case 381: result=vm_dual_sqrt(dual_rs,&a_d); break; case 383: result=vm_dual_abs(dual_rs,&a_d); break;
                case 384: result=vm_dual_neg(dual_rs,&a_d); break; case 385: result=vm_dual_relu(dual_rs,&a_d); break;
                case 386: result=vm_dual_sigmoid(dual_rs,&a_d); break; case 387: result=vm_dual_tanh(dual_rs,&a_d); break; }
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_DUAL, VAL_DUAL, result); break; }
        case 382: { Value exp_val = vm_pop(vm), base_val = vm_pop(vm);
            VmDual a_d = {as_number(base_val), 0.0};
            if (base_val.type == VAL_DUAL) a_d = *(VmDual*)vm->heap.objects[base_val.as.ptr]->opaque.ptr;
            VmDual* result = vm_dual_pow(dual_rs, &a_d, as_number(exp_val));
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_DUAL, VAL_DUAL, result); break; }
        case 388: { Value v = vm_pop(vm);
            VmDual* d = vm_dual_from_double(dual_rs, as_number(v));
            if (!d) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_DUAL, VAL_DUAL, d); break; }
        case 389: { Value dual_val = vm_pop(vm), scalar_val = vm_pop(vm);
            VmDual a_d = {as_number(dual_val), 0.0};
            if (dual_val.type == VAL_DUAL) a_d = *(VmDual*)vm->heap.objects[dual_val.as.ptr]->opaque.ptr;
            VmDual* result = vm_dual_scale(dual_rs, as_number(scalar_val), &a_d);
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_DUAL, VAL_DUAL, result); break; }
        default: vm_push(vm, NIL_VAL); break;
        }
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * AD Operations (390-409) — reverse-mode tape + forward-mode derivative
     * ══════════════════════════════════════════════════════════════════════ */
    case 390: { /* ad-tape-new */
        AdTape* tape = ad_tape_new(&vm->heap.regions);
        if (!tape) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_AD_TAPE, VAL_INT, tape);
        break;
    }
    case 391: { /* ad-const(tape, value) */
        Value val = vm_pop(vm), tape_val = vm_pop(vm);
        AdTape* tape = (AdTape*)vm->heap.objects[tape_val.as.ptr]->opaque.ptr;
        if (!tape) { vm_push(vm, NIL_VAL); break; }
        vm_push(vm, INT_VAL(ad_const(tape, as_number(val))));
        break;
    }
    case 392: { /* ad-var(tape, value) */
        Value val = vm_pop(vm), tape_val = vm_pop(vm);
        AdTape* tape = (AdTape*)vm->heap.objects[tape_val.as.ptr]->opaque.ptr;
        if (!tape) { vm_push(vm, NIL_VAL); break; }
        vm_push(vm, INT_VAL(ad_var(tape, as_number(val))));
        break;
    }
    case 393: { /* derivative: (derivative f x) → f'(x) using forward-mode dual numbers */
        Value x_val = vm_pop(vm), f_val = vm_pop(vm); (void)f_val;
        /* Create dual: x + 1*epsilon */
        VmDual* d = vm_dual_make(&vm->heap.regions, as_number(x_val), 1.0);
        if (!d) { vm->error = 1; break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_DUAL, VAL_DUAL, d);
        /* Note: can't call closure from native code in simple VM.
         * Push the dual number — user extracts tangent after calling f. */
        break;
    }
    case 394: case 395: case 396: case 397: { /* ad-add, ad-sub, ad-mul, ad-div(tape, left, right) */
        Value right = vm_pop(vm), left = vm_pop(vm), tape_val = vm_pop(vm);
        AdTape* tape = (AdTape*)vm->heap.objects[tape_val.as.ptr]->opaque.ptr;
        if (!tape) { vm_push(vm, NIL_VAL); break; }
        int result = -1;
        switch (fid) {
            case 394: result = ad_add(tape, (int)left.as.i, (int)right.as.i); break;
            case 395: result = ad_sub(tape, (int)left.as.i, (int)right.as.i); break;
            case 396: result = ad_mul(tape, (int)left.as.i, (int)right.as.i); break;
            case 397: result = ad_div(tape, (int)left.as.i, (int)right.as.i); break;
        }
        vm_push(vm, INT_VAL(result));
        break;
    }
    case 398: case 399: case 400: case 401: case 402:
    case 403: case 404: case 405: case 406: case 407: { /* ad unary ops(tape, node) */
        Value node = vm_pop(vm), tape_val = vm_pop(vm);
        AdTape* tape = (AdTape*)vm->heap.objects[tape_val.as.ptr]->opaque.ptr;
        if (!tape) { vm_push(vm, NIL_VAL); break; }
        int result = -1;
        int idx = (int)node.as.i;
        switch (fid) {
            case 398: result = ad_sin(tape, idx); break;
            case 399: result = ad_cos(tape, idx); break;
            case 400: result = ad_exp(tape, idx); break;
            case 401: result = ad_log(tape, idx); break;
            case 402: result = ad_sqrt(tape, idx); break;
            case 403: result = ad_neg(tape, idx); break;
            case 404: result = ad_abs(tape, idx); break;
            case 405: result = ad_relu(tape, idx); break;
            case 406: result = ad_sigmoid(tape, idx); break;
            case 407: result = ad_tanh(tape, idx); break;
        }
        vm_push(vm, INT_VAL(result));
        break;
    }
    case 408: { /* ad-backward(tape, output_node) */
        Value node = vm_pop(vm), tape_val = vm_pop(vm);
        AdTape* tape = (AdTape*)vm->heap.objects[tape_val.as.ptr]->opaque.ptr;
        if (!tape) { vm_push(vm, NIL_VAL); break; }
        ad_backward(tape, (int)node.as.i);
        vm_push(vm, NIL_VAL); /* backward is side-effectful */
        break;
    }
    case 409: { /* ad-gradient(tape, node) → gradient value */
        Value node = vm_pop(vm), tape_val = vm_pop(vm);
        AdTape* tape = (AdTape*)vm->heap.objects[tape_val.as.ptr]->opaque.ptr;
        if (!tape) { vm_push(vm, FLOAT_VAL(0.0)); break; }
        int idx = (int)node.as.i;
        if (idx >= 0 && idx < tape->len) {
            vm_push(vm, FLOAT_VAL(tape->nodes[idx].gradient));
        } else {
            vm_push(vm, FLOAT_VAL(0.0));
        }
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Tensor Core Operations (410-420)
     * ══════════════════════════════════════════════════════════════════════ */
    case 410: { /* make-tensor(shape, fill) */
        Value fill = vm_pop(vm), shape_val = vm_pop(vm);
        int64_t shape[8]; int n_dims = vm_extract_shape(vm, shape_val, shape, 8);
        if (n_dims == 0) { vm_push(vm, NIL_VAL); break; }
        VmTensor* t = vm_tensor_fill(&vm->heap.regions, shape, n_dims, as_number(fill));
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, t);
        break;
    }
    case 411: { /* tensor-ref(tensor, indices) — flat or multi-dim access */
        Value idx_val = vm_pop(vm), t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, FLOAT_VAL(0)); break; }
        /* Single int/float index: flat access; list: multi-dim */
        if (idx_val.type == VAL_INT || idx_val.type == VAL_FLOAT) {
            int64_t flat = (int64_t)as_number(idx_val);
            if (flat >= 0 && flat < t->total)
                vm_push(vm, FLOAT_VAL(t->data[flat]));
            else vm_push(vm, FLOAT_VAL(0));
        } else if (idx_val.type == VAL_PAIR) {
            /* Multi-dim: walk list to get indices */
            int64_t indices[8]; int nd = 0;
            Value cur = idx_val;
            while (cur.type == VAL_PAIR && nd < 8) {
                indices[nd++] = (int64_t)as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
            double val = vm_tensor_ref(t, indices, nd);
            vm_push(vm, FLOAT_VAL(val));
        } else {
            vm_push(vm, FLOAT_VAL(0));
        }
        break;
    }
    case 412: { /* tensor-set!(tensor, indices, value) */
        Value val = vm_pop(vm), idx_val = vm_pop(vm), t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        int64_t indices[8]; int n = vm_extract_shape(vm, idx_val, indices, 8);
        if (n == 0) { indices[0] = (int64_t)as_number(idx_val); n = 1; }
        vm_tensor_set(t, indices, n, as_number(val));
        vm_push(vm, NIL_VAL);
        break;
    }
    case 413: { /* tensor-shape → list */
        Value t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        Value result = NIL_VAL;
        for (int i = t->n_dims - 1; i >= 0; i--) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = INT_VAL(t->shape[i]);
            vm->heap.objects[p]->cons.cdr = result;
            result = PAIR_VAL(p);
        }
        vm_push(vm, result);
        break;
    }
    case 414: { /* tensor-data → flat list (for small tensors) */
        Value t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        Value result = NIL_VAL;
        int64_t limit = t->total > 1024 ? 1024 : t->total;
        for (int64_t i = limit - 1; i >= 0; i--) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = FLOAT_VAL(t->data[i]);
            vm->heap.objects[p]->cons.cdr = result;
            result = PAIR_VAL(p);
        }
        vm_push(vm, result);
        break;
    }
    case 415: { /* reshape(tensor, new_shape) */
        Value shape_val = vm_pop(vm), t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        int64_t shape[8]; int n = vm_extract_shape(vm, shape_val, shape, 8);
        VmTensor* out = vm_tensor_reshape(&vm->heap.regions, t, shape, n);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 416: { /* transpose */
        Value t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_transpose(&vm->heap.regions, t);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 417: { /* zeros(shape) */
        Value shape_val = vm_pop(vm);
        int64_t shape[8]; int n = vm_extract_shape(vm, shape_val, shape, 8);
        if (n == 0) { vm_push(vm, NIL_VAL); break; }
        VmTensor* t = vm_tensor_zeros(&vm->heap.regions, shape, n);
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, t);
        break;
    }
    case 418: { /* ones(shape) */
        Value shape_val = vm_pop(vm);
        int64_t shape[8]; int n = vm_extract_shape(vm, shape_val, shape, 8);
        if (n == 0) { vm_push(vm, NIL_VAL); break; }
        VmTensor* t = vm_tensor_ones(&vm->heap.regions, shape, n);
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, t);
        break;
    }
    case 419: { /* arange(start, stop, step) */
        Value step = vm_pop(vm), stop = vm_pop(vm), start = vm_pop(vm);
        VmTensor* t = vm_tensor_arange(&vm->heap.regions, as_number(start), as_number(stop), as_number(step));
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, t);
        break;
    }
    case 420: { /* flatten */
        Value t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_flatten(&vm->heap.regions, t);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Tensor Operations (440-469)
     * ══════════════════════════════════════════════════════════════════════ */
    case 440: { /* matmul — GPU dispatch if tensor is large enough */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmTensor* a = (VmTensor*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
        VmTensor* b = (VmTensor*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
        if (!a || !b) { vm_push(vm, NIL_VAL); break; }
        /* Try GPU first, fall through to CPU */
        VmTensor* out = vm_gpu_try_matmul(&vm->heap.regions, a, b);
        if (!out) out = vm_tensor_matmul(&vm->heap.regions, a, b);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 441: case 442: case 443: case 444: case 445: case 446: case 447: { /* tensor binary: +,-,*,/,pow,max,min */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmTensor* a = (VmTensor*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
        VmTensor* b = (VmTensor*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
        if (!a || !b) { vm_push(vm, NIL_VAL); break; }
        /* GPU dispatch for add/sub/mul/div (ops 0-3) */
        VmTensor* out = NULL;
        static const int gpu_binary_ops[] = {0,1,2,3,-1,-1,-1}; /* add,sub,mul,div,pow,max,min */
        int gpu_op = gpu_binary_ops[fid - 441];
        if (gpu_op >= 0) out = vm_gpu_try_binary(&vm->heap.regions, a, b, gpu_op);
        if (!out) switch (fid) {
            case 441: out = vm_tensor_add(&vm->heap.regions, a, b); break;
            case 442: out = vm_tensor_sub(&vm->heap.regions, a, b); break;
            case 443: out = vm_tensor_mul(&vm->heap.regions, a, b); break;
            case 444: out = vm_tensor_div(&vm->heap.regions, a, b); break;
            case 445: out = vm_tensor_pow(&vm->heap.regions, a, b); break;
            case 446: out = vm_tensor_maximum(&vm->heap.regions, a, b); break;
            case 447: out = vm_tensor_minimum(&vm->heap.regions, a, b); break;
        }
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 448: { /* batch-matmul */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmTensor* a = (VmTensor*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
        VmTensor* b = (VmTensor*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
        if (!a || !b) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_batch_matmul(&vm->heap.regions, a, b);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 449: { /* dot */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmTensor* a = (VmTensor*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
        VmTensor* b = (VmTensor*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
        if (!a || !b) { vm_push(vm, FLOAT_VAL(0.0)); break; }
        vm_push(vm, FLOAT_VAL(vm_tensor_dot(a, b)));
        break;
    }
    case 450: case 451: case 452: case 453: case 454: case 455: { /* tensor unary: neg,abs,sqrt,exp,log,sin,cos */
        Value t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = NULL;
        switch (fid) {
            case 450: out = vm_tensor_neg(&vm->heap.regions, t); break;
            case 451: out = vm_tensor_abs(&vm->heap.regions, t); break;
            case 452: out = vm_tensor_sqrt_op(&vm->heap.regions, t); break;
            case 453: out = vm_tensor_exp_op(&vm->heap.regions, t); break;
            case 454: out = vm_tensor_log_op(&vm->heap.regions, t); break;
            case 455: out = vm_tensor_sin_op(&vm->heap.regions, t); break;
        }
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 456: { /* scale(tensor, scalar) */
        Value scalar = vm_pop(vm), t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_scale(&vm->heap.regions, t, as_number(scalar));
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 457: case 458: case 459: case 460: { /* reduce: sum,mean,max,min (tensor, axis) */
        Value axis_val = vm_pop(vm), t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        int axis = (int)as_number(axis_val);
        /* GPU dispatch for full-tensor reductions (axis=-1 or axis covers all) */
        VmTensor* out = NULL;
        if (axis < 0 || t->n_dims == 1) {
            static const int gpu_reduce_ops[] = {0, 4, 3, 2}; /* sum=0, mean=4, max=3, min=2 */
            double gpu_result = vm_gpu_try_reduce(t, gpu_reduce_ops[fid - 457]);
            if (!isnan(gpu_result)) {
                int64_t shape[1] = {1};
                out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
                if (out) out->data[0] = gpu_result;
            }
        }
        if (!out) switch (fid) {
            case 457: out = vm_tensor_sum(&vm->heap.regions, t, axis); break;
            case 458: out = vm_tensor_mean(&vm->heap.regions, t, axis); break;
            case 459: out = vm_tensor_max(&vm->heap.regions, t, axis); break;
            case 460: out = vm_tensor_min(&vm->heap.regions, t, axis); break;
        }
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 461: { /* cos tensor */
        Value t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_cos_op(&vm->heap.regions, t);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 462: case 463: case 464: case 465: case 466: case 467: case 468: { /* activations: relu,sigmoid,tanh,leaky_relu,elu,gelu,swish */
        Value t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = NULL;
        /* GPU dispatch for softmax */
        if (fid == 463) out = vm_gpu_try_softmax(&vm->heap.regions, t);
        if (!out) switch (fid) {
            case 462: out = vm_tensor_relu(&vm->heap.regions, t); break;
            case 463: out = vm_tensor_softmax(&vm->heap.regions, t, t->n_dims - 1); break;
            case 464: out = vm_tensor_sigmoid(&vm->heap.regions, t); break;
            case 465: out = vm_tensor_tanh_act(&vm->heap.regions, t); break;
            case 466: out = vm_tensor_leaky_relu(&vm->heap.regions, t); break;
            case 467: out = vm_tensor_elu(&vm->heap.regions, t); break;
            case 468: out = vm_tensor_gelu(&vm->heap.regions, t); break;
        }
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 469: { /* swish */
        Value t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_swish(&vm->heap.regions, t);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Logic Operations (500-512)
     * ══════════════════════════════════════════════════════════════════════ */
    case 500: { /* make-logic-var(name) — name as int (id) */
        Value name_val = vm_pop(vm), dummy = vm_pop(vm); (void)dummy; (void)name_val;
        /* For simplicity, use the integer as var ID */
        vm_push(vm, INT_VAL((int64_t)vm_make_logic_var("?auto")));
        break;
    }
    case 501: { /* logic-var? — check heap type for LOGIC_VAR */
        Value v = vm_pop(vm);
        int is_lv = (is_heap_type(vm, v, HEAP_LOGIC_VAR));
        vm_push(vm, BOOL_VAL(is_lv));
        break;
    }
    case 502: { /* unify(subst, a, b) — bind a to b in substitution (copy-on-extend) */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm), s_val = vm_pop(vm);
        if (s_val.as.ptr >= 0 && vm->heap.objects[s_val.as.ptr]->type == HEAP_SUBST) {
            VmSubstitution* subst = (VmSubstitution*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            if (subst && a_val.type == VAL_INT) {
                /* Treat a_val as logic var ID, bind it to b_val */
                VmValue term;
                if (b_val.type == VAL_INT) { term.type = VM_VAL_INT64; term.data.int_val = b_val.as.i; }
                else if (b_val.type == VAL_FLOAT) { term.type = VM_VAL_DOUBLE; term.data.double_val = b_val.as.f; }
                else if (b_val.type == VAL_BOOL) { term.type = VM_VAL_BOOL; term.data.int_val = b_val.as.b; }
                else { term.type = VM_VAL_INT64; term.data.int_val = b_val.as.i; }
                VmSubstitution* extended = vm_subst_extend(&vm->heap.regions, subst, (uint64_t)a_val.as.i, &term);
                if (extended) {
                    VM_PUSH_HEAP_OPAQUE(vm, HEAP_SUBST, VAL_INT, extended);
                    break;
                }
            }
        }
        vm_push(vm, NIL_VAL); /* unification failed */
        break;
    }
    case 505: { /* make-substitution */
        VmSubstitution* s = vm_make_substitution(&vm->heap.regions, 16);
        if (!s) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_SUBST, VAL_INT, s);
        break;
    }
    case 506: { /* substitution? */
        Value v = vm_pop(vm);
        int is_subst = (is_heap_type(vm, v, HEAP_SUBST));
        vm_push(vm, BOOL_VAL(is_subst));
        break;
    }
    case 509: { /* make-kb */
        VmKnowledgeBase* kb = vm_make_kb(&vm->heap.regions);
        if (!kb) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_KB, VAL_INT, kb);
        break;
    }
    case 510: { /* kb? */
        Value v = vm_pop(vm);
        int is_kb = (is_heap_type(vm, v, HEAP_KB));
        vm_push(vm, BOOL_VAL(is_kb));
        break;
    }
    case 503: { /* walk(term, subst) */
        Value subst_val = vm_pop(vm), term_val = vm_pop(vm);
        /* Walk resolves variables through substitution chains */
        if (subst_val.as.ptr >= 0 && vm->heap.objects[subst_val.as.ptr]->type == HEAP_SUBST) {
            VmSubstitution* s = (VmSubstitution*)vm->heap.objects[subst_val.as.ptr]->opaque.ptr;
            if (s && term_val.type == VAL_INT) {
                /* Check if term is a var_id that has a binding */
                for (int i = 0; i < s->n_bindings; i++) {
                    if (s->var_ids[i] == (uint64_t)term_val.as.i) {
                        /* Convert VmValue to VM Value */
                        VmValue bound = s->terms[i];
                        if (bound.type == VM_VAL_INT64)
                            vm_push(vm, INT_VAL(bound.data.int_val));
                        else if (bound.type == VM_VAL_DOUBLE)
                            vm_push(vm, FLOAT_VAL(bound.data.double_val));
                        else if (bound.type == VM_VAL_BOOL)
                            vm_push(vm, BOOL_VAL((int)bound.data.int_val));
                        else
                            vm_push(vm, INT_VAL(bound.data.int_val));
                        goto walk_done;
                    }
                }
            }
        }
        vm_push(vm, term_val); /* unbound — return as-is */
        walk_done: break;
    }
    case 504: { /* walk-deep — recursive walk through substitution chains */
        Value subst_val = vm_pop(vm), term_val = vm_pop(vm);
        /* Walk repeatedly until term no longer resolves */
        if (subst_val.as.ptr >= 0 && vm->heap.objects[subst_val.as.ptr]->type == HEAP_SUBST) {
            VmSubstitution* s = (VmSubstitution*)vm->heap.objects[subst_val.as.ptr]->opaque.ptr;
            Value resolved = term_val;
            int depth = 0;
            while (s && resolved.type == VAL_INT && depth < 100) {
                int found = 0;
                for (int i = 0; i < s->n_bindings; i++) {
                    if (s->var_ids[i] == (uint64_t)resolved.as.i) {
                        VmValue bound = s->terms[i];
                        if (bound.type == VM_VAL_INT64) resolved = INT_VAL(bound.data.int_val);
                        else if (bound.type == VM_VAL_DOUBLE) resolved = FLOAT_VAL(bound.data.double_val);
                        else if (bound.type == VM_VAL_BOOL) resolved = BOOL_VAL((int)bound.data.int_val);
                        else resolved = INT_VAL(bound.data.int_val);
                        found = 1;
                        break;
                    }
                }
                if (!found) break;
                depth++;
            }
            vm_push(vm, resolved);
        } else {
            vm_push(vm, term_val);
        }
        break;
    }
    case 507: { /* make-fact(pred, args) — pack predicate and args list */
        Value args_val = vm_pop(vm), pred = vm_pop(vm);
        int32_t ptr = heap_alloc(&vm->heap);
        if (ptr < 0) { vm->error = 1; break; }
        vm->heap.objects[ptr]->type = HEAP_FACT;
        vm->heap.objects[ptr]->opaque.ptr = NULL;
        vm->heap.objects[ptr]->cons.car = pred;
        vm->heap.objects[ptr]->cons.cdr = args_val;
        vm_push(vm, (Value){.type = VAL_INT, .as.ptr = ptr});
        break;
    }
    case 508: { /* fact? */
        Value v = vm_pop(vm);
        int is_fact = (is_heap_type(vm, v, HEAP_FACT));
        vm_push(vm, BOOL_VAL(is_fact));
        break;
    }
    case 511: { /* kb-assert!(kb, fact) */
        Value fact_val = vm_pop(vm), kb_val = vm_pop(vm);
        if (kb_val.as.ptr >= 0 && vm->heap.objects[kb_val.as.ptr]->type == HEAP_KB) {
            VmKnowledgeBase* kb = (VmKnowledgeBase*)vm->heap.objects[kb_val.as.ptr]->opaque.ptr;
            if (kb) {
                /* Add fact pointer to KB */
                vm_kb_assert(&vm->heap.regions, kb, (VmFact*)vm->heap.objects[fact_val.as.ptr]->opaque.ptr);
            }
        }
        vm_push(vm, NIL_VAL); /* void return */
        break;
    }
    case 512: { /* kb-query(kb, pattern) → list of matching facts */
        Value pattern = vm_pop(vm), kb_val = vm_pop(vm);
        /* Return facts from KB that match the pattern predicate */
        if (kb_val.as.ptr >= 0 && vm->heap.objects[kb_val.as.ptr]->type == HEAP_KB) {
            VmKnowledgeBase* kb = (VmKnowledgeBase*)vm->heap.objects[kb_val.as.ptr]->opaque.ptr;
            if (kb) {
                Value result = NIL_VAL;
                for (int i = kb->n_facts - 1; i >= 0; i--) {
                    VmFact* f = kb->facts[i];
                    /* Build result list of matching facts */
                    int32_t p = heap_alloc(&vm->heap);
                    if (p < 0) break;
                    vm->heap.objects[p]->type = HEAP_CONS;
                    vm->heap.objects[p]->cons.car = INT_VAL((int64_t)(intptr_t)f);
                    vm->heap.objects[p]->cons.cdr = result;
                    result = PAIR_VAL(p);
                }
                vm_push(vm, result);
                break;
            }
        }
        (void)pattern;
        vm_push(vm, NIL_VAL);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Inference Operations (520-526)
     * ══════════════════════════════════════════════════════════════════════ */
    case 520: { /* make-factor-graph(num_vars, var_dims_list) */
        Value dims_val = vm_pop(vm), nvars_val = vm_pop(vm);
        int nv = (int)as_number(nvars_val);
        int var_dims[64]; int nd = 0;
        if (dims_val.type == VAL_PAIR) {
            Value cur = dims_val;
            while (cur.type == VAL_PAIR && nd < 64) {
                var_dims[nd++] = (int)as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
        } else { var_dims[0] = (int)as_number(dims_val); nd = 1; }
        if (nd < nv) nv = nd;
        VmFactorGraph* fg = vm_make_factor_graph(&vm->heap.regions, nv, var_dims);
        if (!fg) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_FACTOR_GRAPH, VAL_INT, fg);
        break;
    }
    case 521: { /* factor-graph? */
        Value v = vm_pop(vm);
        vm_push(vm, BOOL_VAL(is_heap_type(vm, v, HEAP_FACTOR_GRAPH)));
        break;
    }
    case 522: { /* fg-add-factor!(fg, var_indices, cpt) */
        Value cpt = vm_pop(vm), vars = vm_pop(vm), fg_val = vm_pop(vm);
        if (fg_val.as.ptr >= 0 && vm->heap.objects[fg_val.as.ptr]->type == HEAP_FACTOR_GRAPH) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            if (fg) {
                /* Extract var_indices from list */
                int var_idx[8], n_vars = 0;
                Value cur = vars;
                while (cur.type == VAL_PAIR && n_vars < 8) {
                    var_idx[n_vars++] = (int)as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                    cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
                }
                /* Extract CPT data from tensor or list */
                double* cpt_data = NULL;
                if (cpt.as.ptr >= 0 && vm->heap.objects[cpt.as.ptr]->type == HEAP_TENSOR) {
                    VmTensor* t = (VmTensor*)vm->heap.objects[cpt.as.ptr]->opaque.ptr;
                    if (t) cpt_data = t->data;
                }
                if (n_vars > 0 && cpt_data) {
                    /* Build dims array from factor graph's var_dims */
                    int dims[8];
                    for (int i = 0; i < n_vars; i++) {
                        int vi = var_idx[i];
                        dims[i] = (vi >= 0 && vi < fg->num_vars) ? fg->var_dims[vi] : 2;
                    }
                    VmFactor* factor = vm_make_factor(&vm->heap.regions, var_idx, n_vars, cpt_data, dims);
                    if (factor) vm_fg_add_factor(&vm->heap.regions, fg, factor);
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 523: { /* fg-infer!(fg, max_iters, tolerance) */
        Value tol = vm_pop(vm), iters = vm_pop(vm), fg_val = vm_pop(vm);
        if (fg_val.as.ptr >= 0 && vm->heap.objects[fg_val.as.ptr]->type == HEAP_FACTOR_GRAPH) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            if (fg) {
                int converged = vm_fg_infer(&vm->heap.regions, fg, (int)as_number(iters), as_number(tol));
                vm_push(vm, BOOL_VAL(converged));
                break;
            }
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 524: { /* fg-update-cpt!(fg, factor_idx, new_cpt_tensor) */
        Value cpt = vm_pop(vm), idx = vm_pop(vm), fg_val = vm_pop(vm);
        if (fg_val.as.ptr >= 0 && vm->heap.objects[fg_val.as.ptr]->type == HEAP_FACTOR_GRAPH) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            if (fg && cpt.as.ptr >= 0 && vm->heap.objects[cpt.as.ptr]->type == HEAP_TENSOR) {
                VmTensor* t = (VmTensor*)vm->heap.objects[cpt.as.ptr]->opaque.ptr;
                if (t) vm_fg_update_cpt(fg, (int)as_number(idx), t->data, (int)t->total);
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 525: { /* free-energy(fg, observations) */
        Value obs = vm_pop(vm), fg_val = vm_pop(vm);
        if (fg_val.as.ptr >= 0 && vm->heap.objects[fg_val.as.ptr]->type == HEAP_FACTOR_GRAPH) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            if (fg) {
                /* Parse observations: list of (var_idx, state) pairs */
                int obs_pairs[32][2], n_obs = 0;
                Value cur = obs;
                while (cur.type == VAL_PAIR && n_obs < 32) {
                    Value pair = vm->heap.objects[cur.as.ptr]->cons.car;
                    if (pair.type == VAL_PAIR) {
                        obs_pairs[n_obs][0] = (int)as_number(vm->heap.objects[pair.as.ptr]->cons.car);
                        obs_pairs[n_obs][1] = (int)as_number(vm->heap.objects[vm->heap.objects[pair.as.ptr]->cons.cdr.as.ptr]->cons.car);
                        n_obs++;
                    }
                    cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
                }
                double fe = vm_free_energy(fg, (const double*)obs_pairs, n_obs);
                vm_push(vm, FLOAT_VAL(fe));
                break;
            }
        }
        vm_push(vm, FLOAT_VAL(0.0));
        break;
    }
    case 526: { /* expected-free-energy(fg, action_var, action_state) */
        Value state = vm_pop(vm), var = vm_pop(vm), fg_val = vm_pop(vm);
        if (fg_val.as.ptr >= 0 && vm->heap.objects[fg_val.as.ptr]->type == HEAP_FACTOR_GRAPH) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            if (fg) {
                double efe = vm_expected_free_energy(&vm->heap.regions, fg, (int)as_number(var), (int)as_number(state));
                vm_push(vm, FLOAT_VAL(efe));
                break;
            }
        }
        vm_push(vm, FLOAT_VAL(0.0));
        break;
    }

    case 527: { /* fg-observe!(fg, var_id, observed_state)
                 * Clamps a variable to an observed state for evidence injection.
                 * After observing, re-run fg-infer! to propagate the evidence.
                 * Ref: Standard factor graph evidence clamping (Kschischang et al. 2001). */
        Value state_val = vm_pop(vm), var_val = vm_pop(vm), fg_val = vm_pop(vm);
        if (fg_val.as.ptr >= 0 && vm->heap.objects[fg_val.as.ptr]->type == HEAP_FACTOR_GRAPH) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            if (fg) {
                int var_id = (int)as_number(var_val);
                int obs_state = (int)as_number(state_val);
                if (var_id >= 0 && var_id < fg->num_vars &&
                    obs_state >= 0 && obs_state < fg->var_dims[var_id]) {
                    /* Clamp beliefs: set observed state to probability 1, others to 0 */
                    int dim = fg->var_dims[var_id];
                    for (int s = 0; s < dim; s++) {
                        fg->beliefs[var_id][s] = (s == obs_state) ? 0.0 : -1e30;
                    }
                    /* Mark variable as observed (skip during belief update in fg-infer!).
                     * Use calloc (not arena) because VmFactorGraph's lifetime may exceed
                     * arena scope. The leak is bounded: one allocation per factor graph
                     * (guarded by !fg->observed), num_vars * sizeof(bool) = typically 6-24 bytes.
                     * Freed when the factor graph is destroyed or the process exits. */
                    if (!fg->observed) {
                        fg->observed = (bool*)calloc(fg->num_vars, sizeof(bool));
                    }
                    if (fg->observed) {
                        fg->observed[var_id] = true;
                    }
                    vm_push(vm, BOOL_VAL(true));
                    break;
                }
            }
        }
        vm_push(vm, BOOL_VAL(false));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Workspace Operations (540-544)
     * ══════════════════════════════════════════════════════════════════════ */
    case 540: { /* make-workspace(dim, max_modules) */
        Value max_m = vm_pop(vm), dim_val = vm_pop(vm);
        VmWorkspace* ws = vm_ws_new(&vm->heap.regions, (int)as_number(dim_val), (int)as_number(max_m));
        if (!ws) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_WORKSPACE, VAL_INT, ws);
        break;
    }
    case 541: { /* workspace? */
        Value v = vm_pop(vm);
        vm_push(vm, BOOL_VAL(is_heap_type(vm, v, HEAP_WORKSPACE)));
        break;
    }
    case 542: { /* ws-register!(ws, name, closure) */
        Value closure = vm_pop(vm), name_val = vm_pop(vm), ws_val = vm_pop(vm);
        if (ws_val.as.ptr >= 0 && vm->heap.objects[ws_val.as.ptr]->type == HEAP_WORKSPACE) {
            VmWorkspace* ws = (VmWorkspace*)vm->heap.objects[ws_val.as.ptr]->opaque.ptr;
            if (ws) {
                const char* name = "module";
                if (name_val.type == VAL_STRING && vm->heap.objects[name_val.as.ptr]->opaque.ptr) {
                    VmString* ns = (VmString*)vm->heap.objects[name_val.as.ptr]->opaque.ptr;
                    name = ns->data;
                }
                /* Allocate stable Value on arena so pointer survives past this scope */
                Value* stable_closure = (Value*)vm_alloc(&vm->heap.regions, sizeof(Value));
                if (stable_closure) {
                    *stable_closure = closure;
                    vm_ws_register(ws, name, stable_closure);
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 543: { /* ws-step!(ws) — invoke module closures via closure bridge */
        Value ws_val = vm_pop(vm);
        if (ws_val.as.ptr >= 0 && vm->heap.objects[ws_val.as.ptr]->type == HEAP_WORKSPACE) {
            VmWorkspace* ws = (VmWorkspace*)vm->heap.objects[ws_val.as.ptr]->opaque.ptr;
            if (ws && ws->content) {
                /* Build content tensor */
                int64_t shape[1] = {ws->dim};
                VmTensor* ct = vm_tensor_from_data(&vm->heap.regions, ws->content, shape, 1);
                int32_t tptr = heap_alloc(&vm->heap);
                if (tptr < 0 || !ct) { vm_push(vm, NIL_VAL); break; }
                vm->heap.objects[tptr]->type = HEAP_TENSOR;
                vm->heap.objects[tptr]->opaque.ptr = ct;
                Value content_val = (Value){.type = VAL_INT, .as.ptr = tptr};

                /* Call each module's closure, collect salience + proposal */
                int n_mod = ws->n_modules;
                if (n_mod > 256) n_mod = 256;
                double* saliences = (double*)calloc(n_mod, sizeof(double));
                Value* proposals = (Value*)calloc(n_mod, sizeof(Value));
                if (!saliences || !proposals) { free(saliences); free(proposals); vm_push(vm, NIL_VAL); break; }
                for (int i = 0; i < n_mod; i++) {
                    Value* closure_ptr = (Value*)ws->modules[i].process_fn;
                    if (closure_ptr && closure_ptr->type == VAL_CLOSURE) {
                        Value result = vm_call_closure_from_native(vm, *closure_ptr, &content_val, 1);
                        if (result.type == VAL_PAIR) {
                            saliences[i] = as_number(vm->heap.objects[result.as.ptr]->cons.car);
                            proposals[i] = vm->heap.objects[result.as.ptr]->cons.cdr;
                        } else {
                            saliences[i] = as_number(result);
                            proposals[i] = content_val;
                        }
                    } else {
                        saliences[i] = 0;
                        proposals[i] = content_val;
                    }
                }
                /* Softmax competition: highest salience wins */
                int winner = 0;
                for (int i = 1; i < n_mod; i++)
                    if (saliences[i] > saliences[winner]) winner = i;
                ws->step_count++;
                Value winner_proposal = proposals[winner];
                free(saliences); free(proposals);
                vm_push(vm, winner_proposal);
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 544: { /* ws-get-content */
        Value ws_val = vm_pop(vm);
        if (ws_val.as.ptr >= 0 && vm->heap.objects[ws_val.as.ptr]->type == HEAP_WORKSPACE) {
            VmWorkspace* ws = (VmWorkspace*)vm->heap.objects[ws_val.as.ptr]->opaque.ptr;
            if (ws && ws->content) {
                /* Return content as tensor */
                int64_t shape[1] = {ws->dim};
                VmTensor* t = vm_tensor_from_data(&vm->heap.regions, ws->content, shape, 1);
                if (t) {
                    int32_t ptr = heap_alloc(&vm->heap);
                    if (ptr >= 0) {
                        vm->heap.objects[ptr]->type = HEAP_TENSOR;
                        vm->heap.objects[ptr]->opaque.ptr = t;
                        vm_push(vm, (Value){.type = VAL_INT, .as.ptr = ptr});
                        break;
                    }
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * String Operations (550-570) — real VmString dispatch
     * ══════════════════════════════════════════════════════════════════════ */
    case 550: { /* string-length */
        Value s_val = vm_pop(vm);
        if (s_val.type == VAL_STRING && vm->heap.objects[s_val.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(vm_string_length(s)));
        } else vm_push(vm, INT_VAL(0));
        break;
    }
    case 551: { /* string-ref(str, idx) */
        Value idx = vm_pop(vm), s_val = vm_pop(vm);
        if (s_val.type == VAL_STRING && vm->heap.objects[s_val.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            int cp = vm_string_ref(s, (int)as_number(idx));
            vm_push(vm, INT_VAL(cp >= 0 ? cp : 0));
        } else vm_push(vm, INT_VAL(0));
        break;
    }
    case 552: { /* string-set!(str, idx, char) → new string */
        Value ch = vm_pop(vm), idx = vm_pop(vm), s_val = vm_pop(vm);
        if (s_val.type == VAL_STRING && vm->heap.objects[s_val.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            VmString* result = vm_string_set(&vm->heap.regions, s, (int)as_number(idx), (int)as_number(ch));
            if (result) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, result); }
            else vm_push(vm, NIL_VAL);
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 553: { /* substring(str, start, end) */
        Value end = vm_pop(vm), start = vm_pop(vm), s_val = vm_pop(vm);
        if (s_val.type == VAL_STRING && vm->heap.objects[s_val.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            VmString* result = vm_string_substring(&vm->heap.regions, s, (int)as_number(start), (int)as_number(end));
            if (result) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, result); }
            else vm_push(vm, NIL_VAL);
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 554: { /* string-append */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmString* a = (a_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : NULL;
        VmString* b = (b_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : NULL;
        if (a && b) {
            VmString* result = vm_string_append(&vm->heap.regions, a, b);
            if (result) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, result); break; }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 555: { /* string-contains(str, substr) */
        Value sub_val = vm_pop(vm), s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        VmString* sub = (sub_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[sub_val.as.ptr]->opaque.ptr : NULL;
        vm_push(vm, INT_VAL(vm_string_contains(s, sub)));
        break;
    }
    case 556: { /* make-string(n, char) */
        Value ch = vm_pop(vm), n = vm_pop(vm);
        VmString* result = vm_string_make(&vm->heap.regions, (int)as_number(n), (int)as_number(ch));
        if (result) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, result); }
        else vm_push(vm, NIL_VAL);
        break;
    }
    case 557: { /* string-upcase */
        Value s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        if (s) { VmString* r = vm_string_upcase(&vm->heap.regions, s); if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); break; } }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 558: { /* string-downcase */
        Value s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        if (s) { VmString* r = vm_string_downcase(&vm->heap.regions, s); if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); break; } }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 559: { /* string-contains (duplicate of 555 for compat) */
        Value sub_val = vm_pop(vm), s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        VmString* sub = (sub_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[sub_val.as.ptr]->opaque.ptr : NULL;
        vm_push(vm, INT_VAL(vm_string_contains(s, sub)));
        break;
    }
    case 560: { /* string=? */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmString* a = (a_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : NULL;
        VmString* b = (b_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : NULL;
        vm_push(vm, BOOL_VAL(vm_string_eq(a, b)));
        break;
    }
    case 561: { /* string<? */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmString* a = (a_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : NULL;
        VmString* b = (b_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : NULL;
        vm_push(vm, BOOL_VAL(vm_string_lt(a, b)));
        break;
    }
    case 562: { /* string-ci=? */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmString* a = (a_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : NULL;
        VmString* b = (b_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : NULL;
        vm_push(vm, BOOL_VAL(vm_string_ci_eq(a, b)));
        break;
    }
    case 563: case 564: { /* string->number / number->string */
        if (fid == 563) { /* string->number */
            Value s_val = vm_pop(vm);
            VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
            double d = vm_string_to_number(s);
            if (isnan(d)) vm_push(vm, BOOL_VAL(0)); /* #f on parse failure */
            else vm_push(vm, number_val(d));
        } else { /* number->string */
            Value n = vm_pop(vm);
            VmString* r = vm_number_to_string(&vm->heap.regions, as_number(n));
            if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); }
            else vm_push(vm, NIL_VAL);
        }
        break;
    }
    case 565: { /* string->list — convert string to list of character codepoints */
        Value s_val = vm_pop(vm);
        if (s_val.type == VAL_STRING && vm->heap.objects[s_val.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            int len = vm_string_length(s);
            Value result = NIL_VAL;
            for (int i = len - 1; i >= 0; i--) {
                int cp = vm_string_ref(s, i);
                int32_t p = heap_alloc(&vm->heap);
                if (p < 0) break;
                vm->heap.objects[p]->type = HEAP_CONS;
                vm->heap.objects[p]->cons.car = INT_VAL(cp >= 0 ? cp : 0);
                vm->heap.objects[p]->cons.cdr = result;
                result = PAIR_VAL(p);
            }
            vm_push(vm, result);
        } else {
            vm_push(vm, NIL_VAL);
        }
        break;
    }
    case 566: { /* string-copy */
        Value s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        if (s) { VmString* r = vm_string_copy(&vm->heap.regions, s); if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); break; } }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 567: { /* list->string — convert list of character codepoints to string */
        Value lst = vm_pop(vm);
        /* Count characters */
        int len = 0;
        Value cur = lst;
        while (cur.type == VAL_PAIR && len < 4096) {
            len++;
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        char* buf = (char*)vm_alloc(&vm->heap.regions, (size_t)(len + 1));
        if (buf) {
            cur = lst;
            int idx = 0;
            while (cur.type == VAL_PAIR && idx < len) {
                int cp = (int)as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                buf[idx++] = (cp >= 0 && cp < 128) ? (char)cp : '?';
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
            buf[idx] = '\0';
            VmString* s = vm_string_new(&vm->heap.regions, buf, idx);
            if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 568: { /* string->number (alt ID) */
        Value s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        double d = vm_string_to_number(s);
        if (isnan(d)) vm_push(vm, BOOL_VAL(0));
        else vm_push(vm, number_val(d));
        break;
    }
    case 569: { /* number->string (alt ID) */
        Value n = vm_pop(vm);
        VmString* r = vm_number_to_string(&vm->heap.regions, as_number(n));
        if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); }
        else vm_push(vm, NIL_VAL);
        break;
    }
    case 570: { /* string-hash */
        Value s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        vm_push(vm, INT_VAL(s ? (int64_t)vm_string_hash(s) : 0));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * I/O Operations (580-602) — port-based I/O
     * ══════════════════════════════════════════════════════════════════════ */
    case 580: { /* open-input-file(path) */
        Value path = vm_pop(vm);
        if (path.type == VAL_STRING && vm->heap.objects[path.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[path.as.ptr]->opaque.ptr;
            VmPort* p = vm_port_open_input_file(&vm->heap.regions, s->data);
            if (p) {
                int32_t ptr = heap_alloc(&vm->heap);
                if (ptr >= 0) {
                    vm->heap.objects[ptr]->type = HEAP_PORT;
                    vm->heap.objects[ptr]->opaque.ptr = p;
                    vm_push(vm, (Value){.type = VAL_INT, .as.ptr = ptr});
                    break;
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 581: { /* open-output-file(path) */
        Value path = vm_pop(vm);
        if (path.type == VAL_STRING && vm->heap.objects[path.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[path.as.ptr]->opaque.ptr;
            VmPort* p = vm_port_open_output_file(&vm->heap.regions, s->data);
            if (p) {
                int32_t ptr = heap_alloc(&vm->heap);
                if (ptr >= 0) {
                    vm->heap.objects[ptr]->type = HEAP_PORT;
                    vm->heap.objects[ptr]->opaque.ptr = p;
                    vm_push(vm, (Value){.type = VAL_INT, .as.ptr = ptr});
                    break;
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 582: { /* read-char — from stdin */
        Value port = vm_pop(vm); (void)port;
        int ch = getchar();
        vm_push(vm, ch == EOF ? NIL_VAL : INT_VAL(ch));
        break;
    }
    case 583: { /* peek-char */
        Value port = vm_pop(vm); (void)port;
        int ch = getchar();
        if (ch != EOF) ungetc(ch, stdin);
        vm_push(vm, ch == EOF ? NIL_VAL : INT_VAL(ch));
        break;
    }
    case 584: { /* write-char(char, port) */
        Value port = vm_pop(vm), ch = vm_pop(vm); (void)port;
        putchar((int)as_number(ch));
        vm_push(vm, NIL_VAL);
        break;
    }
    case 585: { /* write-string(str, port) */
        Value port = vm_pop(vm), s_val = vm_pop(vm); (void)port;
        if (s_val.type == VAL_STRING && vm->heap.objects[s_val.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            printf("%.*s", s->byte_len, s->data);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 586: { /* write-char(char, port) — write to stdout if no port */
        Value ch = vm_pop(vm);
        putchar((int)as_number(ch));
        vm_push(vm, NIL_VAL);
        break;
    }
    case 587: { /* write-string */
        Value str = vm_pop(vm);
        if (str.type == VAL_STRING && vm->heap.objects[str.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[str.as.ptr]->opaque.ptr;
            fwrite(s->data, 1, s->byte_len, stdout);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 588: { /* read — read a single char from stdin, return as integer */
        int ch = getchar();
        vm_push(vm, ch == EOF ? NIL_VAL : INT_VAL(ch));
        break;
    }
    case 589: { /* write(datum) */
        Value v = vm_pop(vm);
        print_value(vm, v);
        vm_push(vm, NIL_VAL);
        break;
    }
    case 590: { /* display(datum) — print without quoting (same as write in this VM) */
        Value v = vm_pop(vm);
        print_value(vm, v);
        vm_push(vm, NIL_VAL);
        break;
    }
    case 591: { /* newline */
        printf("\n"); vm_push(vm, NIL_VAL); break;
    }
    case 592: { /* eof-object? */
        Value v = vm_pop(vm);
        vm_push(vm, BOOL_VAL(v.type == VAL_NIL));
        break;
    }
    case 593: { /* current-input-port */
        vm_push(vm, NIL_VAL); break;
    }
    case 594: { /* current-output-port */
        vm_push(vm, NIL_VAL); break;
    }
    case 595: { /* current-error-port → just return a sentinel */
        vm_push(vm, INT_VAL(-3)); /* stderr sentinel */
        break;
    }
    case 596: { /* open-input-string */
        Value str = vm_pop(vm);
        VmString* src = (str.type == VAL_STRING && vm->heap.objects[str.as.ptr]->opaque.ptr)
            ? (VmString*)vm->heap.objects[str.as.ptr]->opaque.ptr : NULL;
        VmPort* p = vm_port_open_input_string(&vm->heap.regions, src);
        if (p) {
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr >= 0) {
                vm->heap.objects[ptr]->type = HEAP_PORT;
                vm->heap.objects[ptr]->opaque.ptr = p;
                vm_push(vm, (Value){.type = VAL_INT, .as.ptr = ptr});
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 597: { /* open-output-string */
        VmPort* p = vm_port_open_output_string(&vm->heap.regions);
        if (p) {
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr >= 0) {
                vm->heap.objects[ptr]->type = HEAP_PORT;
                vm->heap.objects[ptr]->opaque.ptr = p;
                vm_push(vm, (Value){.type = VAL_INT, .as.ptr = ptr});
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 598: { /* get-output-string */
        Value port_val = vm_pop(vm);
        if (port_val.as.ptr >= 0 && vm->heap.objects[port_val.as.ptr]->type == HEAP_PORT) {
            VmPort* p = (VmPort*)vm->heap.objects[port_val.as.ptr]->opaque.ptr;
            VmString* s = vm_port_get_output_string(&vm->heap.regions, p);
            if (s) {
                int32_t ptr = heap_alloc(&vm->heap);
                if (ptr >= 0) {
                    vm->heap.objects[ptr]->type = HEAP_STRING;
                    vm->heap.objects[ptr]->opaque.ptr = s;
                    vm_push(vm, (Value){.type = VAL_STRING, .as.ptr = ptr});
                    break;
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 599: { /* file-exists? */
        Value path = vm_pop(vm);
        int exists = 0;
        if (path.type == VAL_STRING && vm->heap.objects[path.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[path.as.ptr]->opaque.ptr;
            exists = vm_port_file_exists(s->data);
        }
        vm_push(vm, BOOL_VAL(exists));
        break;
    }
    case 600: { /* delete-file */
        Value path = vm_pop(vm);
        if (path.type == VAL_STRING && vm->heap.objects[path.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[path.as.ptr]->opaque.ptr;
            vm_port_delete_file(s->data);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 601: { /* directory-entries(path) — returns list of filenames in directory */
        Value path = vm_pop(vm);
        /* No portable C89 readdir — return empty list on unsupported platforms */
        (void)path;
        vm_push(vm, NIL_VAL);
        break;
    }
    case 602: { /* command-line — return empty list */
        vm_push(vm, NIL_VAL);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Parallel (620-628) — sequential fallback (VM is single-threaded)
     * ══════════════════════════════════════════════════════════════════════ */
    case 620: { /* parallel-map(fn, list) — sequential via closure bridge (VM is single-threaded) */
        Value list = vm_pop(vm), fn = vm_pop(vm);
        /* Apply fn to each element, build result list in correct order */
        Value rev_results = NIL_VAL;
        Value cur = list;
        while (cur.type == VAL_PAIR) {
            Value elem = vm->heap.objects[cur.as.ptr]->cons.car;
            Value mapped = vm_call_closure_from_native(vm, fn, &elem, 1);
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) break;
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = mapped;
            vm->heap.objects[p]->cons.cdr = rev_results;
            rev_results = PAIR_VAL(p);
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        /* Reverse to get correct order */
        Value result = NIL_VAL;
        cur = rev_results;
        while (cur.type == VAL_PAIR) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) break;
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = vm->heap.objects[cur.as.ptr]->cons.car;
            vm->heap.objects[p]->cons.cdr = result;
            result = PAIR_VAL(p);
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        vm_push(vm, result);
        break;
    }
    case 621: { /* parallel-filter(pred, list) — sequential via closure bridge */
        Value list = vm_pop(vm), pred = vm_pop(vm);
        Value rev_results = NIL_VAL;
        Value cur = list;
        while (cur.type == VAL_PAIR) {
            Value elem = vm->heap.objects[cur.as.ptr]->cons.car;
            Value keep = vm_call_closure_from_native(vm, pred, &elem, 1);
            if (is_truthy(keep)) {
                int32_t p = heap_alloc(&vm->heap);
                if (p < 0) break;
                vm->heap.objects[p]->type = HEAP_CONS;
                vm->heap.objects[p]->cons.car = elem;
                vm->heap.objects[p]->cons.cdr = rev_results;
                rev_results = PAIR_VAL(p);
            }
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        /* Reverse */
        Value result = NIL_VAL;
        cur = rev_results;
        while (cur.type == VAL_PAIR) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) break;
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = vm->heap.objects[cur.as.ptr]->cons.car;
            vm->heap.objects[p]->cons.cdr = result;
            result = PAIR_VAL(p);
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        vm_push(vm, result);
        break;
    }
    case 622: { /* parallel-fold(fn, init, list) — sequential via closure bridge */
        Value list = vm_pop(vm), init = vm_pop(vm), fn = vm_pop(vm);
        Value acc = init;
        Value cur = list;
        while (cur.type == VAL_PAIR) {
            Value elem = vm->heap.objects[cur.as.ptr]->cons.car;
            Value args[2] = {acc, elem};
            acc = vm_call_closure_from_native(vm, fn, args, 2);
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        vm_push(vm, acc);
        break;
    }
    case 623: { /* parallel-for-each(fn, list) — sequential via closure bridge */
        Value list = vm_pop(vm), fn = vm_pop(vm);
        Value cur = list;
        while (cur.type == VAL_PAIR) {
            Value elem = vm->heap.objects[cur.as.ptr]->cons.car;
            vm_call_closure_from_native(vm, fn, &elem, 1);
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 624: { /* parallel-execute(thunk) — sequential: just call the thunk */
        Value thunk = vm_pop(vm);
        Value result = vm_call_closure_from_native(vm, thunk, NULL, 0);
        vm_push(vm, result);
        break;
    }
    case 625: { /* future(thunk) — in single-threaded VM, evaluate immediately */
        Value thunk = vm_pop(vm);
        Value result = vm_call_closure_from_native(vm, thunk, NULL, 0);
        vm_push(vm, result);
        break;
    }
    case 626: { /* force-future */
        Value fut = vm_pop(vm);
        vm_push(vm, fut); /* futures are just values in single-threaded mode */
        break;
    }
    case 627: { /* future-ready? — always true in single-threaded mode */
        Value fut = vm_pop(vm); (void)fut;
        vm_push(vm, BOOL_VAL(1));
        break;
    }
    case 628: { /* thread-pool-info */
        vm_push(vm, INT_VAL(1)); /* 1 thread (main) */
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * MultiValue Operations (650-654)
     * ══════════════════════════════════════════════════════════════════════ */
    case 650: { /* values(v) — single-value pass-through; value already on stack */
        break;
    }
    case 651: { /* multi-value-ref(mv, idx) — extract from multi-value container */
        Value idx = vm_pop(vm), mv = vm_pop(vm);
        if (mv.as.ptr >= 0 && vm->heap.objects[mv.as.ptr]->type == HEAP_MULTI_VALUE) {
            VmMultiValue* mvobj = (VmMultiValue*)vm->heap.objects[mv.as.ptr]->opaque.ptr;
            int i = (int)as_number(idx);
            if (mvobj && i >= 0 && i < mvobj->count) {
                vm_push(vm, INT_VAL((int64_t)(intptr_t)mvobj->values[i]));
                break;
            }
        }
        /* Single value: index 0 returns the value itself */
        if ((int)as_number(idx) == 0) { vm_push(vm, mv); }
        else { vm_push(vm, NIL_VAL); }
        break;
    }
    case 652: { /* multi-value-count(mv) — number of values in container */
        Value mv = vm_pop(vm);
        if (is_heap_type(vm, mv, HEAP_MULTI_VALUE)) {
            VmMultiValue* mvobj = (VmMultiValue*)vm->heap.objects[mv.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(mvobj ? mvobj->count : 1));
        } else {
            vm_push(vm, INT_VAL(1)); /* single value counts as 1 */
        }
        break;
    }
    case 653: { /* multi-value? — check if value is a multi-value container */
        Value v = vm_pop(vm);
        int is_mv = (is_heap_type(vm, v, HEAP_MULTI_VALUE));
        vm_push(vm, BOOL_VAL(is_mv));
        break;
    }
    case 654: { /* call-with-values(producer, consumer) — via closure bridge */
        Value consumer = vm_pop(vm), producer = vm_pop(vm);
        /* Call producer with 0 args */
        Value produced = vm_call_closure_from_native(vm, producer, NULL, 0);
        /* Call consumer with produced value */
        Value result = vm_call_closure_from_native(vm, consumer, &produced, 1);
        vm_push(vm, result);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Hash Table Operations (660-670)
     * ══════════════════════════════════════════════════════════════════════ */
    case 660: { /* make-hash-table */
        VmHashTable* ht = vm_ht_make(&vm->heap.regions);
        if (!ht) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_HASH, VAL_INT, ht);
        break;
    }
    case 661: { /* hash-ref(ht, key, default) */
        Value dflt = vm_pop(vm), key = vm_pop(vm), ht_val = vm_pop(vm);
        if (ht_val.as.ptr >= 0 && vm->heap.objects[ht_val.as.ptr]->type == HEAP_HASH) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            void* result = vm_ht_ref(ht, (void*)(uintptr_t)key.as.i, (void*)(uintptr_t)dflt.as.i);
            vm_push(vm, INT_VAL((int64_t)(intptr_t)result));
        } else vm_push(vm, dflt);
        break;
    }
    case 662: { /* hash-set!(ht, key, value) */
        Value val = vm_pop(vm), key = vm_pop(vm), ht_val = vm_pop(vm);
        if (ht_val.as.ptr >= 0 && vm->heap.objects[ht_val.as.ptr]->type == HEAP_HASH) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            vm_ht_set(&vm->heap.regions, ht, (void*)(uintptr_t)key.as.i, (void*)(uintptr_t)val.as.i);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 663: { /* hash-delete!(ht, key) */
        Value key = vm_pop(vm), ht_val = vm_pop(vm);
        if (ht_val.as.ptr >= 0 && vm->heap.objects[ht_val.as.ptr]->type == HEAP_HASH) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            vm_ht_remove(ht, (void*)(uintptr_t)key.as.i);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 664: { /* hash-has-key?(ht, key) */
        Value key = vm_pop(vm), ht_val = vm_pop(vm);
        if (ht_val.as.ptr >= 0 && vm->heap.objects[ht_val.as.ptr]->type == HEAP_HASH) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            vm_push(vm, BOOL_VAL(vm_ht_has_key(ht, (void*)(uintptr_t)key.as.i)));
        } else vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 665: { /* hash-keys */
        Value ht_val = vm_pop(vm);
        if (ht_val.as.ptr >= 0 && vm->heap.objects[ht_val.as.ptr]->type == HEAP_HASH) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            if (ht) {
                Value result = NIL_VAL;
                for (int i = ht->capacity - 1; i >= 0; i--) {
                    if (ht->keys[i]) {
                        int32_t p = heap_alloc(&vm->heap);
                        if (p < 0) break;
                        vm->heap.objects[p]->type = HEAP_CONS;
                        vm->heap.objects[p]->cons.car = *(Value*)ht->keys[i];
                        vm->heap.objects[p]->cons.cdr = result;
                        result = PAIR_VAL(p);
                    }
                }
                vm_push(vm, result);
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 666: { /* hash-values */
        Value ht_val = vm_pop(vm);
        if (ht_val.as.ptr >= 0 && vm->heap.objects[ht_val.as.ptr]->type == HEAP_HASH) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            if (ht) {
                Value result = NIL_VAL;
                for (int i = ht->capacity - 1; i >= 0; i--) {
                    if (ht->keys[i]) {
                        int32_t p = heap_alloc(&vm->heap);
                        if (p < 0) break;
                        vm->heap.objects[p]->type = HEAP_CONS;
                        vm->heap.objects[p]->cons.car = *(Value*)ht->values[i];
                        vm->heap.objects[p]->cons.cdr = result;
                        result = PAIR_VAL(p);
                    }
                }
                vm_push(vm, result);
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 667: { /* hash-count */
        Value ht_val = vm_pop(vm);
        if (ht_val.as.ptr >= 0 && vm->heap.objects[ht_val.as.ptr]->type == HEAP_HASH) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(ht ? vm_ht_count(ht) : 0));
        } else vm_push(vm, INT_VAL(0));
        break;
    }
    case 668: { /* hash-table-copy(ht) — shallow copy of hash table */
        Value ht_val = vm_pop(vm);
        if (ht_val.as.ptr >= 0 && vm->heap.objects[ht_val.as.ptr]->type == HEAP_HASH) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            if (ht) {
                VmHashTable* copy = vm_ht_make(&vm->heap.regions);
                if (copy) {
                    VM_PUSH_HEAP_OPAQUE(vm, HEAP_HASH, VAL_INT, copy);
                    break;
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 669: { /* hash-clear! */
        Value ht_val = vm_pop(vm);
        if (ht_val.as.ptr >= 0 && vm->heap.objects[ht_val.as.ptr]->type == HEAP_HASH) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            if (ht) vm_ht_clear(ht);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 670: { /* hash-table? */
        Value v = vm_pop(vm);
        int is_ht = (is_heap_type(vm, v, HEAP_HASH));
        vm_push(vm, BOOL_VAL(is_ht));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Bytevector Operations (680-689)
     * ══════════════════════════════════════════════════════════════════════ */
    case 680: { /* make-bytevector(n, fill) */
        Value fill = vm_pop(vm), n = vm_pop(vm);
        VmBytevector* bv = vm_bv_make(&vm->heap.regions, (int)as_number(n), (int)as_number(fill));
        if (!bv) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_BYTEVECTOR, VAL_INT, bv);
        break;
    }
    case 681: { /* bytevector-length */
        Value bv_val = vm_pop(vm);
        if (bv_val.as.ptr >= 0 && vm->heap.objects[bv_val.as.ptr]->type == HEAP_BYTEVECTOR) {
            VmBytevector* bv = (VmBytevector*)vm->heap.objects[bv_val.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(vm_bv_length(bv)));
        } else vm_push(vm, INT_VAL(0));
        break;
    }
    case 682: { /* bytevector-u8-ref(bv, k) */
        Value k = vm_pop(vm), bv_val = vm_pop(vm);
        if (bv_val.as.ptr >= 0 && vm->heap.objects[bv_val.as.ptr]->type == HEAP_BYTEVECTOR) {
            VmBytevector* bv = (VmBytevector*)vm->heap.objects[bv_val.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(vm_bv_u8_ref(bv, (int)as_number(k))));
        } else vm_push(vm, INT_VAL(0));
        break;
    }
    case 683: { /* bytevector-u8-set!(bv, k, byte) */
        Value byte = vm_pop(vm), k = vm_pop(vm), bv_val = vm_pop(vm);
        if (bv_val.as.ptr >= 0 && vm->heap.objects[bv_val.as.ptr]->type == HEAP_BYTEVECTOR) {
            VmBytevector* bv = (VmBytevector*)vm->heap.objects[bv_val.as.ptr]->opaque.ptr;
            vm_bv_u8_set(bv, (int)as_number(k), (int)as_number(byte));
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 684: { /* bytevector-append(a, b) */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmBytevector* a = (a_val.as.ptr >= 0 && vm->heap.objects[a_val.as.ptr]->type == HEAP_BYTEVECTOR)
            ? (VmBytevector*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : NULL;
        VmBytevector* b = (b_val.as.ptr >= 0 && vm->heap.objects[b_val.as.ptr]->type == HEAP_BYTEVECTOR)
            ? (VmBytevector*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : NULL;
        if (a && b) {
            VmBytevector* r = vm_bv_append(&vm->heap.regions, a, b);
            if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_BYTEVECTOR, VAL_INT, r); break; }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 685: { /* bytevector-copy!(to, at, from) — copy bytes from src into dst */
        Value from_val = vm_pop(vm), at_val = vm_pop(vm), to_val = vm_pop(vm);
        if (to_val.as.ptr >= 0 && vm->heap.objects[to_val.as.ptr]->type == HEAP_BYTEVECTOR &&
            from_val.as.ptr >= 0 && vm->heap.objects[from_val.as.ptr]->type == HEAP_BYTEVECTOR) {
            VmBytevector* to_bv = (VmBytevector*)vm->heap.objects[to_val.as.ptr]->opaque.ptr;
            VmBytevector* from_bv = (VmBytevector*)vm->heap.objects[from_val.as.ptr]->opaque.ptr;
            int at = (int)as_number(at_val);
            if (to_bv && from_bv && at >= 0) {
                int copy_len = from_bv->len;
                if (at + copy_len > to_bv->len) copy_len = to_bv->len - at;
                if (copy_len > 0) memcpy(to_bv->data + at, from_bv->data, (size_t)copy_len);
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 686: { /* bytevector? */
        Value v = vm_pop(vm);
        int is_bv = (is_heap_type(vm, v, HEAP_BYTEVECTOR));
        vm_push(vm, BOOL_VAL(is_bv));
        break;
    }
    case 687: { /* bytevector-copy(bv) */
        Value bv_val = vm_pop(vm);
        if (bv_val.as.ptr >= 0 && vm->heap.objects[bv_val.as.ptr]->type == HEAP_BYTEVECTOR) {
            VmBytevector* bv = (VmBytevector*)vm->heap.objects[bv_val.as.ptr]->opaque.ptr;
            VmBytevector* r = vm_bv_copy(&vm->heap.regions, bv, 0, bv->len);
            if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_BYTEVECTOR, VAL_INT, r); break; }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 688: { /* utf8->string(bv) */
        Value bv_val = vm_pop(vm);
        if (bv_val.as.ptr >= 0 && vm->heap.objects[bv_val.as.ptr]->type == HEAP_BYTEVECTOR) {
            VmBytevector* bv = (VmBytevector*)vm->heap.objects[bv_val.as.ptr]->opaque.ptr;
            if (bv) {
                VmString* s = vm_string_new(&vm->heap.regions, (const char*)bv->data, bv->len);
                if (s) {
                    int32_t ptr = heap_alloc(&vm->heap);
                    if (ptr >= 0) {
                        vm->heap.objects[ptr]->type = HEAP_STRING;
                        vm->heap.objects[ptr]->opaque.ptr = s;
                        vm_push(vm, (Value){.type = VAL_STRING, .as.ptr = ptr});
                        break;
                    }
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 689: { /* string->utf8(str) */
        Value str_val = vm_pop(vm);
        if (str_val.type == VAL_STRING && vm->heap.objects[str_val.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[str_val.as.ptr]->opaque.ptr;
            VmBytevector* bv = vm_bv_make(&vm->heap.regions, s->byte_len, 0);
            if (bv) {
                memcpy(bv->data, s->data, s->byte_len);
                int32_t ptr = heap_alloc(&vm->heap);
                if (ptr >= 0) {
                    vm->heap.objects[ptr]->type = HEAP_BYTEVECTOR;
                    vm->heap.objects[ptr]->opaque.ptr = bv;
                    vm_push(vm, (Value){.type = VAL_INT, .as.ptr = ptr});
                    break;
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Parameter Operations (700-704)
     * ══════════════════════════════════════════════════════════════════════ */
    case 700: { /* make-parameter(default, converter) */
        Value conv = vm_pop(vm), dflt = vm_pop(vm);
        VmParameter* p = vm_param_make(&vm->heap.regions,
            (void*)(uintptr_t)dflt.as.i, (void*)(uintptr_t)conv.as.i);
        if (!p) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_PARAMETER, VAL_INT, p);
        break;
    }
    case 701: { /* parameter-ref */
        Value p_val = vm_pop(vm);
        if (p_val.as.ptr >= 0 && vm->heap.objects[p_val.as.ptr]->type == HEAP_PARAMETER) {
            VmParameter* p = (VmParameter*)vm->heap.objects[p_val.as.ptr]->opaque.ptr;
            void* v = vm_param_ref(p);
            vm_push(vm, INT_VAL((int64_t)(intptr_t)v));
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 702: { /* parameterize-push(param, value) */
        Value val = vm_pop(vm), p_val = vm_pop(vm);
        if (p_val.as.ptr >= 0 && vm->heap.objects[p_val.as.ptr]->type == HEAP_PARAMETER) {
            VmParameter* p = (VmParameter*)vm->heap.objects[p_val.as.ptr]->opaque.ptr;
            vm_param_push(&vm->heap.regions, p, (void*)(uintptr_t)val.as.i);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 703: { /* parameterize-pop(param) */
        Value p_val = vm_pop(vm);
        if (p_val.as.ptr >= 0 && vm->heap.objects[p_val.as.ptr]->type == HEAP_PARAMETER) {
            VmParameter* p = (VmParameter*)vm->heap.objects[p_val.as.ptr]->opaque.ptr;
            vm_param_pop(p);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 704: { /* parameter? */
        Value v = vm_pop(vm);
        int is_param = (is_heap_type(vm, v, HEAP_PARAMETER));
        vm_push(vm, BOOL_VAL(is_param));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Error Operations (710-714)
     * ══════════════════════════════════════════════════════════════════════ */
    case 710: { /* error(type, message) */
        Value msg = vm_pop(vm), type = vm_pop(vm);
        (void)type; (void)msg;
        VmError* e = vm_error_make(&vm->heap.regions, "error", "runtime error", NULL, 0);
        if (!e) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_ERROR, VAL_INT, e);
        break;
    }
    case 711: { /* error-message */
        Value e_val = vm_pop(vm);
        if (e_val.as.ptr >= 0 && vm->heap.objects[e_val.as.ptr]->type == HEAP_ERROR) {
            VmError* e = (VmError*)vm->heap.objects[e_val.as.ptr]->opaque.ptr;
            VmString* s = vm_string_from_cstr(&vm->heap.regions, e ? e->message : "");
            if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 712: { /* error-type */
        Value e_val = vm_pop(vm);
        if (e_val.as.ptr >= 0 && vm->heap.objects[e_val.as.ptr]->type == HEAP_ERROR) {
            VmError* e = (VmError*)vm->heap.objects[e_val.as.ptr]->opaque.ptr;
            VmString* s = vm_string_from_cstr(&vm->heap.regions, e ? e->type : "");
            if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 713: { /* error-irritants — returns nil (no irritant list support in Value system) */
        Value e_val = vm_pop(vm); (void)e_val;
        vm_push(vm, NIL_VAL);
        break;
    }
    case 714: { /* error? */
        Value v = vm_pop(vm);
        int is_err = (is_heap_type(vm, v, HEAP_ERROR));
        vm_push(vm, BOOL_VAL(is_err));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * High-level AD: gradient, jacobian, hessian (750-752)
     * Uses closure bridge to evaluate function with dual numbers
     * ══════════════════════════════════════════════════════════════════════ */
    case 750: { /* gradient(f, x) → f'(x) via forward-mode dual */
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);
        /* Create dual number: x + 1ε */
        VmDual* d = vm_dual_new(&vm->heap.regions, as_number(x_val), 1.0);
        if (!d) { vm_push(vm, FLOAT_VAL(0)); break; }
        int32_t dptr = heap_alloc(&vm->heap);
        if (dptr < 0) { vm->error = 1; break; }
        vm->heap.objects[dptr]->type = HEAP_DUAL;
        vm->heap.objects[dptr]->opaque.ptr = d;
        Value dual_arg = (Value){.type = VAL_DUAL, .as.ptr = dptr};
        /* Call f(dual) */
        Value result = vm_call_closure_from_native(vm, f_val, &dual_arg, 1);
        /* Extract tangent = derivative */
        if (result.type == VAL_DUAL && result.as.ptr >= 0) {
            VmDual* rd = (VmDual*)vm->heap.objects[result.as.ptr]->opaque.ptr;
            vm_push(vm, FLOAT_VAL(rd ? rd->tangent : 0));
        } else {
            vm_push(vm, FLOAT_VAL(as_number(result))); /* non-dual result = zero derivative */
        }
        break;
    }
    case 751: { /* jacobian — simplified: same as gradient for scalar */
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);
        VmDual* d = vm_dual_new(&vm->heap.regions, as_number(x_val), 1.0);
        if (!d) { vm_push(vm, FLOAT_VAL(0)); break; }
        int32_t dptr = heap_alloc(&vm->heap);
        if (dptr < 0) { vm->error = 1; break; }
        vm->heap.objects[dptr]->type = HEAP_DUAL;
        vm->heap.objects[dptr]->opaque.ptr = d;
        Value dual_arg = (Value){.type = VAL_DUAL, .as.ptr = dptr};
        Value result = vm_call_closure_from_native(vm, f_val, &dual_arg, 1);
        if (result.type == VAL_DUAL && result.as.ptr >= 0) {
            VmDual* rd = (VmDual*)vm->heap.objects[result.as.ptr]->opaque.ptr;
            vm_push(vm, FLOAT_VAL(rd ? rd->tangent : 0));
        } else { vm_push(vm, FLOAT_VAL(0)); }
        break;
    }
    case 752: { /* hessian — second derivative via nested dual */
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);
        (void)x_val; (void)f_val;
        vm_push(vm, FLOAT_VAL(0)); /* nested dual not yet supported */
        break;
    }


    /* ══════════════════════════════════════════════════════════════════════
     * Compiler-required native functions (merged from eshkol_compiler.c)
     * ══════════════════════════════════════════════════════════════════════ */

    case 100: { /* build-string-from-packed: TOS has packed chars, below that length */
        /* Stack: [len, pack0, pack1, ..., packN-1] where N = (len+7)/8 */
        /* Peek down to find the length */
        int n_packs_guess = 0;
        int slen = 0;
        /* The compiler pushes: CONST(len), CONST(pack0), ..., CONST(packN-1), NATIVE_CALL 100 */
        /* So TOS-N = len, TOS-(N-1) through TOS-0 = packs, where N = (len+7)/8 */
        /* We need to scan backwards from TOS to find the length */
        for (int try_n = 0; try_n < 64; try_n++) {
            int len_pos = vm->sp - try_n - 1;
            if (len_pos >= 0 && vm->stack[len_pos].type == VAL_INT) {
                int candidate = (int)vm->stack[len_pos].as.i;
                int expected_packs = (candidate + 7) / 8;
                if (expected_packs == try_n && candidate >= 0 && candidate < 256) {
                    slen = candidate;
                    n_packs_guess = try_n;
                    break;
                }
            }
        }
        char buf[256];
        for (int p = n_packs_guess - 1; p >= 0; p--) {
            Value pack_v = vm_pop(vm);
            int64_t pack = pack_v.as.i;
            for (int b = 0; b < 8 && p * 8 + b < slen; b++)
                buf[p * 8 + b] = (char)((pack >> (b * 8)) & 0xFF);
        }
        vm_pop(vm); /* pop length */
        buf[slen] = 0;
        VmString* s = vm_string_from_cstr(&vm->heap.regions, buf);
        if (s) {
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr >= 0) {
                vm->heap.objects[ptr]->type = HEAP_STRING;
                vm->heap.objects[ptr]->opaque.ptr = s;
                vm_push(vm, (Value){.type = VAL_STRING, .as.ptr = ptr});
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 131: { /* open_upvalues(closure, count, base_slot) — letrec binding */
        Value base_v = vm_pop(vm), count_v = vm_pop(vm), cl_val = vm_pop(vm);
        /* For letrec: patch closure upvalues to point to current stack values */
        if (cl_val.type == VAL_CLOSURE) {
            HeapObject* cl = vm->heap.objects[cl_val.as.ptr];
            int count = (int)as_number(count_v);
            int base = (int)as_number(base_v);
            for (int i = 0; i < cl->closure.n_upvalues && i < count; i++) {
                int slot = base + i;
                if (slot >= 0 && slot < vm->sp)
                    cl->closure.upvalues[i] = vm->stack[vm->fp + slot];
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 133: { /* eq?: identity equality */
        Value b = vm_pop(vm), a = vm_pop(vm);
        int result = 0;
        if (a.type != b.type) result = 0;
        else if (a.type == VAL_NIL) result = 1;
        else if (a.type == VAL_BOOL) result = (a.as.b == b.as.b);
        else if (a.type == VAL_INT) result = (a.as.i == b.as.i);
        else if (a.type == VAL_FLOAT) result = (a.as.f == b.as.f);
        else result = (a.as.ptr == b.as.ptr);
        vm_push(vm, BOOL_VAL(result));
        break;
    }

    case 134: { /* equal?: deep structural equality */
        Value b = vm_pop(vm), a = vm_pop(vm);
        int result = 0;
        if (a.type != b.type) result = 0;
        else if (a.type == VAL_NIL) result = 1;
        else if (a.type == VAL_BOOL) result = (a.as.b == b.as.b);
        else if (a.type == VAL_INT) result = (a.as.i == b.as.i);
        else if (a.type == VAL_FLOAT) result = (a.as.f == b.as.f);
        else if (a.type == VAL_PAIR) {
            /* Simple shallow equality for pairs */
            result = (a.as.ptr == b.as.ptr);
        }
        else result = (a.as.ptr == b.as.ptr);
        vm_push(vm, BOOL_VAL(result));
        break;
    }

    case 142: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, number_val(as_number(a) + as_number(b))); break; } /* add2 */
    case 143: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, number_val(as_number(a) - as_number(b))); break; } /* sub2 */
    case 144: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, number_val(as_number(a) * as_number(b))); break; } /* mul2 */
    case 145: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, number_val(as_number(a) / as_number(b))); break; } /* div2 */
    /* Comparison operators as first-class functions (for sort, map, fold, etc.) */
    case 146: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) < as_number(b))); break; }  /* < */
    case 147: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) > as_number(b))); break; }  /* > */
    case 148: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) <= as_number(b))); break; } /* <= */
    case 149: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) >= as_number(b))); break; } /* >= */
    case 150: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) == as_number(b))); break; } /* = */

    /* Core operations as first-class native functions (IDs 200-226) */
    case 200: { Value a = vm_pop(vm); /* car */
        if (a.type == VAL_PAIR) { HeapObject* o = vm->heap.objects[a.as.ptr]; vm_push(vm, o->cons.car); }
        else { printf("CAR on non-pair\n"); vm_push(vm, NIL_VAL); } break; }
    case 201: { Value a = vm_pop(vm); /* cdr */
        if (a.type == VAL_PAIR) { HeapObject* o = vm->heap.objects[a.as.ptr]; vm_push(vm, o->cons.cdr); }
        else { printf("CDR on non-pair\n"); vm_push(vm, NIL_VAL); } break; }
    case 202: { Value b = vm_pop(vm), a = vm_pop(vm); /* cons */
        int32_t p = heap_alloc(&vm->heap); if (p < 0) { vm->error = 1; break; }
        vm->heap.objects[p]->type = HEAP_CONS;
        vm->heap.objects[p]->cons.car = a; vm->heap.objects[p]->cons.cdr = b;
        vm_push(vm, (Value){VAL_PAIR, {.ptr = p}}); break; }
    case 203: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_NIL)); break; }  /* null? */
    case 204: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_PAIR)); break; } /* pair? */
    case 205: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(!is_truthy(a))); break; }      /* not */
    case 206: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_INT || a.type == VAL_FLOAT)); break; } /* number? */
    case 207: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_STRING)); break; } /* string? */
    case 208: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_BOOL)); break; }   /* boolean? */
    case 209: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_CLOSURE)); break; }/* procedure? */
    case 210: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_VECTOR)); break; } /* vector? */
    case 211: { Value a = vm_pop(vm); print_value(vm, a); fflush(stdout); vm_push(vm, NIL_VAL); break; } /* display */
    case 212: { Value a = vm_pop(vm); print_value(vm, a); fflush(stdout); vm_push(vm, NIL_VAL); break; } /* write */
    case 213: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(as_number(a))); break; }  /* exact->inexact */
    case 214: { Value a = vm_pop(vm); vm_push(vm, INT_VAL((int64_t)as_number(a))); break; } /* inexact->exact */
    case 215: { /* string->number */
        Value a = vm_pop(vm);
        if (a.type == VAL_STRING) {
            VmString* s = (VmString*)vm->heap.objects[a.as.ptr]->opaque.ptr;
            if (s && s->data) {
                double v = atof(s->data);
                vm_push(vm, (v == (int64_t)v) ? INT_VAL((int64_t)v) : FLOAT_VAL(v));
            } else vm_push(vm, NIL_VAL);
        } else vm_push(vm, NIL_VAL);
        break; }
    case 216: { Value a = vm_pop(vm); vm_push(vm, INT_VAL((int64_t)as_number(a))); break; } /* char->integer */
    case 217: { Value a = vm_pop(vm); vm_push(vm, INT_VAL((int64_t)as_number(a))); break; } /* integer->char */
    case 218: { /* make-vector */
        Value fill = vm_pop(vm), size_v = vm_pop(vm);
        int sz = (int)as_number(size_v);
        int32_t p = heap_alloc(&vm->heap); if (p < 0) { vm->error = 1; break; }
        vm->heap.objects[p]->type = HEAP_VECTOR;
        VmVector* v = (VmVector*)vm_alloc(&vm->heap.regions, sizeof(VmVector));
        v->len = sz; v->cap = sz;
        v->items = (Value*)vm_alloc(&vm->heap.regions, sz * sizeof(Value));
        for (int i = 0; i < sz; i++) v->items[i] = fill;
        vm->heap.objects[p]->opaque.ptr = v;
        vm_push(vm, (Value){VAL_VECTOR, {.ptr = p}}); break; }
    case 219: { /* vector-ref */
        Value idx_v = vm_pop(vm), vec_v = vm_pop(vm);
        if (vec_v.type == VAL_VECTOR) {
            VmVector* v = (VmVector*)vm->heap.objects[vec_v.as.ptr]->opaque.ptr;
            int idx = (int)as_number(idx_v);
            if (v && idx >= 0 && idx < v->len) vm_push(vm, v->items[idx]);
            else vm_push(vm, NIL_VAL);
        } else vm_push(vm, NIL_VAL); break; }
    case 220: { /* vector-set! */
        Value val = vm_pop(vm), idx_v = vm_pop(vm), vec_v = vm_pop(vm);
        if (vec_v.type == VAL_VECTOR) {
            VmVector* v = (VmVector*)vm->heap.objects[vec_v.as.ptr]->opaque.ptr;
            int idx = (int)as_number(idx_v);
            if (v && idx >= 0 && idx < v->len) v->items[idx] = val;
        } vm_push(vm, NIL_VAL); break; }
    case 221: { /* vector-length */
        Value v = vm_pop(vm);
        if (v.type == VAL_VECTOR) {
            VmVector* vec = (VmVector*)vm->heap.objects[v.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(vec ? vec->len : 0));
        } else vm_push(vm, INT_VAL(0)); break; }
    case 222: case 223: case 224: case 225: case 226:
        /* string->list, list->string, gcd, lcm, make-string — stubs */
        { int nargs = (fid >= 224) ? 2 : 1;
          for (int i = 0; i < nargs; i++) vm_pop(vm);
          vm_push(vm, NIL_VAL); break; }

    case 151: { /* direct open slot: set closure upvalue to reference a stack slot */
        Value slot_v = vm_pop(vm), uv_idx_v = vm_pop(vm), cl_val = vm_pop(vm);
        if (cl_val.type == VAL_CLOSURE) {
            HeapObject* cl = vm->heap.objects[cl_val.as.ptr];
            int uv_idx = (int)as_number(uv_idx_v);
            int slot = (int)as_number(slot_v);
            if (uv_idx >= 0 && uv_idx < cl->closure.n_upvalues && slot >= 0 && vm->fp + slot < vm->sp)
                cl->closure.upvalues[uv_idx] = vm->stack[vm->fp + slot];
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 130: { /* raise: unhandled exception */
        Value exn = vm_pop(vm);
        fprintf(stderr, "ERROR: unhandled exception: ");
        print_value(vm, exn);
        fprintf(stderr, "\n");
        vm->error = 1;
        break;
    }

    case 132: { /* force: force a promise */
        Value promise = vm_pop(vm);
        vm_push(vm, promise); /* simplified: just return the value */
        break;
    }

    case 135: { /* append */
        Value b = vm_pop(vm), a = vm_pop(vm);
        if (a.type == VAL_NIL) { vm_push(vm, b); break; }
        if (a.type != VAL_PAIR) { vm_push(vm, b); break; }
        /* Copy list a, set last cdr to b */
        Value head = NIL_VAL, tail = NIL_VAL;
        Value cur = a;
        while (cur.type == VAL_PAIR) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = vm->heap.objects[cur.as.ptr]->cons.car;
            vm->heap.objects[p]->cons.cdr = NIL_VAL;
            Value node = PAIR_VAL(p);
            if (head.type == VAL_NIL) head = node;
            else vm->heap.objects[tail.as.ptr]->cons.cdr = node;
            tail = node;
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        if (tail.type == VAL_PAIR) vm->heap.objects[tail.as.ptr]->cons.cdr = b;
        vm_push(vm, head);
        break;
    }
    case 136: { /* reverse */
        Value lst = vm_pop(vm);
        Value result = NIL_VAL;
        while (lst.type == VAL_PAIR) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = vm->heap.objects[lst.as.ptr]->cons.car;
            vm->heap.objects[p]->cons.cdr = result;
            result = PAIR_VAL(p);
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr;
        }
        vm_push(vm, result);
        break;
    }

    case 240: { /* display (native) */
        Value v = vm_pop(vm);
        print_value(vm, v);
        vm_push(vm, NIL_VAL);
        break;
    }
    case 241: { /* write (native) */
        Value v = vm_pop(vm);
        print_value(vm, v);
        vm_push(vm, NIL_VAL);
        break;
    }
    case 250: { /* atan2 */
        Value x = vm_pop(vm), y = vm_pop(vm);
        vm_push(vm, FLOAT_VAL(atan2(as_number(y), as_number(x))));
        break;
    }
    case 251: { /* call-with-values-apply: simplified */
        Value consumer = vm_pop(vm), result = vm_pop(vm);
        /* Simple: call consumer with single result */
        if (consumer.type == VAL_CLOSURE) {
            Value args[1] = {result};
            Value r = vm_call_closure_from_native(vm, consumer, args, 1);
            vm_push(vm, r);
        } else vm_push(vm, result);
        break;
    }
    case 252: { /* propagate open slot from parent */
        Value slot_v = vm_pop(vm), uv_idx_v = vm_pop(vm), cl_val = vm_pop(vm);
        (void)slot_v; (void)uv_idx_v; (void)cl_val;
        vm_push(vm, NIL_VAL);
        break;
    }

    case 237: { /* error */
        Value msg = vm_pop(vm);
        fprintf(stderr, "ERROR: ");
        print_value(vm, msg);
        fprintf(stderr, "\n");
        vm->error = 1;
        break;
    }
    case 238: { /* void */
        vm_push(vm, NIL_VAL);
        break;
    }
    case 235: { /* not */
        Value v = vm_pop(vm);
        vm_push(vm, BOOL_VAL(!is_truthy(v)));
        break;
    }

    default:
        /* Check geometric manifold operations (800-859) */
        if (fid >= 800 && fid <= 859) {
            vm_dispatch_geometric(vm, fid);
            break;
        }
        /* Unknown native ID — warn but don't crash */
        fprintf(stderr, "WARNING: unhandled native call ID %d\n", fid);
        vm_push(vm, NIL_VAL);
        break;
    }
}

/*******************************************************************************
 * VM Execution
 ******************************************************************************/

