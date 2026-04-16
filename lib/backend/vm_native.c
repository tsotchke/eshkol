/* Heap pointer validation — prevent OOB heap access from untrusted values.
 * Used before any vm->heap.objects[val.as.ptr] dereference. */
#define VM_VALIDATE_HEAP(vm, val) \
    (is_valid_heap_ptr((vm), (val).as.ptr) ? \
     (vm)->heap.objects[(val).as.ptr] : \
     (fprintf(stderr, "HEAP ACCESS: invalid ptr %d (max %d)\n", \
              (val).as.ptr, (vm)->heap.next_free), \
      (vm)->error = 1, (HeapObject*)NULL))

static void vm_dispatch_native(VM* vm, int fid) {
    switch (fid) {
    /* ══════════════════════════════════════════════════════════════════════
     * Math functions (20-35)
     * ══════════════════════════════════════════════════════════════════════ */
    case 20: { Value a = vm_pop(vm); if (a.type==VAL_DUAL) { vm_push(vm,a); vm_dispatch_native(vm,377); } else vm_push(vm, FLOAT_VAL(sin(as_number(a)))); break; }
    case 21: { Value a = vm_pop(vm); if (a.type==VAL_DUAL) { vm_push(vm,a); vm_dispatch_native(vm,378); } else vm_push(vm, FLOAT_VAL(cos(as_number(a)))); break; }
    case 22: { Value a = vm_pop(vm); if (a.type==VAL_DUAL) { /* tan = sin/cos */ vm_push(vm,a); vm_dispatch_native(vm,377); Value s=vm_pop(vm); vm_push(vm,a); vm_dispatch_native(vm,378); Value c=vm_pop(vm); vm_push(vm,s); vm_push(vm,c); vm_dispatch_native(vm,376); } else vm_push(vm, FLOAT_VAL(tan(as_number(a)))); break; }
    case 23: { Value a = vm_pop(vm); if (a.type==VAL_DUAL) { vm_push(vm,a); vm_dispatch_native(vm,379); } else vm_push(vm, FLOAT_VAL(exp(as_number(a)))); break; }
    case 24: { Value a = vm_pop(vm); if (a.type==VAL_DUAL) { vm_push(vm,a); vm_dispatch_native(vm,380); } else vm_push(vm, FLOAT_VAL(log(as_number(a)))); break; }
    case 25: { Value a = vm_pop(vm); if (a.type==VAL_DUAL) { vm_push(vm,a); vm_dispatch_native(vm,381); } else vm_push(vm, FLOAT_VAL(sqrt(as_number(a)))); break; }
    case 26: { Value a = vm_pop(vm); vm_push(vm, number_val(floor(as_number_vm(vm,a)))); break; }
    case 27: { Value a = vm_pop(vm); vm_push(vm, number_val(ceil(as_number_vm(vm,a)))); break; }
    case 28: { Value a = vm_pop(vm); vm_push(vm, number_val(round(as_number_vm(vm,a)))); break; }
    case 29: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(asin(as_number_vm(vm,a)))); break; }
    case 30: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(acos(as_number_vm(vm,a)))); break; }
    case 31: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(atan(as_number_vm(vm,a)))); break; }
    case 32: { Value b = vm_pop(vm); Value a = vm_pop(vm);
        if (a.type==VAL_DUAL||b.type==VAL_DUAL) { vm_push(vm,a); vm_push(vm,b); vm_dispatch_native(vm,385); }
        else vm_push(vm, FLOAT_VAL(pow(as_number(a), as_number(b)))); break; }
    case 33: { Value b = vm_pop(vm); Value a = vm_pop(vm); double da=as_number_vm(vm,a),db=as_number_vm(vm,b); vm_push(vm, number_val(da<db?da:db)); break; }
    case 34: { Value b = vm_pop(vm); Value a = vm_pop(vm); double da=as_number_vm(vm,a),db=as_number_vm(vm,b); vm_push(vm, number_val(da>db?da:db)); break; }
    case 35: { Value a = vm_pop(vm); if (a.type==VAL_DUAL) { vm_push(vm,a); vm_dispatch_native(vm,383); } else vm_push(vm, number_val(fabs(as_number(a)))); break; }
    /* modulo, remainder, quotient — first-class closure versions */
    case 36: { Value b = vm_pop(vm); Value a = vm_pop(vm);
        int64_t ia=(int64_t)as_number(a), ib=(int64_t)as_number(b);
        if (ib==0){vm->error=1;break;}
        int64_t r=ia%ib; if(r!=0&&((r^ib)<0)) r+=ib;
        vm_push(vm, INT_VAL(r)); break; }
    case 37: { Value b = vm_pop(vm); Value a = vm_pop(vm);
        int64_t ia=(int64_t)as_number(a), ib=(int64_t)as_number(b);
        if (ib==0){vm->error=1;break;}
        vm_push(vm, INT_VAL(ia%ib)); break; }
    case 38: { Value b = vm_pop(vm); Value a = vm_pop(vm);
        int64_t ia=(int64_t)as_number(a), ib=(int64_t)as_number(b);
        if (ib==0){vm->error=1;break;}
        vm_push(vm, INT_VAL(ia/ib)); break; }

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
    case 51: { /* number->string (n, radix) — radix defaults to 10 via prelude wrapper */
        Value radix_val = vm_pop(vm);
        Value a = vm_pop(vm);
        int radix = (radix_val.type == VAL_INT) ? (int)radix_val.as.i : 10;
        char buf[128];
        if (radix == 10 || radix <= 1 || radix > 16) {
            if (a.type == VAL_INT) snprintf(buf, sizeof(buf), "%lld", (long long)a.as.i);
            else snprintf(buf, sizeof(buf), "%.15g", as_number(a));
        } else {
            int64_t n = (a.type == VAL_INT) ? a.as.i : (int64_t)as_number(a);
            if (n == 0) { buf[0] = '0'; buf[1] = '\0'; }
            else {
                static const char digits[] = "0123456789abcdef";
                char tmp[128]; int pos = 0, neg = (n < 0);
                uint64_t un = neg ? (uint64_t)(-(n + 1)) + 1 : (uint64_t)n;
                while (un > 0) { tmp[pos++] = digits[un % (uint64_t)radix]; un /= (uint64_t)radix; }
                if (neg) tmp[pos++] = '-';
                for (int i = 0; i < pos; i++) buf[i] = tmp[pos - 1 - i];
                buf[pos] = '\0';
            }
        }
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
    case 55: { Value b = vm_pop(vm); Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) == as_number(b))); break; }

    /* ══════════════════════════════════════════════════════════════════════
     * I/O (60-61)
     * ══════════════════════════════════════════════════════════════════════ */
    case 60: printf("\n"); fflush(stdout); vm_push(vm, (Value){.type = VAL_VOID}); break;
    case 61: { Value v = vm_pop(vm); print_value(vm, v); fflush(stdout); vm_push(vm, (Value){.type = VAL_VOID}); break; }

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
        while (lst.type == VAL_PAIR) { len++; if (len > 1000000) { vm->error = 1; break; } lst = vm->heap.objects[lst.as.ptr]->cons.cdr; }
        if (vm->error) break;
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
     * List search: member (137), assoc (138)
     * ══════════════════════════════════════════════════════════════════════ */
    case 137: { /* member: (member obj list) — returns sublist starting from obj, or #f */
        Value lst = vm_pop(vm), obj = vm_pop(vm);
        while (lst.type == VAL_PAIR) {
            Value car = vm->heap.objects[lst.as.ptr]->cons.car;
            if (car.type == obj.type && ((car.type == VAL_INT && car.as.i == obj.as.i) ||
                (car.type == VAL_FLOAT && car.as.f == obj.as.f) ||
                (car.type == VAL_BOOL && car.as.b == obj.as.b) ||
                (car.type == VAL_STRING && obj.type == VAL_STRING &&
                 vm->heap.objects[car.as.ptr]->opaque.ptr && vm->heap.objects[obj.as.ptr]->opaque.ptr &&
                 strcmp(((VmString*)vm->heap.objects[car.as.ptr]->opaque.ptr)->data,
                        ((VmString*)vm->heap.objects[obj.as.ptr]->opaque.ptr)->data) == 0))) {
                vm_push(vm, lst); break;
            }
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr;
        }
        if (lst.type != VAL_PAIR) vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 138: { /* assoc: (assoc key alist) — returns (key . val) pair, or #f */
        Value alist = vm_pop(vm), key = vm_pop(vm);
        int found = 0;
        while (alist.type == VAL_PAIR) {
            Value pair = vm->heap.objects[alist.as.ptr]->cons.car;
            if (pair.type == VAL_PAIR) {
                Value car = vm->heap.objects[pair.as.ptr]->cons.car;
                if (car.type == key.type && ((car.type == VAL_INT && car.as.i == key.as.i) ||
                    (car.type == VAL_FLOAT && car.as.f == key.as.f) ||
                    (car.type == VAL_BOOL && car.as.b == key.as.b))) {
                    vm_push(vm, pair); found = 1; break;
                }
            }
            alist = vm->heap.objects[alist.as.ptr]->cons.cdr;
        }
        if (!found) vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 139: { /* memq: (memq obj list) — identity-based search (eq?) */
        Value lst = vm_pop(vm), obj = vm_pop(vm);
        while (lst.type == VAL_PAIR) {
            Value car = vm->heap.objects[lst.as.ptr]->cons.car;
            /* eq?: same type + same value/pointer */
            if (car.type == obj.type && car.as.i == obj.as.i) {
                vm_push(vm, lst); break;
            }
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr;
        }
        if (lst.type != VAL_PAIR) vm_push(vm, BOOL_VAL(0));
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
            if (!result) { vm_push(vm, NIL_VAL); break; }
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
            if (!a_bn || !b_bn) { vm_push(vm, NIL_VAL); break; }
            vm_push(vm, INT_VAL(bignum_compare(a_bn, b_bn))); break; }
        case 364: { Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmBignum* a_bn = (a_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(a_val));
            VmBignum* b_bn = (b_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(b_val));
            if (!a_bn || !b_bn) { vm_push(vm, NIL_VAL); break; }
            VmBignum* result = bignum_gcd(bn_rs, a_bn, b_bn);
            if (!result) { vm_push(vm, NIL_VAL); break; }
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
     * Hyper-Dual Numbers (1900-1921) — exact second derivatives
     * ══════════════════════════════════════════════════════════════════════ */
    case 1900: case 1901: case 1902: case 1903: case 1904:
    case 1905: case 1906: case 1907: case 1908: case 1909:
    case 1910: case 1911: case 1912: case 1913: case 1914:
    case 1915: case 1916: case 1917: case 1918: case 1919:
    case 1920: case 1921: {
        VmRegionStack* hd_rs = &vm->heap.regions;
        switch (fid) {
        case 1900: { /* make-hyper-dual(f, f1, f2, f12) */
            Value f12v = vm_pop(vm), f2v = vm_pop(vm), f1v = vm_pop(vm), fv = vm_pop(vm);
            VmHyperDual* h = vm_hd_make(hd_rs, as_number(fv), as_number(f1v), as_number(f2v), as_number(f12v));
            if (!h) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_HYPER_DUAL, VAL_HYPER_DUAL, h); break; }
        case 1901: { Value v = vm_pop(vm); /* f component */
            if (v.type == VAL_HYPER_DUAL) { VmHyperDual* h = (VmHyperDual*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, FLOAT_VAL(h->f)); }
            else vm_push(vm, FLOAT_VAL(as_number(v))); break; }
        case 1902: { Value v = vm_pop(vm); /* f1 component */
            if (v.type == VAL_HYPER_DUAL) { VmHyperDual* h = (VmHyperDual*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, FLOAT_VAL(h->f1)); }
            else vm_push(vm, FLOAT_VAL(0.0)); break; }
        case 1903: { Value v = vm_pop(vm); /* f2 component */
            if (v.type == VAL_HYPER_DUAL) { VmHyperDual* h = (VmHyperDual*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, FLOAT_VAL(h->f2)); }
            else vm_push(vm, FLOAT_VAL(0.0)); break; }
        case 1904: { Value v = vm_pop(vm); /* f12 — the second derivative */
            if (v.type == VAL_HYPER_DUAL) { VmHyperDual* h = (VmHyperDual*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, FLOAT_VAL(h->f12)); }
            else vm_push(vm, FLOAT_VAL(0.0)); break; }
        case 1905: case 1906: case 1907: case 1908: { /* binary: add/sub/mul/div */
            Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmHyperDual a_h = {as_number(a_val), 0, 0, 0}, b_h = {as_number(b_val), 0, 0, 0};
            if (a_val.type == VAL_HYPER_DUAL) a_h = *(VmHyperDual*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_HYPER_DUAL) b_h = *(VmHyperDual*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            VmHyperDual* result = NULL;
            switch (fid) {
                case 1905: result = vm_hd_add(hd_rs, &a_h, &b_h); break;
                case 1906: result = vm_hd_sub(hd_rs, &a_h, &b_h); break;
                case 1907: result = vm_hd_mul(hd_rs, &a_h, &b_h); break;
                case 1908: result = vm_hd_div(hd_rs, &a_h, &b_h); break;
            }
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_HYPER_DUAL, VAL_HYPER_DUAL, result); break; }
        case 1909: case 1910: case 1911: case 1912: case 1913:
        case 1914: case 1916: case 1917: case 1918: case 1919: { /* unary */
            Value v = vm_pop(vm);
            VmHyperDual a_h = {as_number(v), 0, 0, 0};
            if (v.type == VAL_HYPER_DUAL) a_h = *(VmHyperDual*)vm->heap.objects[v.as.ptr]->opaque.ptr;
            VmHyperDual* result = NULL;
            switch (fid) {
                case 1909: result = vm_hd_neg(hd_rs, &a_h); break;
                case 1910: result = vm_hd_sin(hd_rs, &a_h); break;
                case 1911: result = vm_hd_cos(hd_rs, &a_h); break;
                case 1912: result = vm_hd_exp(hd_rs, &a_h); break;
                case 1913: result = vm_hd_log(hd_rs, &a_h); break;
                case 1914: result = vm_hd_sqrt(hd_rs, &a_h); break;
                case 1916: result = vm_hd_abs(hd_rs, &a_h); break;
                case 1917: result = vm_hd_relu(hd_rs, &a_h); break;
                case 1918: result = vm_hd_sigmoid(hd_rs, &a_h); break;
                case 1919: result = vm_hd_tanh(hd_rs, &a_h); break;
            }
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_HYPER_DUAL, VAL_HYPER_DUAL, result); break; }
        case 1915: { /* pow(base, exp) */
            Value exp_val = vm_pop(vm), base_val = vm_pop(vm);
            VmHyperDual a_h = {as_number(base_val), 0, 0, 0};
            if (base_val.type == VAL_HYPER_DUAL) a_h = *(VmHyperDual*)vm->heap.objects[base_val.as.ptr]->opaque.ptr;
            VmHyperDual* result = vm_hd_pow(hd_rs, &a_h, as_number(exp_val));
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_HYPER_DUAL, VAL_HYPER_DUAL, result); break; }
        case 1920: { /* from-double */
            Value v = vm_pop(vm);
            VmHyperDual* h = vm_hd_from_double(hd_rs, as_number(v));
            if (!h) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_HYPER_DUAL, VAL_HYPER_DUAL, h); break; }
        case 1921: { /* scale(scalar, hd) */
            Value hd_val = vm_pop(vm), scalar_val = vm_pop(vm);
            VmHyperDual a_h = {as_number(hd_val), 0, 0, 0};
            if (hd_val.type == VAL_HYPER_DUAL) a_h = *(VmHyperDual*)vm->heap.objects[hd_val.as.ptr]->opaque.ptr;
            VmHyperDual* result = vm_hd_scale(hd_rs, as_number(scalar_val), &a_h);
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_HYPER_DUAL, VAL_HYPER_DUAL, result); break; }
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
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_AD_TAPE, VAL_AD_TAPE, tape);
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
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);
        /* Create dual number: x + 1ε */
        VmDual* d = vm_dual_make(&vm->heap.regions, as_number(x_val), 1.0);
        if (!d) { vm_push(vm, FLOAT_VAL(0)); break; }
        int32_t dptr = heap_alloc(&vm->heap);
        if (dptr < 0) { vm->error = 1; break; }
        vm->heap.objects[dptr]->type = HEAP_DUAL;
        vm->heap.objects[dptr]->opaque.ptr = d;
        Value dual_arg = (Value){.type = VAL_DUAL, .as.ptr = dptr};
        /* Call f(dual) via closure bridge */
        Value result = vm_call_closure_from_native(vm, f_val, &dual_arg, 1);
        /* Extract tangent = derivative */
        if (result.type == VAL_DUAL && result.as.ptr >= 0) {
            VmDual* rd = (VmDual*)vm->heap.objects[result.as.ptr]->opaque.ptr;
            vm_push(vm, FLOAT_VAL(rd ? rd->tangent : 0));
        } else {
            vm_push(vm, FLOAT_VAL(0)); /* non-dual result = constant function */
        }
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
        /* Scalar fallback: tensors now have dedicated VAL_TENSOR type.
         * Plain VAL_INT and VAL_FLOAT are genuine scalars. */
        int is_tensor = (t_val.type == VAL_TENSOR &&
                         t_val.as.ptr >= 0 &&
                         t_val.as.ptr < vm->heap.capacity &&
                         vm->heap.objects[t_val.as.ptr]);
        if (!is_tensor && (t_val.type == VAL_INT || t_val.type == VAL_FLOAT)) {
            double x = as_number(t_val);
            double r;
            switch (fid) {
                case 462: r = x > 0 ? x : 0; break;            /* relu */
                case 464: r = 1.0 / (1.0 + exp(-x)); break;    /* sigmoid */
                case 465: r = x > 0 ? x : 0.01 * x; break;     /* leaky_relu */
                case 463: case 466: case 467: case 468:
                default:  r = x; break;
            }
            vm_push(vm, (Value){.type = VAL_FLOAT, .as.f = r});
            break;
        }
        if (!is_tensor) { vm_push(vm, NIL_VAL); break; }
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
     * Model I/O (800-803)
     * ══════════════════════════════════════════════════════════════════════ */
    case 800: { /* model-save(path, entries) */
        Value entries = vm_pop(vm), path = vm_pop(vm);
        vm_push(vm, BOOL_VAL(vm_model_save_model_file(vm, path, entries)));
        break;
    }
    case 801: { /* model-load(path) */
        vm_model_model_load(vm);
        break;
    }
    case 802: { /* tensor-save(path, tensor) */
        Value tensor = vm_pop(vm), path = vm_pop(vm);
        vm_push(vm, BOOL_VAL(vm_model_save_tensor_file(vm, path, tensor)));
        break;
    }
    case 803: { /* tensor-load(path) */
        vm_model_tensor_load(vm);
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
                    VM_PUSH_HEAP_OPAQUE(vm, HEAP_SUBST, VAL_SUBST, extended);
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
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_SUBST, VAL_SUBST, s);
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
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_KB, VAL_KB, kb);
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
    case 507: { /* make-fact(datum) — store a list/value as a fact */
        Value datum = vm_pop(vm);
        int32_t ptr = heap_alloc(&vm->heap);
        if (ptr < 0) { vm->error = 1; break; }
        vm->heap.objects[ptr]->type = HEAP_FACT;
        vm->heap.objects[ptr]->cons.car = datum;
        vm->heap.objects[ptr]->cons.cdr = NIL_VAL;
        vm->heap.objects[ptr]->opaque.ptr = NULL;
        vm_push(vm, (Value){.type = VAL_PAIR, .as.ptr = ptr});
        break;
    }
    case 508: { /* fact? */
        Value v = vm_pop(vm);
        int is_fact = (is_heap_type(vm, v, HEAP_FACT));
        vm_push(vm, BOOL_VAL(is_fact));
        break;
    }
    case 511: { /* kb-assert!(kb, fact) — store fact in the KB */
        Value fact_val = vm_pop(vm), kb_val = vm_pop(vm);
        if (kb_val.as.ptr >= 0 && vm->heap.objects[kb_val.as.ptr]->type == HEAP_KB) {
            VmKnowledgeBase* kb_obj = (VmKnowledgeBase*)vm->heap.objects[kb_val.as.ptr]->opaque.ptr;
            if (kb_obj && kb_obj->n_facts < kb_obj->capacity) {
                VmFact* f = (VmFact*)vm_alloc(&vm->heap.regions, sizeof(VmFact));
                if (f) {
                    memset(f, 0, sizeof(VmFact));
                    /* Store the raw Scheme value as datum_ptr */
                    if (fact_val.as.ptr >= 0 && vm->heap.objects[fact_val.as.ptr]->type == HEAP_FACT) {
                        f->has_datum = 1;
                        f->datum_ptr = vm->heap.objects[fact_val.as.ptr]->cons.car.as.ptr;
                    } else if (fact_val.type == VAL_PAIR) {
                        f->has_datum = 1;
                        f->datum_ptr = fact_val.as.ptr;
                    }
                    kb_obj->facts[kb_obj->n_facts++] = f;
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 512: { /* kb-query(kb, pattern) → list of matching facts */
        Value pattern = vm_pop(vm), kb_val = vm_pop(vm);
        if (kb_val.as.ptr >= 0 && vm->heap.objects[kb_val.as.ptr]->type == HEAP_KB) {
            VmKnowledgeBase* kb_obj = (VmKnowledgeBase*)vm->heap.objects[kb_val.as.ptr]->opaque.ptr;
            if (kb_obj) {
                Value result = NIL_VAL;
                for (int i = kb_obj->n_facts - 1; i >= 0; i--) {
                    VmFact* f = kb_obj->facts[i];
                    if (!f || !f->has_datum) continue;
                    Value fact_datum = PAIR_VAL(f->datum_ptr);
                    int matches = 1;
                    if (pattern.type == VAL_PAIR && fact_datum.type == VAL_PAIR) {
                        Value pc = pattern, fc = fact_datum;
                        while (pc.type == VAL_PAIR && fc.type == VAL_PAIR) {
                            Value pe = vm->heap.objects[pc.as.ptr]->cons.car;
                            Value fe = vm->heap.objects[fc.as.ptr]->cons.car;
                            /* Check if pattern element is a logic variable (?x) */
                            int is_var = 0;
                            if (pe.type == VAL_STRING && pe.as.ptr >= 0) {
                                VmString* ps = (VmString*)vm->heap.objects[pe.as.ptr]->opaque.ptr;
                                if (ps && ps->byte_len > 0 && ps->data[0] == '?') is_var = 1;
                            }
                            if (!is_var) {
                                /* Must match exactly */
                                if (pe.type != fe.type) { matches = 0; break; }
                                if (pe.type == VAL_INT && pe.as.i != fe.as.i) { matches = 0; break; }
                                if (pe.type == VAL_STRING && pe.as.ptr >= 0 && fe.as.ptr >= 0) {
                                    VmString* a = (VmString*)vm->heap.objects[pe.as.ptr]->opaque.ptr;
                                    VmString* b = (VmString*)vm->heap.objects[fe.as.ptr]->opaque.ptr;
                                    if (a && b && (a->byte_len != b->byte_len || memcmp(a->data, b->data, a->byte_len) != 0))
                                        { matches = 0; break; }
                                }
                            }
                            pc = vm->heap.objects[pc.as.ptr]->cons.cdr;
                            fc = vm->heap.objects[fc.as.ptr]->cons.cdr;
                        }
                    } else if (pattern.type == VAL_NIL) {
                        matches = 1; /* empty pattern matches everything */
                    } else {
                        matches = 1; /* non-list pattern: return all */
                    }
                    if (matches) {
                        int32_t p = heap_alloc(&vm->heap);
                        if (p < 0) break;
                        vm->heap.objects[p]->type = HEAP_CONS;
                        vm->heap.objects[p]->cons.car = fact_datum;
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
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_FACTOR_GRAPH, VAL_FACTOR_GRAPH, fg);
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
                     * Allocated via VM arena — lives as long as the factor graph. */
                    if (!fg->observed) {
                        fg->observed = (bool*)vm_alloc(&vm->heap.regions, fg->num_vars * sizeof(bool));
                        if (fg->observed) memset(fg->observed, 0, fg->num_vars * sizeof(bool));
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
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_WORKSPACE, VAL_WORKSPACE, ws);
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
                Value content_val = (Value){.type = VAL_TENSOR, .as.ptr = tptr};

                /* Call each module's closure, collect salience + proposal */
                int n_mod = ws->n_modules;
                if (n_mod > 256) n_mod = 256;
                double* saliences = (double*)vm_alloc(&vm->heap.regions, n_mod * sizeof(double));
                Value* proposals = (Value*)vm_alloc(&vm->heap.regions, n_mod * sizeof(Value));
                if (!saliences || !proposals) { vm_push(vm, NIL_VAL); break; }
                memset(saliences, 0, n_mod * sizeof(double));
                memset(proposals, 0, n_mod * sizeof(Value));
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
                        vm_push(vm, (Value){.type = VAL_TENSOR, .as.ptr = ptr});
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
                    vm_push(vm, (Value){.type = VAL_PORT, .as.ptr = ptr});
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
                    vm_push(vm, (Value){.type = VAL_PORT, .as.ptr = ptr});
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
        fflush(stdout);
        vm_push(vm, (Value){.type = VAL_VOID});
        break;
    }
    case 587: { /* write-string */
        Value str = vm_pop(vm);
        if (str.type == VAL_STRING && vm->heap.objects[str.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[str.as.ptr]->opaque.ptr;
            fwrite(s->data, 1, s->byte_len, stdout);
            fflush(stdout);
        }
        vm_push(vm, (Value){.type = VAL_VOID});
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
                vm_push(vm, (Value){.type = VAL_PORT, .as.ptr = ptr});
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
                vm_push(vm, (Value){.type = VAL_PORT, .as.ptr = ptr});
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
    case 601: { /* directory-entries(path) → list of filename strings */
        Value path_val = vm_pop(vm);
#ifdef ESHKOL_VM_WASM
        vm_push(vm, NIL_VAL); break;
#else
        if (path_val.type != VAL_STRING) { vm_push(vm, NIL_VAL); break; }
        VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
        if (!ps) { vm_push(vm, NIL_VAL); break; }
        DIR* dir = opendir(ps->data);
        if (!dir) { vm_push(vm, NIL_VAL); break; }
        Value result601 = NIL_VAL;
        struct dirent* ent;
        while ((ent = readdir(dir)) != NULL) {
            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
            VmString* s = vm_string_from_cstr(&vm->heap.regions, ent->d_name);
            if (!s) continue;
            int32_t sp601 = heap_alloc(&vm->heap); if (sp601 < 0) continue;
            vm->heap.objects[sp601]->type = HEAP_STRING;
            vm->heap.objects[sp601]->opaque.ptr = s;
            int32_t cp601 = heap_alloc(&vm->heap); if (cp601 < 0) continue;
            vm->heap.objects[cp601]->type = HEAP_CONS;
            vm->heap.objects[cp601]->cons.car = (Value){.type = VAL_STRING, .as.ptr = sp601};
            vm->heap.objects[cp601]->cons.cdr = result601;
            result601 = PAIR_VAL(cp601);
        }
        closedir(dir);
        /* reverse to get alphabetical order */
        Value rev601 = NIL_VAL;
        while (result601.type == VAL_PAIR) {
            Value car601 = vm->heap.objects[result601.as.ptr]->cons.car;
            int32_t rp601 = heap_alloc(&vm->heap); if (rp601 < 0) break;
            vm->heap.objects[rp601]->type = HEAP_CONS;
            vm->heap.objects[rp601]->cons.car = car601;
            vm->heap.objects[rp601]->cons.cdr = rev601;
            rev601 = PAIR_VAL(rp601);
            result601 = vm->heap.objects[result601.as.ptr]->cons.cdr;
        }
        vm_push(vm, rev601);
        break;
#endif
    }
    case 602: { /* command-line → list of argv strings */
        Value result602 = NIL_VAL;
        for (int i = g_vm_argc - 1; i >= 0; i--) {
            const char* arg = (g_vm_argv && g_vm_argv[i]) ? g_vm_argv[i] : "";
            VmString* s = vm_string_from_cstr(&vm->heap.regions, arg);
            if (!s) continue;
            int32_t sp602 = heap_alloc(&vm->heap); if (sp602 < 0) continue;
            vm->heap.objects[sp602]->type = HEAP_STRING;
            vm->heap.objects[sp602]->opaque.ptr = s;
            int32_t cp602 = heap_alloc(&vm->heap); if (cp602 < 0) continue;
            vm->heap.objects[cp602]->type = HEAP_CONS;
            vm->heap.objects[cp602]->cons.car = (Value){.type = VAL_STRING, .as.ptr = sp602};
            vm->heap.objects[cp602]->cons.cdr = result602;
            result602 = PAIR_VAL(cp602);
        }
        vm_push(vm, result602);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Parallel (620-628) — sequential fallback (VM is single-threaded)
     * ══════════════════════════════════════════════════════════════════════ */
    case 620: { /* parallel-map(fn, list) — parallel via thread pool when available */
        Value list = vm_pop(vm), fn = vm_pop(vm);

        /* Count elements */
        int n = 0;
        Value cur = list;
        while (cur.type == VAL_PAIR && n < 4096) {
            n++;
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }

        if (n == 0) { vm_push(vm, NIL_VAL); break; }

        /* Extract elements into array */
        Value* elems = (Value*)vm_alloc(&vm->heap.regions, (size_t)n * sizeof(Value));
        if (!elems) { vm_push(vm, NIL_VAL); break; }
        cur = list;
        for (int i = 0; i < n; i++) {
            elems[i] = vm->heap.objects[cur.as.ptr]->cons.car;
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }

        /* Allocate results array */
        Value* results = (Value*)vm_alloc(&vm->heap.regions, (size_t)n * sizeof(Value));
        if (!results) { vm_push(vm, NIL_VAL); break; }

        /* Use thread pool if available and list is large enough */
        if (g_pool && n >= 4) {
            VmParMapTask* tasks = (VmParMapTask*)vm_alloc(&vm->heap.regions,
                                      (size_t)n * sizeof(VmParMapTask));
            if (tasks) {
                for (int i = 0; i < n; i++) {
                    tasks[i].main_vm = vm;
                    tasks[i].closure = fn;
                    tasks[i].input = elems[i];
                    tasks[i].output = NIL_VAL;
                }
                for (int i = 0; i < n; i++) {
                    vm_pool_submit(g_pool, vm_parmap_task_fn, &tasks[i], NULL);
                }
                vm_pool_wait_all(g_pool);
                for (int i = 0; i < n; i++) {
                    results[i] = tasks[i].output;
                }
            } else {
                /* Fallback: sequential */
                for (int i = 0; i < n; i++) {
                    results[i] = vm_call_closure_from_native(vm, fn, &elems[i], 1);
                }
            }
        } else {
            /* Sequential: small list or no pool */
            for (int i = 0; i < n; i++) {
                results[i] = vm_call_closure_from_native(vm, fn, &elems[i], 1);
            }
        }

        /* Build result list from array (reverse order → cons → correct order) */
        Value result = NIL_VAL;
        for (int i = n - 1; i >= 0; i--) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) break;
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = results[i];
            vm->heap.objects[p]->cons.cdr = result;
            result = PAIR_VAL(p);
        }
        vm_push(vm, result);
        break;
    }
    case 621: { /* parallel-filter(pred, list) — parallel predicate evaluation */
        Value list = vm_pop(vm), pred = vm_pop(vm);

        /* Count and extract elements */
        int n = 0;
        Value cur = list;
        while (cur.type == VAL_PAIR && n < 4096) { n++; cur = vm->heap.objects[cur.as.ptr]->cons.cdr; }
        if (n == 0) { vm_push(vm, NIL_VAL); break; }

        Value* elems = (Value*)vm_alloc(&vm->heap.regions, (size_t)n * sizeof(Value));
        Value* preds = (Value*)vm_alloc(&vm->heap.regions, (size_t)n * sizeof(Value));
        if (!elems || !preds) { vm_push(vm, NIL_VAL); break; }
        cur = list;
        for (int i = 0; i < n; i++) {
            elems[i] = vm->heap.objects[cur.as.ptr]->cons.car;
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }

        /* Evaluate predicates (parallel via pool if available) */
        if (g_pool && n >= 4) {
            VmParMapTask* tasks = (VmParMapTask*)vm_alloc(&vm->heap.regions,
                                      (size_t)n * sizeof(VmParMapTask));
            if (tasks) {
                for (int i = 0; i < n; i++) {
                    tasks[i].main_vm = vm; tasks[i].closure = pred;
                    tasks[i].input = elems[i]; tasks[i].output = NIL_VAL;
                }
                for (int i = 0; i < n; i++)
                    vm_pool_submit(g_pool, vm_parmap_task_fn, &tasks[i], NULL);
                vm_pool_wait_all(g_pool);
                for (int i = 0; i < n; i++) preds[i] = tasks[i].output;
            } else {
                for (int i = 0; i < n; i++)
                    preds[i] = vm_call_closure_from_native(vm, pred, &elems[i], 1);
            }
        } else {
            for (int i = 0; i < n; i++)
                preds[i] = vm_call_closure_from_native(vm, pred, &elems[i], 1);
        }

        /* Build filtered list (correct order) */
        Value result = NIL_VAL;
        for (int i = n - 1; i >= 0; i--) {
            if (is_truthy(preds[i])) {
                int32_t p = heap_alloc(&vm->heap);
                if (p < 0) break;
                vm->heap.objects[p]->type = HEAP_CONS;
                vm->heap.objects[p]->cons.car = elems[i];
                vm->heap.objects[p]->cons.cdr = result;
                result = PAIR_VAL(p);
            }
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
    case 623: { /* parallel-for-each(fn, list) — parallel side-effect execution */
        Value list = vm_pop(vm), fn = vm_pop(vm);

        int n = 0;
        Value cur = list;
        while (cur.type == VAL_PAIR && n < 4096) { n++; cur = vm->heap.objects[cur.as.ptr]->cons.cdr; }
        if (n == 0) { vm_push(vm, NIL_VAL); break; }

        Value* elems = (Value*)vm_alloc(&vm->heap.regions, (size_t)n * sizeof(Value));
        if (!elems) { vm_push(vm, NIL_VAL); break; }
        cur = list;
        for (int i = 0; i < n; i++) {
            elems[i] = vm->heap.objects[cur.as.ptr]->cons.car;
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }

        if (g_pool && n >= 4) {
            VmParMapTask* tasks = (VmParMapTask*)vm_alloc(&vm->heap.regions,
                                      (size_t)n * sizeof(VmParMapTask));
            if (tasks) {
                for (int i = 0; i < n; i++) {
                    tasks[i].main_vm = vm; tasks[i].closure = fn;
                    tasks[i].input = elems[i]; tasks[i].output = NIL_VAL;
                }
                for (int i = 0; i < n; i++)
                    vm_pool_submit(g_pool, vm_parmap_task_fn, &tasks[i], NULL);
                vm_pool_wait_all(g_pool);
            } else {
                for (int i = 0; i < n; i++)
                    vm_call_closure_from_native(vm, fn, &elems[i], 1);
            }
        } else {
            for (int i = 0; i < n; i++)
                vm_call_closure_from_native(vm, fn, &elems[i], 1);
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
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_HASH, VAL_HASH, ht);
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
                    VM_PUSH_HEAP_OPAQUE(vm, HEAP_HASH, VAL_HASH, copy);
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
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_BYTEVECTOR, VAL_BYTEVECTOR, bv);
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
            if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_BYTEVECTOR, VAL_BYTEVECTOR, r); break; }
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
            if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_BYTEVECTOR, VAL_BYTEVECTOR, r); break; }
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
                    vm_push(vm, (Value){.type = VAL_BYTEVECTOR, .as.ptr = ptr});
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
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_PARAMETER, VAL_PARAMETER_OBJ, p);
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
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_ERROR, VAL_ERROR_OBJ, e);
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
     * High-level AD: gradient, jacobian, hessian, divergence, curl,
     *                laplacian, directional-derivative (750-756)
     *
     * Architecture:
     *   Forward-mode AD via dual numbers (VmDual). For multi-variable
     *   functions f(x1,...,xn), we perform N forward passes, each seeding
     *   one variable with tangent=1 and the rest with tangent=0. The
     *   tangent of the result gives the partial derivative ∂f/∂xi.
     *
     *   Hessian uses central difference of exact gradients:
     *     f''(x) ≈ (f'(x+h) - f'(x-h)) / (2h)
     *   where f' is computed exactly via dual numbers, giving O(h²)
     *   accuracy with exact first derivatives.
     *
     * Helper: vm_ad_make_dual_val — allocate a dual on the heap, return Value
     * Helper: vm_ad_extract_point — read point from scalar/list/tensor
     * Helper: vm_ad_partial — compute ∂f/∂xi at a multi-variable point
     * Helper: vm_ad_eval_component — call f, extract i-th output component
     * ══════════════════════════════════════════════════════════════════════ */

#define VM_AD_MAX_VARS 64

    /* --- Helper: create a dual number Value on the heap --- */
#define VM_AD_MAKE_DUAL(vm, primal_val, tangent_val, out_val) do { \
    VmDual* _d = vm_dual_new(&(vm)->heap.regions, (primal_val), (tangent_val)); \
    if (!_d) { (out_val) = FLOAT_VAL(0); break; } \
    int32_t _dp = heap_alloc(&(vm)->heap); \
    if (_dp < 0) { (vm)->error = 1; (out_val) = FLOAT_VAL(0); break; } \
    (vm)->heap.objects[_dp]->type = HEAP_DUAL; \
    (vm)->heap.objects[_dp]->opaque.ptr = _d; \
    (out_val) = (Value){.type = VAL_DUAL, .as.ptr = _dp}; \
} while(0)

    case 750: { /* gradient(f, point) → scalar or tensor of partial derivatives
                 * Scalar point: returns f'(x) (same as derivative)
                 * List/tensor point: returns #(∂f/∂x1 ∂f/∂x2 ... ∂f/∂xn) */
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);

        /* Extract point values */
        double point[VM_AD_MAX_VARS];
        int n = 0;

        if (x_val.type == VAL_PAIR) {
            /* List of values: (list x1 x2 ... xn) */
            Value cur = x_val;
            while (cur.type == VAL_PAIR && n < VM_AD_MAX_VARS) {
                point[n++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
        } else if (x_val.type == VAL_TENSOR && x_val.as.ptr >= 0) {
            /* Tensor of values: #(x1 x2 ... xn) */
            VmTensor* t = (VmTensor*)vm->heap.objects[x_val.as.ptr]->opaque.ptr;
            if (t && t->data) {
                n = (int)(t->total < VM_AD_MAX_VARS ? t->total : VM_AD_MAX_VARS);
                for (int i = 0; i < n; i++) point[i] = t->data[i];
            }
        } else {
            /* Scalar: single-variable derivative */
            point[0] = as_number(x_val);
            n = 1;
        }

        if (n == 0) { vm_push(vm, FLOAT_VAL(0)); break; }

        if (n == 1) {
            /* Scalar case: f'(x) via single dual pass */
            Value dual_arg;
            VM_AD_MAKE_DUAL(vm, point[0], 1.0, dual_arg);
            Value result = vm_call_closure_from_native(vm, f_val, &dual_arg, 1);
            if (result.type == VAL_DUAL && result.as.ptr >= 0) {
                VmDual* rd = (VmDual*)vm->heap.objects[result.as.ptr]->opaque.ptr;
                vm_push(vm, FLOAT_VAL(rd ? rd->tangent : 0));
            } else {
                vm_push(vm, FLOAT_VAL(0)); /* constant function */
            }
        } else {
            /* Multi-variable: N forward passes, seed each variable in turn */
            double partials[VM_AD_MAX_VARS];
            for (int i = 0; i < n; i++) {
                Value args[VM_AD_MAX_VARS];
                for (int j = 0; j < n; j++) {
                    VM_AD_MAKE_DUAL(vm, point[j], (j == i) ? 1.0 : 0.0, args[j]);
                }
                Value result = vm_call_closure_from_native(vm, f_val, args, n);
                if (result.type == VAL_DUAL && result.as.ptr >= 0) {
                    VmDual* rd = (VmDual*)vm->heap.objects[result.as.ptr]->opaque.ptr;
                    partials[i] = rd ? rd->tangent : 0;
                } else {
                    partials[i] = 0;
                }
            }
            /* Return as tensor */
            int64_t shape[1] = { n };
            VmTensor* t = vm_tensor_from_data(&vm->heap.regions, partials, shape, 1);
            if (t) {
                VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t);
            } else {
                vm_push(vm, NIL_VAL);
            }
        }
        break;
    }

    case 751: { /* jacobian(f, point) → matrix of partial derivatives
                 * For f: R^n → R^m, returns m×n tensor J[i][j] = ∂fi/∂xj
                 * Scalar→scalar: returns f'(x)
                 * Multi-var→scalar: returns gradient (1×n)
                 * Multi-var→vector: returns m×n Jacobian matrix */
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);

        /* Extract point */
        double point[VM_AD_MAX_VARS];
        int n = 0;
        if (x_val.type == VAL_PAIR) {
            Value cur = x_val;
            while (cur.type == VAL_PAIR && n < VM_AD_MAX_VARS) {
                point[n++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
        } else if (x_val.type == VAL_TENSOR && x_val.as.ptr >= 0) {
            VmTensor* t = (VmTensor*)vm->heap.objects[x_val.as.ptr]->opaque.ptr;
            if (t && t->data) {
                n = (int)(t->total < VM_AD_MAX_VARS ? t->total : VM_AD_MAX_VARS);
                for (int i = 0; i < n; i++) point[i] = t->data[i];
            }
        } else {
            point[0] = as_number(x_val);
            n = 1;
        }

        if (n == 0) { vm_push(vm, FLOAT_VAL(0)); break; }

        /* First pass with variable 0 seeded to determine output dimension m */
        Value probe_args[VM_AD_MAX_VARS];
        for (int j = 0; j < n; j++) {
            VM_AD_MAKE_DUAL(vm, point[j], (j == 0) ? 1.0 : 0.0, probe_args[j]);
        }
        Value probe_result = vm_call_closure_from_native(vm, f_val, probe_args, n);

        /* Determine output dimension: scalar (m=1) or tensor (m = tensor size) */
        int m = 1;
        if (probe_result.type == VAL_TENSOR && probe_result.as.ptr >= 0) {
            VmTensor* rt = (VmTensor*)vm->heap.objects[probe_result.as.ptr]->opaque.ptr;
            if (rt) m = (int)rt->total;
        }

        if (n == 1 && m == 1) {
            /* Scalar → scalar: just the derivative */
            if (probe_result.type == VAL_DUAL && probe_result.as.ptr >= 0) {
                VmDual* rd = (VmDual*)vm->heap.objects[probe_result.as.ptr]->opaque.ptr;
                vm_push(vm, FLOAT_VAL(rd ? rd->tangent : 0));
            } else {
                vm_push(vm, FLOAT_VAL(0));
            }
        } else {
            /* Build m×n Jacobian matrix */
            double* jac_data = (double*)vm_alloc(&vm->heap.regions,
                                                  (size_t)(m * n) * sizeof(double));
            if (!jac_data) { vm_push(vm, NIL_VAL); break; }
            memset(jac_data, 0, (size_t)(m * n) * sizeof(double));

            for (int i = 0; i < n; i++) {
                Value args[VM_AD_MAX_VARS];
                for (int j = 0; j < n; j++) {
                    VM_AD_MAKE_DUAL(vm, point[j], (j == i) ? 1.0 : 0.0, args[j]);
                }
                Value result = vm_call_closure_from_native(vm, f_val, args, n);

                if (m == 1) {
                    /* Scalar output */
                    if (result.type == VAL_DUAL && result.as.ptr >= 0) {
                        VmDual* rd = (VmDual*)vm->heap.objects[result.as.ptr]->opaque.ptr;
                        jac_data[i] = rd ? rd->tangent : 0; /* row 0, col i */
                    }
                } else if (result.type == VAL_TENSOR && result.as.ptr >= 0) {
                    /* Vector output: each element is a dual or scalar */
                    VmTensor* rt = (VmTensor*)vm->heap.objects[result.as.ptr]->opaque.ptr;
                    if (rt && rt->data) {
                        /* Tensor of doubles — tangents already extracted by AD */
                        for (int k = 0; k < m && k < (int)rt->total; k++) {
                            jac_data[k * n + i] = rt->data[k]; /* J[k][i] */
                        }
                    }
                } else if (result.type == VAL_DUAL && result.as.ptr >= 0) {
                    /* Single dual result for m=1 case */
                    VmDual* rd = (VmDual*)vm->heap.objects[result.as.ptr]->opaque.ptr;
                    jac_data[i] = rd ? rd->tangent : 0;
                }
            }

            if (m == 1) {
                /* 1×n → return as 1D tensor (gradient) */
                int64_t shape[1] = { n };
                VmTensor* t = vm_tensor_from_data(&vm->heap.regions, jac_data, shape, 1);
                if (t) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t); }
                else { vm_push(vm, NIL_VAL); }
            } else {
                /* m×n Jacobian matrix */
                int64_t shape[2] = { m, n };
                VmTensor* t = vm_tensor_from_data(&vm->heap.regions, jac_data, shape, 2);
                if (t) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t); }
                else { vm_push(vm, NIL_VAL); }
            }
        }
        break;
    }

    case 752: { /* hessian(f, point) → EXACT second derivative via hyper-dual numbers
                 * Scalar: seed x = (x,1,1,0) → f₁₂ = f''(x)
                 * Multi-var: H[i][j] = ∂²f/∂xᵢ∂xⱼ via hyper-dual seeding
                 * NO finite differences — mathematically exact. */
#define VM_HD_MAKE(vm, fv, f1v, f2v, f12v, out) do { \
    VmHyperDual* _h = vm_hd_make(&(vm)->heap.regions, (fv), (f1v), (f2v), (f12v)); \
    if (!_h) { (out) = FLOAT_VAL(0); break; } \
    int32_t _hp = heap_alloc(&(vm)->heap); \
    if (_hp < 0) { (vm)->error = 1; (out) = FLOAT_VAL(0); break; } \
    (vm)->heap.objects[_hp]->type = HEAP_HYPER_DUAL; \
    (vm)->heap.objects[_hp]->opaque.ptr = _h; \
    (out) = (Value){.type = VAL_HYPER_DUAL, .as.ptr = _hp}; \
} while(0)
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);

        double point[VM_AD_MAX_VARS];
        int n = 0;
        if (x_val.type == VAL_PAIR) {
            Value cur = x_val;
            while (cur.type == VAL_PAIR && n < VM_AD_MAX_VARS) {
                point[n++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
        } else if (x_val.type == VAL_TENSOR && x_val.as.ptr >= 0) {
            VmTensor* t = (VmTensor*)vm->heap.objects[x_val.as.ptr]->opaque.ptr;
            if (t && t->data) {
                n = (int)(t->total < VM_AD_MAX_VARS ? t->total : VM_AD_MAX_VARS);
                for (int i = 0; i < n; i++) point[i] = t->data[i];
            }
        } else {
            point[0] = as_number(x_val);
            n = 1;
        }

        if (n == 0) { vm_push(vm, FLOAT_VAL(0)); break; }

        if (n == 1) {
            /* Scalar hessian via hyper-dual: seed (x, 1, 1, 0) → f₁₂ = f''(x) */
            Value hd_arg;
            VM_HD_MAKE(vm, point[0], 1.0, 1.0, 0.0, hd_arg);
            Value result = vm_call_closure_from_native(vm, f_val, &hd_arg, 1);
            if (result.type == VAL_HYPER_DUAL && result.as.ptr >= 0) {
                VmHyperDual* rh = (VmHyperDual*)vm->heap.objects[result.as.ptr]->opaque.ptr;
                vm_push(vm, FLOAT_VAL(rh ? rh->f12 : 0.0));
            } else {
                vm_push(vm, FLOAT_VAL(0.0));
            }
        } else {
            /* Multi-variable Hessian via hyper-dual: H[i][j] = ∂²f/∂xᵢ∂xⱼ
             * Seed xₖ = (point[k], δₖᵢ, δₖⱼ, 0) → result.f12 = H[i][j] */
            double* hess_data = (double*)vm_alloc(&vm->heap.regions,
                                                   (size_t)(n * n) * sizeof(double));
            if (!hess_data) { vm_push(vm, NIL_VAL); break; }

            for (int i = 0; i < n; i++) {
                for (int j = i; j < n; j++) {
                    Value args[VM_AD_MAX_VARS];
                    for (int k = 0; k < n; k++) {
                        VM_HD_MAKE(vm, point[k], (k==i)?1.0:0.0, (k==j)?1.0:0.0, 0.0, args[k]);
                    }
                    Value r = vm_call_closure_from_native(vm, f_val, args, n);
                    double h_ij = 0.0;
                    if (r.type == VAL_HYPER_DUAL && r.as.ptr >= 0) {
                        VmHyperDual* rh = (VmHyperDual*)vm->heap.objects[r.as.ptr]->opaque.ptr;
                        if (rh) h_ij = rh->f12;
                    }
                    hess_data[i * n + j] = h_ij;
                    hess_data[j * n + i] = h_ij;
                }
            }

            int64_t shape[2] = { n, n };
            VmTensor* t = vm_tensor_from_data(&vm->heap.regions, hess_data, shape, 2);
            if (t) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t); }
            else { vm_push(vm, NIL_VAL); }
        }
#undef VM_HD_MAKE
        break;
    }

    case 753: { /* divergence(F, point) → scalar
                 * div(F) = ∂F1/∂x1 + ∂F2/∂x2 + ... + ∂Fn/∂xn
                 * F: R^n → R^n (vector field), point: list or tensor */
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);

        double point[VM_AD_MAX_VARS];
        int n = 0;
        if (x_val.type == VAL_PAIR) {
            Value cur = x_val;
            while (cur.type == VAL_PAIR && n < VM_AD_MAX_VARS) {
                point[n++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
        } else if (x_val.type == VAL_TENSOR && x_val.as.ptr >= 0) {
            VmTensor* t = (VmTensor*)vm->heap.objects[x_val.as.ptr]->opaque.ptr;
            if (t && t->data) {
                n = (int)(t->total < VM_AD_MAX_VARS ? t->total : VM_AD_MAX_VARS);
                for (int i = 0; i < n; i++) point[i] = t->data[i];
            }
        } else {
            point[0] = as_number(x_val);
            n = 1;
        }

        if (n == 0) { vm_push(vm, FLOAT_VAL(0)); break; }

        /* Sum of ∂Fi/∂xi: for each i, seed variable i, extract component i */
        double div = 0;
        for (int i = 0; i < n; i++) {
            Value args[VM_AD_MAX_VARS];
            for (int j = 0; j < n; j++) {
                VM_AD_MAKE_DUAL(vm, point[j], (j == i) ? 1.0 : 0.0, args[j]);
            }
            Value result = vm_call_closure_from_native(vm, f_val, args, n);

            /* Extract the i-th component's tangent */
            if (result.type == VAL_TENSOR && result.as.ptr >= 0) {
                /* F returns a tensor — we need the tangent of element i.
                 * Since the tensor contains primals (doubles), the tangent
                 * information is lost. We need to use a different approach:
                 * call F component-wise. But if F returns a tensor of duals,
                 * we can extract directly. For tensor-returning functions,
                 * use finite differences as fallback. */
                VmTensor* rt = (VmTensor*)vm->heap.objects[result.as.ptr]->opaque.ptr;
                if (rt && rt->data && i < (int)rt->total) {
                    /* Tensor element — use finite difference for this component */
                    double fplus, fminus;
                    double pt_plus[VM_AD_MAX_VARS], pt_minus[VM_AD_MAX_VARS];
                    double h = 1e-7;
                    for (int k = 0; k < n; k++) {
                        pt_plus[k] = point[k] + ((k == i) ? h : 0);
                        pt_minus[k] = point[k] - ((k == i) ? h : 0);
                    }
                    /* F(x + h*ei)[i] */
                    Value ap[VM_AD_MAX_VARS];
                    for (int k = 0; k < n; k++) ap[k] = FLOAT_VAL(pt_plus[k]);
                    Value rp = vm_call_closure_from_native(vm, f_val, ap, n);
                    fplus = 0;
                    if (rp.type == VAL_TENSOR && rp.as.ptr >= 0) {
                        VmTensor* tp = (VmTensor*)vm->heap.objects[rp.as.ptr]->opaque.ptr;
                        if (tp && tp->data && i < (int)tp->total) fplus = tp->data[i];
                    }
                    /* F(x - h*ei)[i] */
                    Value am[VM_AD_MAX_VARS];
                    for (int k = 0; k < n; k++) am[k] = FLOAT_VAL(pt_minus[k]);
                    Value rm = vm_call_closure_from_native(vm, f_val, am, n);
                    fminus = 0;
                    if (rm.type == VAL_TENSOR && rm.as.ptr >= 0) {
                        VmTensor* tm = (VmTensor*)vm->heap.objects[rm.as.ptr]->opaque.ptr;
                        if (tm && tm->data && i < (int)tm->total) fminus = tm->data[i];
                    }
                    div += (fplus - fminus) / (2.0 * h);
                }
            } else if (result.type == VAL_PAIR) {
                /* F returns a list — walk to i-th element, extract tangent */
                Value cur = result;
                for (int k = 0; k < i && cur.type == VAL_PAIR; k++) {
                    cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
                }
                if (cur.type == VAL_PAIR) {
                    Value elem = vm->heap.objects[cur.as.ptr]->cons.car;
                    if (elem.type == VAL_DUAL && elem.as.ptr >= 0) {
                        VmDual* rd = (VmDual*)vm->heap.objects[elem.as.ptr]->opaque.ptr;
                        if (rd) div += rd->tangent;
                    }
                }
            } else if (n == 1 && result.type == VAL_DUAL && result.as.ptr >= 0) {
                /* Scalar function */
                VmDual* rd = (VmDual*)vm->heap.objects[result.as.ptr]->opaque.ptr;
                if (rd) div += rd->tangent;
            }
        }
        vm_push(vm, FLOAT_VAL(div));
        break;
    }

    case 754: { /* curl(F, point) → 3D vector
                 * curl(F) = (∂F3/∂y - ∂F2/∂z, ∂F1/∂z - ∂F3/∂x, ∂F2/∂x - ∂F1/∂y)
                 * F: R^3 → R^3, point must have exactly 3 components */
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);

        double point[3];
        int n = 0;
        if (x_val.type == VAL_PAIR) {
            Value cur = x_val;
            while (cur.type == VAL_PAIR && n < 3) {
                point[n++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
        } else if (x_val.type == VAL_TENSOR && x_val.as.ptr >= 0) {
            VmTensor* t = (VmTensor*)vm->heap.objects[x_val.as.ptr]->opaque.ptr;
            if (t && t->data) {
                n = (int)(t->total < 3 ? t->total : 3);
                for (int i = 0; i < n; i++) point[i] = t->data[i];
            }
        }

        if (n != 3) {
            /* Curl requires exactly 3 dimensions */
            vm_push(vm, NIL_VAL);
            break;
        }

        /* Compute the 3×3 Jacobian via central differences:
         * J[i][j] = ∂Fi/∂xj
         * curl = (J[2][1] - J[1][2], J[0][2] - J[2][0], J[1][0] - J[0][1]) */
        double jac[3][3];
        double h = 1e-7;

        for (int j = 0; j < 3; j++) {
            /* ∂F/∂xj via central difference */
            Value ap[3], am[3];
            for (int k = 0; k < 3; k++) {
                ap[k] = FLOAT_VAL(point[k] + ((k == j) ? h : 0));
                am[k] = FLOAT_VAL(point[k] - ((k == j) ? h : 0));
            }
            Value rp = vm_call_closure_from_native(vm, f_val, ap, 3);
            Value rm = vm_call_closure_from_native(vm, f_val, am, 3);

            /* Extract 3 components from each result */
            double fp[3] = {0,0,0}, fm[3] = {0,0,0};
            if (rp.type == VAL_TENSOR && rp.as.ptr >= 0) {
                VmTensor* tp = (VmTensor*)vm->heap.objects[rp.as.ptr]->opaque.ptr;
                if (tp && tp->data) for (int i = 0; i < 3 && i < (int)tp->total; i++) fp[i] = tp->data[i];
            } else if (rp.type == VAL_PAIR) {
                Value cur = rp; int idx = 0;
                while (cur.type == VAL_PAIR && idx < 3) {
                    fp[idx++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                    cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
                }
            }
            if (rm.type == VAL_TENSOR && rm.as.ptr >= 0) {
                VmTensor* tm = (VmTensor*)vm->heap.objects[rm.as.ptr]->opaque.ptr;
                if (tm && tm->data) for (int i = 0; i < 3 && i < (int)tm->total; i++) fm[i] = tm->data[i];
            } else if (rm.type == VAL_PAIR) {
                Value cur = rm; int idx = 0;
                while (cur.type == VAL_PAIR && idx < 3) {
                    fm[idx++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                    cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
                }
            }

            for (int i = 0; i < 3; i++) {
                jac[i][j] = (fp[i] - fm[i]) / (2.0 * h);
            }
        }

        /* curl = (∂F3/∂y - ∂F2/∂z, ∂F1/∂z - ∂F3/∂x, ∂F2/∂x - ∂F1/∂y) */
        double curl_data[3] = {
            jac[2][1] - jac[1][2],  /* ∂F3/∂y - ∂F2/∂z */
            jac[0][2] - jac[2][0],  /* ∂F1/∂z - ∂F3/∂x */
            jac[1][0] - jac[0][1]   /* ∂F2/∂x - ∂F1/∂y */
        };
        int64_t shape[1] = { 3 };
        VmTensor* t = vm_tensor_from_data(&vm->heap.regions, curl_data, shape, 1);
        if (t) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t); }
        else { vm_push(vm, NIL_VAL); }
        break;
    }

    case 755: { /* laplacian(f, point) → scalar
                 * ∇²f = ∂²f/∂x1² + ∂²f/∂x2² + ... + ∂²f/∂xn²
                 * = trace of the Hessian matrix */
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);

        double point[VM_AD_MAX_VARS];
        int n = 0;
        if (x_val.type == VAL_PAIR) {
            Value cur = x_val;
            while (cur.type == VAL_PAIR && n < VM_AD_MAX_VARS) {
                point[n++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
        } else if (x_val.type == VAL_TENSOR && x_val.as.ptr >= 0) {
            VmTensor* t = (VmTensor*)vm->heap.objects[x_val.as.ptr]->opaque.ptr;
            if (t && t->data) {
                n = (int)(t->total < VM_AD_MAX_VARS ? t->total : VM_AD_MAX_VARS);
                for (int i = 0; i < n; i++) point[i] = t->data[i];
            }
        } else {
            point[0] = as_number(x_val);
            n = 1;
        }

        if (n == 0) { vm_push(vm, FLOAT_VAL(0)); break; }

        /* Trace of Hessian: Σ ∂²f/∂xi²
         * Each diagonal uses: (f'_i(x+h*ei) - f'_i(x-h*ei)) / (2h)
         * where f'_i is the partial derivative w.r.t. xi (exact via dual) */
        double h = 1e-5;
        double laplacian = 0;

        for (int i = 0; i < n; i++) {
            /* Compute ∂f/∂xi at x + h*ei and x - h*ei */
            double grad_plus = 0, grad_minus = 0;

            /* ∂f/∂xi at x + h*ei */
            {
                double pt[VM_AD_MAX_VARS];
                for (int k = 0; k < n; k++) pt[k] = point[k] + ((k == i) ? h : 0);
                Value args[VM_AD_MAX_VARS];
                for (int k = 0; k < n; k++) {
                    VM_AD_MAKE_DUAL(vm, pt[k], (k == i) ? 1.0 : 0.0, args[k]);
                }
                Value r = vm_call_closure_from_native(vm, f_val, args, n);
                if (r.type == VAL_DUAL && r.as.ptr >= 0) {
                    VmDual* rd = (VmDual*)vm->heap.objects[r.as.ptr]->opaque.ptr;
                    if (rd) grad_plus = rd->tangent;
                }
            }
            /* ∂f/∂xi at x - h*ei */
            {
                double pt[VM_AD_MAX_VARS];
                for (int k = 0; k < n; k++) pt[k] = point[k] - ((k == i) ? h : 0);
                Value args[VM_AD_MAX_VARS];
                for (int k = 0; k < n; k++) {
                    VM_AD_MAKE_DUAL(vm, pt[k], (k == i) ? 1.0 : 0.0, args[k]);
                }
                Value r = vm_call_closure_from_native(vm, f_val, args, n);
                if (r.type == VAL_DUAL && r.as.ptr >= 0) {
                    VmDual* rd = (VmDual*)vm->heap.objects[r.as.ptr]->opaque.ptr;
                    if (rd) grad_minus = rd->tangent;
                }
            }

            laplacian += (grad_plus - grad_minus) / (2.0 * h);
        }

        vm_push(vm, FLOAT_VAL(laplacian));
        break;
    }

    case 756: { /* directional-derivative(f, point, direction) → scalar
                 * D_v(f) = ∇f · v = Σ (∂f/∂xi * vi)
                 * Uses a single forward pass with tangent = direction vector */
        Value dir_val = vm_pop(vm), x_val = vm_pop(vm), f_val = vm_pop(vm);

        double point[VM_AD_MAX_VARS], dir[VM_AD_MAX_VARS];
        int n = 0, nd = 0;

        /* Extract point */
        if (x_val.type == VAL_PAIR) {
            Value cur = x_val;
            while (cur.type == VAL_PAIR && n < VM_AD_MAX_VARS) {
                point[n++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
        } else if (x_val.type == VAL_TENSOR && x_val.as.ptr >= 0) {
            VmTensor* t = (VmTensor*)vm->heap.objects[x_val.as.ptr]->opaque.ptr;
            if (t && t->data) {
                n = (int)(t->total < VM_AD_MAX_VARS ? t->total : VM_AD_MAX_VARS);
                for (int i = 0; i < n; i++) point[i] = t->data[i];
            }
        } else {
            point[0] = as_number(x_val);
            n = 1;
        }

        /* Extract direction */
        if (dir_val.type == VAL_PAIR) {
            Value cur = dir_val;
            while (cur.type == VAL_PAIR && nd < VM_AD_MAX_VARS) {
                dir[nd++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
        } else if (dir_val.type == VAL_TENSOR && dir_val.as.ptr >= 0) {
            VmTensor* t = (VmTensor*)vm->heap.objects[dir_val.as.ptr]->opaque.ptr;
            if (t && t->data) {
                nd = (int)(t->total < VM_AD_MAX_VARS ? t->total : VM_AD_MAX_VARS);
                for (int i = 0; i < nd; i++) dir[i] = t->data[i];
            }
        } else {
            dir[0] = as_number(dir_val);
            nd = 1;
        }

        if (n == 0 || nd != n) {
            vm_push(vm, FLOAT_VAL(0));
            break;
        }

        /* Single forward pass: seed tangent = direction vector
         * D_v(f)(x) = Σ vi * ∂f/∂xi = tangent when all tangents are vi
         * This is the efficient approach — one pass instead of n+1 */
        Value args[VM_AD_MAX_VARS];
        for (int j = 0; j < n; j++) {
            VM_AD_MAKE_DUAL(vm, point[j], dir[j], args[j]);
        }
        Value result = vm_call_closure_from_native(vm, f_val, args, n);
        if (result.type == VAL_DUAL && result.as.ptr >= 0) {
            VmDual* rd = (VmDual*)vm->heap.objects[result.as.ptr]->opaque.ptr;
            vm_push(vm, FLOAT_VAL(rd ? rd->tangent : 0));
        } else {
            vm_push(vm, FLOAT_VAL(0));
        }
        break;
    }

#undef VM_AD_MAKE_DUAL
#undef VM_AD_MAX_VARS


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

    case 142: { /* add2 — complex-aware */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        if (a_val.type == VAL_COMPLEX || b_val.type == VAL_COMPLEX) {
            VmComplex a_z = {as_number(a_val), 0}, b_z = {as_number(b_val), 0};
            if (a_val.type == VAL_COMPLEX) a_z = *(VmComplex*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_COMPLEX) b_z = *(VmComplex*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            VmComplex* r = vm_complex_add(&vm->heap.regions, &a_z, &b_z);
            if (!r) { vm->error = 1; break; }
            int32_t p = heap_alloc(&vm->heap); if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_COMPLEX; vm->heap.objects[p]->opaque.ptr = r;
            vm_push(vm, (Value){.type = VAL_COMPLEX, .as.ptr = p});
        } else { vm_push(vm, number_val(as_number(a_val) + as_number(b_val))); }
        break; }
    case 143: { /* sub2 — complex-aware */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        if (a_val.type == VAL_COMPLEX || b_val.type == VAL_COMPLEX) {
            VmComplex a_z = {as_number(a_val), 0}, b_z = {as_number(b_val), 0};
            if (a_val.type == VAL_COMPLEX) a_z = *(VmComplex*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_COMPLEX) b_z = *(VmComplex*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            VmComplex* r = vm_complex_sub(&vm->heap.regions, &a_z, &b_z);
            if (!r) { vm->error = 1; break; }
            int32_t p = heap_alloc(&vm->heap); if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_COMPLEX; vm->heap.objects[p]->opaque.ptr = r;
            vm_push(vm, (Value){.type = VAL_COMPLEX, .as.ptr = p});
        } else { vm_push(vm, number_val(as_number(a_val) - as_number(b_val))); }
        break; }
    case 144: { /* mul2 — complex-aware */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        if (a_val.type == VAL_COMPLEX || b_val.type == VAL_COMPLEX) {
            VmComplex a_z = {as_number(a_val), 0}, b_z = {as_number(b_val), 0};
            if (a_val.type == VAL_COMPLEX) a_z = *(VmComplex*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_COMPLEX) b_z = *(VmComplex*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            VmComplex* r = vm_complex_mul(&vm->heap.regions, &a_z, &b_z);
            if (!r) { vm->error = 1; break; }
            int32_t p = heap_alloc(&vm->heap); if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_COMPLEX; vm->heap.objects[p]->opaque.ptr = r;
            vm_push(vm, (Value){.type = VAL_COMPLEX, .as.ptr = p});
        } else { vm_push(vm, number_val(as_number(a_val) * as_number(b_val))); }
        break; }
    case 145: { /* div2 — complex-aware */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        if (a_val.type == VAL_COMPLEX || b_val.type == VAL_COMPLEX) {
            VmComplex a_z = {as_number(a_val), 0}, b_z = {as_number(b_val), 0};
            if (a_val.type == VAL_COMPLEX) a_z = *(VmComplex*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_COMPLEX) b_z = *(VmComplex*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            VmComplex* r = vm_complex_div(&vm->heap.regions, &a_z, &b_z);
            if (!r) { vm->error = 1; break; }
            int32_t p = heap_alloc(&vm->heap); if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_COMPLEX; vm->heap.objects[p]->opaque.ptr = r;
            vm_push(vm, (Value){.type = VAL_COMPLEX, .as.ptr = p});
        } else { vm_push(vm, number_val(as_number(a_val) / as_number(b_val))); }
        break; }
    /* Comparison operators as first-class functions (for sort, map, fold, etc.) */
    case 146: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) < as_number(b))); break; }  /* < */
    case 147: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) > as_number(b))); break; }  /* > */
    case 148: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) <= as_number(b))); break; } /* <= */
    case 149: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) >= as_number(b))); break; } /* >= */
    case 150: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) == as_number(b))); break; } /* = */

    /* Core operations as first-class native functions (IDs 200-226) */
    case 200: { Value a = vm_pop(vm); /* car */
        if (a.type == VAL_PAIR) { HeapObject* o = vm->heap.objects[a.as.ptr]; vm_push(vm, o->cons.car); }
        else { fprintf(stderr, "CAR on non-pair\n"); vm_push(vm, NIL_VAL); } break; }
    case 201: { Value a = vm_pop(vm); /* cdr */
        if (a.type == VAL_PAIR) { HeapObject* o = vm->heap.objects[a.as.ptr]; vm_push(vm, o->cons.cdr); }
        else { fprintf(stderr, "CDR on non-pair\n"); vm_push(vm, NIL_VAL); } break; }
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
    case 211: { Value a = vm_pop(vm); print_value(vm, a); fflush(stdout); vm_push(vm, (Value){.type = VAL_VOID}); break; } /* display */
    case 212: { Value a = vm_pop(vm); print_value(vm, a); fflush(stdout); vm_push(vm, (Value){.type = VAL_VOID}); break; } /* write */
    case 213: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(as_number_vm(vm, a))); break; }  /* exact->inexact */
    case 214: { Value a = vm_pop(vm); vm_push(vm, INT_VAL((int64_t)as_number_vm(vm, a))); break; } /* inexact->exact */
    case 215: { /* string->number — handles #x/#b/#o/#d prefixes */
        Value a = vm_pop(vm);
        if (a.type != VAL_STRING) { vm_push(vm, BOOL_VAL(0)); break; }
        VmString* s215 = (VmString*)vm->heap.objects[a.as.ptr]->opaque.ptr;
        if (!s215 || !s215->data || s215->data[0] == '\0') { vm_push(vm, BOOL_VAL(0)); break; }
        const char* p215 = s215->data;
        int radix215 = 10;
        if (p215[0] == '#') {
            char pfx = (char)(p215[1] | 32); /* lowercase */
            if      (pfx == 'x') { radix215 = 16; p215 += 2; }
            else if (pfx == 'b') { radix215 = 2;  p215 += 2; }
            else if (pfx == 'o') { radix215 = 8;  p215 += 2; }
            else if (pfx == 'd') { radix215 = 10; p215 += 2; }
            else { vm_push(vm, BOOL_VAL(0)); break; }
        }
        char* end215 = NULL;
        if (radix215 == 10) {
            /* Try integer first; fall back to float */
            long long iv = strtoll(p215, &end215, 10);
            if (*end215 == '\0' && end215 != p215) { vm_push(vm, INT_VAL((int64_t)iv)); break; }
            double dv = strtod(p215, &end215);
            if (*end215 == '\0' && end215 != p215) { vm_push(vm, FLOAT_VAL(dv)); break; }
            vm_push(vm, BOOL_VAL(0));
        } else {
            long long iv = strtoll(p215, &end215, radix215);
            if (*end215 == '\0' && end215 != p215) { vm_push(vm, INT_VAL((int64_t)iv)); break; }
            vm_push(vm, BOOL_VAL(0));
        }
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
    case 222: { /* string->list */
        Value s_val = vm_pop(vm);
        Value result = NIL_VAL;
        if (s_val.type == VAL_STRING) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            if (s) for (int i = s->char_len - 1; i >= 0; i--) {
                int cp = vm_string_ref(s, i);
                int32_t p = heap_alloc(&vm->heap); if (p < 0) break;
                vm->heap.objects[p]->type = HEAP_CONS;
                vm->heap.objects[p]->cons.car = INT_VAL(cp);
                vm->heap.objects[p]->cons.cdr = result;
                result = PAIR_VAL(p);
            }
        }
        vm_push(vm, result); break;
    }
    case 223: { /* list->string */
        Value lst = vm_pop(vm);
        char buf[4096]; int len = 0;
        Value cur = lst;
        while (cur.type == VAL_PAIR && len < 4095) {
            int cp = (int)as_number(vm->heap.objects[cur.as.ptr]->cons.car);
            if (cp >= 0 && cp < 128) buf[len++] = (char)cp;
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        buf[len] = 0;
        VmString* s = vm_string_from_cstr(&vm->heap.regions, buf);
        if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); }
        else vm_push(vm, NIL_VAL);
        break;
    }
    case 224: { /* gcd */
        Value b = vm_pop(vm), a = vm_pop(vm);
        int64_t x = llabs((int64_t)as_number(a)), y = llabs((int64_t)as_number(b));
        while (y != 0) { int64_t t = y; y = x % y; x = t; }
        vm_push(vm, INT_VAL(x)); break;
    }
    case 225: { /* lcm */
        Value b = vm_pop(vm), a = vm_pop(vm);
        int64_t x = llabs((int64_t)as_number(a)), y = llabs((int64_t)as_number(b));
        if (x == 0 || y == 0) { vm_push(vm, INT_VAL(0)); break; }
        int64_t g = x, h = y;
        while (h != 0) { int64_t t = h; h = g % h; g = t; }
        vm_push(vm, INT_VAL(x / g * y)); break;
    }
    case 226: { /* make-string(n, char) */
        Value ch = vm_pop(vm), n = vm_pop(vm);
        int sz = (int)as_number(n), c = (int)as_number(ch);
        if (sz < 0) sz = 0; if (sz > 65536) sz = 65536;
        char* buf = (char*)vm_alloc(&vm->heap.regions, (size_t)(sz + 1));
        if (buf) { memset(buf, c > 0 && c < 128 ? c : ' ', sz); buf[sz] = 0;
            VmString* s = vm_string_from_cstr(&vm->heap.regions, buf);
            if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
        }
        vm_push(vm, NIL_VAL); break;
    }

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

    case 130: { /* raise: dispatch to handler or error */
        Value exn = vm_pop(vm);
        vm->current_exception = exn;
        if (vm->n_handlers > 0) {
            vm->n_handlers--;
            /* Unwind dynamic-wind after-thunks */
            int target_winds = vm->handler_stack[vm->n_handlers].n_winds;
            while (vm->n_winds > target_winds) {
                vm->n_winds--;
                Value after = vm->wind_stack[vm->n_winds].after;
                if (after.type == VAL_CLOSURE)
                    vm_call_closure_from_native(vm, after, NULL, 0);
            }
            vm->sp = vm->handler_stack[vm->n_handlers].sp;
            vm->fp = vm->handler_stack[vm->n_handlers].fp;
            vm->frame_count = vm->handler_stack[vm->n_handlers].frame_count;
            vm->pc = vm->handler_stack[vm->n_handlers].pc;
        } else {
            fprintf(stderr, "ERROR: unhandled exception: ");
            print_value(vm, exn);
            fprintf(stderr, "\n");
            vm->error = 1;
        }
        break;
    }

    case 132: { /* force: force a promise (thunk memoization) */
        Value promise = vm_pop(vm);
        if (promise.type == VAL_VECTOR) {
            /* Promise is a vector: #(forced? thunk result) */
            VmVector* v = (VmVector*)vm->heap.objects[promise.as.ptr]->opaque.ptr;
            if (v && v->len >= 3) {
                if (is_truthy(v->items[0])) {
                    vm_push(vm, v->items[2]); /* already forced: return cached */
                } else {
                    /* Call thunk, cache result */
                    Value thunk = v->items[1];
                    Value result = vm_call_closure_from_native(vm, thunk, NULL, 0);
                    v->items[0] = BOOL_VAL(1); /* mark forced */
                    v->items[2] = result;
                    vm_push(vm, result);
                }
                break;
            }
        }
        /* Fallback: non-promise, return as-is */
        vm_push(vm, promise);
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
    case 251: { /* call-with-values-apply: unpack multi-value result */
        Value consumer = vm_pop(vm), result = vm_pop(vm);
        if (consumer.type == VAL_CLOSURE) {
            /* If result is a vector (multi-value container), unpack */
            if (result.type == VAL_VECTOR) {
                VmVector* mv = (VmVector*)vm->heap.objects[result.as.ptr]->opaque.ptr;
                if (mv && mv->len > 0) {
                    Value r = vm_call_closure_from_native(vm, consumer, mv->items, mv->len);
                    vm_push(vm, r);
                } else {
                    Value r = vm_call_closure_from_native(vm, consumer, NULL, 0);
                    vm_push(vm, r);
                }
            } else {
                Value args[1] = {result};
                Value r = vm_call_closure_from_native(vm, consumer, args, 1);
                vm_push(vm, r);
            }
        } else vm_push(vm, result);
        break;
    }
    case 252: { /* propagate upvalue: copy parent closure's upvalue[slot] into child upvalue[uv_idx].
                 * Called when a lambda inside a function captures a variable via the parent's upvalue
                 * (is_local=false). The parent closure lives at stack[fp-1] per calling convention. */
        Value slot_v = vm_pop(vm), uv_idx_v = vm_pop(vm), cl_val = vm_pop(vm);
        if (cl_val.type == VAL_CLOSURE) {
            HeapObject* cl = vm->heap.objects[cl_val.as.ptr];
            int uv_idx = (int)as_number(uv_idx_v);
            int slot = (int)as_number(slot_v);
            /* Read from the PARENT closure's upvalue array, not the stack frame.
             * Bug was: vm->stack[vm->fp + slot] reads local slot index `slot` which is
             * wrong — `slot` is an upvalue index, not a stack-frame offset. */
            if (vm->fp > 0) {
                Value parent_val = vm->stack[vm->fp - 1];
                if (parent_val.type == VAL_CLOSURE) {
                    HeapObject* parent_cl = vm->heap.objects[parent_val.as.ptr];
                    if (uv_idx >= 0 && uv_idx < cl->closure.n_upvalues &&
                        slot >= 0 && slot < parent_cl->closure.n_upvalues) {
                        cl->closure.upvalues[uv_idx] = parent_cl->closure.upvalues[slot];
                    }
                }
            }
        }
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

    /* ══════════════════════════════════════════════════════════════════════
     * Character operations (1680-1691)
     * ══════════════════════════════════════════════════════════════════════ */
    case 1680: { Value a = vm_pop(vm); int c = (int)as_number(a); vm_push(vm, BOOL_VAL(isalpha(c))); break; }
    case 1681: { Value a = vm_pop(vm); int c = (int)as_number(a); vm_push(vm, BOOL_VAL(isdigit(c))); break; }
    case 1682: { Value a = vm_pop(vm); int c = (int)as_number(a); vm_push(vm, BOOL_VAL(isspace(c))); break; }
    case 1683: { Value a = vm_pop(vm); int c = (int)as_number(a); vm_push(vm, BOOL_VAL(isupper(c))); break; }
    case 1684: { Value a = vm_pop(vm); int c = (int)as_number(a); vm_push(vm, BOOL_VAL(islower(c))); break; }
    case 1685: { Value a = vm_pop(vm); int c = (int)as_number(a); vm_push(vm, INT_VAL(toupper(c))); break; }
    case 1686: { Value a = vm_pop(vm); int c = (int)as_number(a); vm_push(vm, INT_VAL(tolower(c))); break; }
    case 1687: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL((int)as_number(a) == (int)as_number(b))); break; }
    case 1688: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL((int)as_number(a) < (int)as_number(b))); break; }
    case 1689: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL((int)as_number(a) > (int)as_number(b))); break; }

    /* ══════════════════════════════════════════════════════════════════════
     * Bitwise operations (1692-1696)
     * ══════════════════════════════════════════════════════════════════════ */
    case 1692: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, INT_VAL((int64_t)as_number(a) & (int64_t)as_number(b))); break; }
    case 1693: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, INT_VAL((int64_t)as_number(a) | (int64_t)as_number(b))); break; }
    case 1694: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, INT_VAL((int64_t)as_number(a) ^ (int64_t)as_number(b))); break; }
    case 1695: { Value a = vm_pop(vm); vm_push(vm, INT_VAL(~(int64_t)as_number(a))); break; }
    case 1696: { Value b = vm_pop(vm), a = vm_pop(vm);
        int64_t val = (int64_t)as_number(a), shift = (int64_t)as_number(b);
        vm_push(vm, INT_VAL(shift >= 0 ? val << shift : val >> (-shift))); break; }

    /* ══════════════════════════════════════════════════════════════════════
     * Type predicates (1697-1699)
     * ══════════════════════════════════════════════════════════════════════ */
    case 1697: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_INT || a.type == VAL_FLOAT || a.type == VAL_RATIONAL)); break; }
    case 1698: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_RATIONAL)); break; }
    case 1699: { Value a = vm_pop(vm);
        vm_push(vm, BOOL_VAL(a.type == VAL_TENSOR)); break; }

    /* ══════════════════════════════════════════════════════════════════════
     * Additional predicates (160-166)
     * ══════════════════════════════════════════════════════════════════════ */
    case 160: { Value a = vm_pop(vm); /* symbol? — in the VM, symbols are interned strings */
        vm_push(vm, BOOL_VAL(a.type == VAL_STRING)); break; }
    case 161: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_INT && as_number(a) >= 0 && as_number(a) <= 0x10FFFF)); break; } /* char? */
    case 162: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_INT || a.type == VAL_RATIONAL)); break; } /* exact? */
    case 163: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_FLOAT)); break; } /* inexact? */
    case 164: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_FLOAT && isnan(a.as.f))); break; } /* nan? */
    case 165: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_FLOAT && isinf(a.as.f))); break; } /* infinite? */
    case 166: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type != VAL_FLOAT || isfinite(a.as.f))); break; } /* finite? */

    /* ══════════════════════════════════════════════════════════════════════
     * Additional list ops (186-189)
     * ══════════════════════════════════════════════════════════════════════ */
    case 186: { /* list-ref */
        Value idx = vm_pop(vm), lst = vm_pop(vm);
        int n = (int)as_number(idx);
        while (n > 0 && lst.type == VAL_PAIR) { lst = vm->heap.objects[lst.as.ptr]->cons.cdr; n--; }
        vm_push(vm, (lst.type == VAL_PAIR) ? vm->heap.objects[lst.as.ptr]->cons.car : NIL_VAL);
        break; }
    case 187: { /* list-tail */
        Value idx = vm_pop(vm), lst = vm_pop(vm);
        int n = (int)as_number(idx);
        while (n > 0 && lst.type == VAL_PAIR) { lst = vm->heap.objects[lst.as.ptr]->cons.cdr; n--; }
        vm_push(vm, lst); break; }
    case 188: { /* last-pair */
        Value lst = vm_pop(vm);
        if (lst.type != VAL_PAIR) { vm_push(vm, lst); break; }
        while (vm->heap.objects[lst.as.ptr]->cons.cdr.type == VAL_PAIR)
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr;
        vm_push(vm, lst); break; }
    case 189: { /* list? */
        Value lst = vm_pop(vm);
        int is_list = (lst.type == VAL_NIL);
        if (!is_list) {
            Value cur = lst; int limit = 10000;
            while (cur.type == VAL_PAIR && limit-- > 0) cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            is_list = (cur.type == VAL_NIL);
        }
        vm_push(vm, BOOL_VAL(is_list)); break; }

    /* Complex ops 300-319 already implemented above (line ~218) */

    /* ══════════════════════════════════════════════════════════════════════
     * Math extensions (720-746)
     * ══════════════════════════════════════════════════════════════════════ */
    case 720: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(cosh(as_number(a)))); break; }
    case 721: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(sinh(as_number(a)))); break; }
    case 722: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(tanh(as_number(a)))); break; }
    case 726: { /* write-line */
        Value s = vm_pop(vm);
        if (s.type == VAL_STRING) { VmString* vs = (VmString*)vm->heap.objects[s.as.ptr]->opaque.ptr;
            if (vs) { printf("%.*s\n", vs->byte_len, vs->data); fflush(stdout); } }
        vm_push(vm, NIL_VAL); break; }
    case 728: { /* input-port? */
        Value a = vm_pop(vm);
        int is_ip = 0;
        if (a.type == VAL_PORT && a.as.ptr >= 0 && a.as.ptr < vm->heap.next_free) {
            VmPort* p = (VmPort*)vm->heap.objects[a.as.ptr]->opaque.ptr;
            is_ip = (p && p->dir == VM_PORT_INPUT);
        }
        vm_push(vm, BOOL_VAL(is_ip)); break; }
    case 729: { /* output-port? */
        Value a = vm_pop(vm);
        int is_op = 0;
        if (a.type == VAL_PORT && a.as.ptr >= 0 && a.as.ptr < vm->heap.next_free) {
            VmPort* p = (VmPort*)vm->heap.objects[a.as.ptr]->opaque.ptr;
            is_op = (p && p->dir == VM_PORT_OUTPUT);
        }
        vm_push(vm, BOOL_VAL(is_op)); break; }
    case 730: { /* port? */
        Value a = vm_pop(vm);
        vm_push(vm, BOOL_VAL(a.type == VAL_PORT)); break; }
    case 740: { /* type-of */
        Value a = vm_pop(vm);
        const char* t = "unknown";
        switch ((int)a.type) {
            case VAL_NIL: t = "nil"; break; case VAL_INT: t = "integer"; break;
            case VAL_FLOAT: t = "float"; break; case VAL_BOOL: t = "boolean"; break;
            case VAL_PAIR: t = "pair"; break; case VAL_CLOSURE: t = "procedure"; break;
            case VAL_STRING: t = "string"; break; case VAL_VECTOR: t = "vector"; break;
            case VAL_COMPLEX: t = "complex"; break; case VAL_RATIONAL: t = "rational"; break;
        }
        VmString* s = vm_string_from_cstr(&vm->heap.regions, t);
        if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); }
        else vm_push(vm, NIL_VAL); break; }
    case 743: { Value a = vm_pop(vm); double v = as_number(a);
        vm_push(vm, INT_VAL(v > 0 ? 1 : (v < 0 ? -1 : 0))); break; }
    case 745: { /* eye(n) — identity matrix as n×n tensor */
        Value n_val = vm_pop(vm);
        int n = (int)as_number(n_val);
        if (n <= 0 || n > 1024) { vm_push(vm, NIL_VAL); break; }
        int64_t shape[2] = {n, n};
        VmTensor* t = vm_tensor_zeros(&vm->heap.regions, shape, 2);
        if (!t) { vm_push(vm, NIL_VAL); break; }
        for (int i = 0; i < n; i++) t->data[i * n + i] = 1.0;
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t);
        break; }
    case 746: { /* linspace(start, stop, n) — n evenly spaced points */
        Value n_val = vm_pop(vm), stop_val = vm_pop(vm), start_val = vm_pop(vm);
        double s = as_number(start_val), e = as_number(stop_val);
        int n = (int)as_number(n_val);
        if (n <= 0 || n > 100000) { vm_push(vm, NIL_VAL); break; }
        int64_t shape[1] = {n};
        VmTensor* t = vm_tensor_zeros(&vm->heap.regions, shape, 1);
        if (!t) { vm_push(vm, NIL_VAL); break; }
        double step = (n > 1) ? (e - s) / (n - 1) : 0;
        for (int i = 0; i < n; i++) t->data[i] = s + i * step;
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t);
        break; }

    /* ══════════════════════════════════════════════════════════════════════
     * Higher-order native accelerated (900-910)
     * ══════════════════════════════════════════════════════════════════════ */
    case 900: { /* any(pred, list) */
        Value lst = vm_pop(vm), pred = vm_pop(vm);
        int found = 0;
        while (lst.type == VAL_PAIR) {
            Value car = vm->heap.objects[lst.as.ptr]->cons.car;
            Value r = vm_call_closure_from_native(vm, pred, &car, 1);
            if (is_truthy(r)) { found = 1; break; }
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr;
        }
        vm_push(vm, BOOL_VAL(found)); break; }
    case 901: { /* every(pred, list) */
        Value lst = vm_pop(vm), pred = vm_pop(vm);
        int all = 1;
        while (lst.type == VAL_PAIR) {
            Value car = vm->heap.objects[lst.as.ptr]->cons.car;
            Value r = vm_call_closure_from_native(vm, pred, &car, 1);
            if (!is_truthy(r)) { all = 0; break; }
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr;
        }
        vm_push(vm, BOOL_VAL(all)); break; }
    case 902: { /* find(pred, list) */
        Value lst = vm_pop(vm), pred = vm_pop(vm);
        while (lst.type == VAL_PAIR) {
            Value car = vm->heap.objects[lst.as.ptr]->cons.car;
            Value r = vm_call_closure_from_native(vm, pred, &car, 1);
            if (is_truthy(r)) { vm_push(vm, car); goto done_find; }
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr;
        }
        vm_push(vm, BOOL_VAL(0));
        done_find: break; }
    case 903: { /* take(n, list) */
        Value lst = vm_pop(vm), n_val = vm_pop(vm);
        int n = (int)as_number(n_val);
        Value result = NIL_VAL;
        while (n > 0 && lst.type == VAL_PAIR) {
            Value car = vm->heap.objects[lst.as.ptr]->cons.car;
            int32_t p = heap_alloc(&vm->heap); if (p < 0) break;
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = car; vm->heap.objects[p]->cons.cdr = result;
            result = PAIR_VAL(p);
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr; n--;
        }
        /* reverse */
        Value rev = NIL_VAL;
        while (result.type == VAL_PAIR) {
            Value car = vm->heap.objects[result.as.ptr]->cons.car;
            int32_t rp = heap_alloc(&vm->heap); if (rp < 0) break;
            vm->heap.objects[rp]->type = HEAP_CONS;
            vm->heap.objects[rp]->cons.car = car; vm->heap.objects[rp]->cons.cdr = rev;
            rev = PAIR_VAL(rp);
            result = vm->heap.objects[result.as.ptr]->cons.cdr;
        }
        vm_push(vm, rev); break; }
    case 904: { /* drop(n, list) */
        Value lst = vm_pop(vm), n_val = vm_pop(vm);
        int n = (int)as_number(n_val);
        while (n > 0 && lst.type == VAL_PAIR) { lst = vm->heap.objects[lst.as.ptr]->cons.cdr; n--; }
        vm_push(vm, lst); break; }
    case 905: { /* string-reverse */
        Value s_val = vm_pop(vm);
        if (s_val.type == VAL_STRING) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            if (s && s->byte_len > 0) {
                char* buf = (char*)vm_alloc(&vm->heap.regions, (size_t)(s->byte_len + 1));
                if (buf) { for (int i = 0; i < s->byte_len; i++) buf[i] = s->data[s->byte_len - 1 - i]; buf[s->byte_len] = 0;
                    VmString* r = vm_string_from_cstr(&vm->heap.regions, buf);
                    if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); break; } } }
        }
        vm_push(vm, s_val); break; }
    case 906: { /* string-repeat(str, n) */
        Value n_val = vm_pop(vm), s_val = vm_pop(vm);
        if (s_val.type == VAL_STRING) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            int n = (int)as_number(n_val);
            if (s && n > 0 && n < 10000 && s->byte_len > 0) {
                int total = s->byte_len * n;
                char* buf = (char*)vm_alloc(&vm->heap.regions, (size_t)(total + 1));
                if (buf) { for (int i = 0; i < n; i++) memcpy(buf + i * s->byte_len, s->data, s->byte_len); buf[total] = 0;
                    VmString* r = vm_string_from_cstr(&vm->heap.regions, buf);
                    if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); break; } } }
        }
        vm_push(vm, s_val); break; }
    case 907: { /* string-trim */
        Value s_val = vm_pop(vm);
        if (s_val.type == VAL_STRING) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            if (s) { int start = 0, end = s->byte_len;
                while (start < end && isspace((unsigned char)s->data[start])) start++;
                while (end > start && isspace((unsigned char)s->data[end-1])) end--;
                VmString* r = vm_string_new(&vm->heap.regions, s->data + start, end - start);
                if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); break; } } }
        vm_push(vm, s_val); break; }
    case 908: { /* string-split(str, delim) */
        Value d_val = vm_pop(vm), s_val = vm_pop(vm);
        Value result = NIL_VAL;
        if (s_val.type == VAL_STRING && d_val.type == VAL_STRING) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            VmString* d = (VmString*)vm->heap.objects[d_val.as.ptr]->opaque.ptr;
            if (s && d && d->byte_len > 0) {
                const char* p = s->data; int slen = s->byte_len, dlen = d->byte_len;
                while (1) {
                    const char* found = (slen >= dlen) ? strstr(p, d->data) : NULL;
                    int seg_len = found ? (int)(found - p) : (int)(s->data + slen - p);
                    VmString* seg = vm_string_new(&vm->heap.regions, p, seg_len);
                    if (seg) { int32_t sp2 = heap_alloc(&vm->heap); if (sp2 >= 0) {
                        vm->heap.objects[sp2]->type = HEAP_STRING; vm->heap.objects[sp2]->opaque.ptr = seg;
                        int32_t cp2 = heap_alloc(&vm->heap); if (cp2 >= 0) {
                            vm->heap.objects[cp2]->type = HEAP_CONS;
                            vm->heap.objects[cp2]->cons.car = (Value){.type = VAL_STRING, .as.ptr = sp2};
                            vm->heap.objects[cp2]->cons.cdr = result; result = PAIR_VAL(cp2); } } }
                    if (!found) break;
                    p = found + dlen;
                }
                /* reverse */
                Value rev2 = NIL_VAL;
                while (result.type == VAL_PAIR) {
                    Value car2 = vm->heap.objects[result.as.ptr]->cons.car;
                    int32_t rp2 = heap_alloc(&vm->heap); if (rp2 < 0) break;
                    vm->heap.objects[rp2]->type = HEAP_CONS;
                    vm->heap.objects[rp2]->cons.car = car2; vm->heap.objects[rp2]->cons.cdr = rev2;
                    rev2 = PAIR_VAL(rp2); result = vm->heap.objects[result.as.ptr]->cons.cdr;
                }
                vm_push(vm, rev2); break;
            }
        }
        vm_push(vm, NIL_VAL); break; }
    case 909: { /* string-join(list, delim) */
        Value d_val = vm_pop(vm), lst = vm_pop(vm);
        char buf[8192]; int pos = 0; int first = 1;
        const char* delim = "";  int dlen = 0;
        if (d_val.type == VAL_STRING) { VmString* ds = (VmString*)vm->heap.objects[d_val.as.ptr]->opaque.ptr;
            if (ds) { delim = ds->data; dlen = ds->byte_len; } }
        while (lst.type == VAL_PAIR && pos < 8000) {
            Value car = vm->heap.objects[lst.as.ptr]->cons.car;
            if (!first && dlen > 0 && pos + dlen < 8000) { memcpy(buf + pos, delim, dlen); pos += dlen; }
            first = 0;
            if (car.type == VAL_STRING) { VmString* cs = (VmString*)vm->heap.objects[car.as.ptr]->opaque.ptr;
                if (cs && pos + cs->byte_len < 8000) { memcpy(buf + pos, cs->data, cs->byte_len); pos += cs->byte_len; } }
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr;
        }
        buf[pos] = 0;
        VmString* r = vm_string_from_cstr(&vm->heap.regions, buf);
        if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); }
        else vm_push(vm, NIL_VAL); break; }

    /* ══════════════════════════════════════════════════════════════════════
     * System Information (1700-1719)
     * ══════════════════════════════════════════════════════════════════════ */

    case 1700: { /* os-type → "darwin", "linux", "windows", etc. */
#ifdef __APPLE__
        const char* os = "darwin";
#elif defined(_WIN32)
        const char* os = "windows";
#elif defined(__linux__)
        const char* os = "linux";
#elif defined(__FreeBSD__)
        const char* os = "freebsd";
#else
        const char* os = "unknown";
#endif
        VmString* s = vm_string_from_cstr(&vm->heap.regions, os);
        if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); }
        else { vm_push(vm, NIL_VAL); }
        break;
    }

    case 1701: { /* os-arch → "arm64", "x86_64", etc. */
#if defined(__aarch64__) || defined(_M_ARM64)
        const char* arch = "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
        const char* arch = "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
        const char* arch = "x86";
#elif defined(__riscv)
        const char* arch = "riscv64";
#else
        const char* arch = "unknown";
#endif
        VmString* s = vm_string_from_cstr(&vm->heap.regions, arch);
        if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); }
        else { vm_push(vm, NIL_VAL); }
        break;
    }

    case 1702: { /* home-directory → "/Users/foo" or "/home/foo" */
#ifndef ESHKOL_VM_WASM
        const char* home = getenv("HOME");
#ifdef _WIN32
        if (!home) home = getenv("USERPROFILE");
#endif
        if (home) {
            VmString* s = vm_string_from_cstr(&vm->heap.regions, home);
            if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
        }
#endif
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1703: { /* current-directory → cwd string */
#ifndef ESHKOL_VM_WASM
        char cwd[4096];
        if (getcwd(cwd, sizeof(cwd))) {
            VmString* s = vm_string_from_cstr(&vm->heap.regions, cwd);
            if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
        }
#endif
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1704: { /* set-current-directory!(path) → #t or #f */
#ifndef ESHKOL_VM_WASM
        Value path_val = vm_pop(vm);
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps && chdir(ps->data) == 0) { vm_push(vm, BOOL_VAL(1)); break; }
        }
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1705: { /* hostname → string */
#ifndef ESHKOL_VM_WASM
        char hostname[256];
        if (gethostname(hostname, sizeof(hostname)) == 0) {
            VmString* s = vm_string_from_cstr(&vm->heap.regions, hostname);
            if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
        }
#endif
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1706: { /* username → string */
#ifndef ESHKOL_VM_WASM
        const char* user = getenv("USER");
#ifdef _WIN32
        if (!user) user = getenv("USERNAME");
#endif
        if (user) {
            VmString* s = vm_string_from_cstr(&vm->heap.regions, user);
            if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
        }
#endif
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1707: { /* cpu-count → integer */
#ifndef ESHKOL_VM_WASM
#ifdef _WIN32
        SYSTEM_INFO si; GetSystemInfo(&si);
        vm_push(vm, INT_VAL(si.dwNumberOfProcessors));
#elif defined(_SC_NPROCESSORS_ONLN)
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        vm_push(vm, INT_VAL(n > 0 ? n : 1));
#else
        vm_push(vm, INT_VAL(1));
#endif
#else
        vm_push(vm, INT_VAL(1));
#endif
        break;
    }

    case 1708: { /* executable-exists?(name) → #t or #f */
        Value name_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (name_val.type == VAL_STRING) {
            VmString* ns = (VmString*)vm->heap.objects[name_val.as.ptr]->opaque.ptr;
            if (ns) {
                /* Search PATH for the executable */
                const char* path_env = getenv("PATH");
                if (path_env) {
                    char buf[4096];
                    strncpy(buf, path_env, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
                    char* dir = strtok(buf, ":");
                    while (dir) {
                        char full[4096];
                        snprintf(full, sizeof(full), "%s/%s", dir, ns->data);
                        if (access(full, X_OK) == 0) {
                            vm_push(vm, BOOL_VAL(1)); goto done_1708;
                        }
                        dir = strtok(NULL, ":");
                    }
                }
            }
        }
#else
        (void)name_val;
#endif
        vm_push(vm, BOOL_VAL(0));
#ifndef ESHKOL_VM_WASM
        done_1708:
#endif
        break;
    }

    case 1709: { /* current-time-ms → integer (milliseconds since epoch) */
#ifndef ESHKOL_VM_WASM
        struct timeval tv;
        gettimeofday(&tv, NULL);
        int64_t ms = (int64_t)tv.tv_sec * 1000 + (int64_t)tv.tv_usec / 1000;
        vm_push(vm, INT_VAL(ms));
#else
        vm_push(vm, INT_VAL(0));
#endif
        break;
    }

    case 1710: { /* getpid → integer */
#ifndef ESHKOL_VM_WASM
        vm_push(vm, INT_VAL((int64_t)getpid()));
#else
        vm_push(vm, INT_VAL(0));
#endif
        break;
    }

    case 1711: { /* sleep-ms(milliseconds) → void */
        Value ms_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        int64_t ms = (int64_t)as_number(ms_val);
        if (ms > 0) {
            struct timespec ts;
            ts.tv_sec = ms / 1000;
            ts.tv_nsec = (ms % 1000) * 1000000L;
            nanosleep(&ts, NULL);
        }
#else
        (void)ms_val;
#endif
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1712: { /* setenv(name, value) → #t or #f */
        Value val_v = vm_pop(vm), name_v = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (name_v.type == VAL_STRING && val_v.type == VAL_STRING) {
            VmString* ns = (VmString*)vm->heap.objects[name_v.as.ptr]->opaque.ptr;
            VmString* vs = (VmString*)vm->heap.objects[val_v.as.ptr]->opaque.ptr;
            if (ns && vs && setenv(ns->data, vs->data, 1) == 0) {
                vm_push(vm, BOOL_VAL(1)); break;
            }
        }
#else
        (void)val_v; (void)name_v;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1713: { /* unsetenv(name) → #t or #f */
        Value name_v = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (name_v.type == VAL_STRING) {
            VmString* ns = (VmString*)vm->heap.objects[name_v.as.ptr]->opaque.ptr;
            if (ns && unsetenv(ns->data) == 0) {
                vm_push(vm, BOOL_VAL(1)); break;
            }
        }
#else
        (void)name_v;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1714: { /* current-error-port → port for stderr */
        /* Return a pointer to the static vm_stderr_port (same pattern as current-output-port) */
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_PORT, VAL_PORT, &vm_stderr_port);
        break;
    }

    case 1715: { /* get-environment-variable(name) → string or #f */
        Value name_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (name_val.type == VAL_STRING) {
            VmString* ns = (VmString*)vm->heap.objects[name_val.as.ptr]->opaque.ptr;
            if (ns) {
                const char* val = getenv(ns->data);
                if (val) {
                    VmString* s = vm_string_from_cstr(&vm->heap.regions, val);
                    if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
                }
            }
        }
#else
        (void)name_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1716: { /* delete-file(path) — alias for 600 but with proper registration */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps && unlink(ps->data) == 0) { vm_push(vm, BOOL_VAL(1)); break; }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Path Manipulation (1720-1739)
     * ══════════════════════════════════════════════════════════════════════ */

    case 1720: { /* path-join(a, b) → string */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        if (a_val.type == VAL_STRING && b_val.type == VAL_STRING) {
            VmString* as = (VmString*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            VmString* bs = (VmString*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            if (as && bs) {
                char buf[4096];
                int alen = (int)strlen(as->data);
                if (alen > 0 && as->data[alen - 1] == '/')
                    snprintf(buf, sizeof(buf), "%s%s", as->data, bs->data);
                else
                    snprintf(buf, sizeof(buf), "%s/%s", as->data, bs->data);
                VmString* s = vm_string_from_cstr(&vm->heap.regions, buf);
                if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1721: { /* path-dirname(path) → string */
        Value path_val = vm_pop(vm);
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                char buf[4096];
                strncpy(buf, ps->data, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
                /* Find last / */
                char* last_slash = strrchr(buf, '/');
                if (last_slash) {
                    if (last_slash == buf) buf[1] = 0; /* root "/" */
                    else *last_slash = 0;
                } else {
                    strcpy(buf, ".");
                }
                VmString* s = vm_string_from_cstr(&vm->heap.regions, buf);
                if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1722: { /* path-basename(path) → string */
        Value path_val = vm_pop(vm);
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                const char* last_slash = strrchr(ps->data, '/');
                const char* base = last_slash ? last_slash + 1 : ps->data;
                VmString* s = vm_string_from_cstr(&vm->heap.regions, base);
                if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1723: { /* path-extname(path) → string (e.g., ".txt") or "" */
        Value path_val = vm_pop(vm);
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                const char* base = strrchr(ps->data, '/');
                if (!base) base = ps->data; else base++;
                const char* dot = strrchr(base, '.');
                const char* ext = (dot && dot != base) ? dot : "";
                VmString* s = vm_string_from_cstr(&vm->heap.regions, ext);
                if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1724: { /* path-is-absolute?(path) → #t or #f */
        Value path_val = vm_pop(vm);
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps && ps->data[0] == '/') {
                vm_push(vm, BOOL_VAL(1)); break;
            }
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1725: { /* path-normalize(path) → string with resolved . and .. */
        Value path_val = vm_pop(vm);
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                /* Split by /, resolve . and .., reassemble */
                char buf[4096];
                strncpy(buf, ps->data, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
                char* parts[256]; int nparts = 0;
                int absolute = (buf[0] == '/');
                char* tok = strtok(buf, "/");
                while (tok && nparts < 256) {
                    if (strcmp(tok, ".") == 0) { /* skip */ }
                    else if (strcmp(tok, "..") == 0) { if (nparts > 0) nparts--; }
                    else { parts[nparts++] = tok; }
                    tok = strtok(NULL, "/");
                }
                char result[4096];
                int pos = 0;
                if (absolute) result[pos++] = '/';
                for (int i = 0; i < nparts; i++) {
                    if (i > 0) result[pos++] = '/';
                    int len = (int)strlen(parts[i]);
                    if (pos + len >= 4095) break;
                    memcpy(result + pos, parts[i], len);
                    pos += len;
                }
                if (pos == 0) { result[0] = '.'; pos = 1; }
                result[pos] = 0;
                VmString* s = vm_string_from_cstr(&vm->heap.regions, result);
                if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1726: { /* realpath(path) → resolved absolute path string */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                char resolved[4096];
                if (realpath(ps->data, resolved)) {
                    VmString* s = vm_string_from_cstr(&vm->heap.regions, resolved);
                    if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
                }
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, NIL_VAL);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Filesystem Operations (1740-1769)
     * ══════════════════════════════════════════════════════════════════════ */

    case 1740: { /* file-size(path) → integer (bytes) or #f */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                struct stat st;
                if (stat(ps->data, &st) == 0) {
                    vm_push(vm, INT_VAL((int64_t)st.st_size));
                    break;
                }
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1741: { /* file-stat(path) → list: (size mtime-sec type-char) or #f
                  * type-char: 'f' = regular, 'd' = directory, 'l' = symlink, '?' = other */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                struct stat st;
                if (lstat(ps->data, &st) == 0) {
                    char type_ch = '?';
                    if (S_ISREG(st.st_mode)) type_ch = 'f';
                    else if (S_ISDIR(st.st_mode)) type_ch = 'd';
                    else if (S_ISLNK(st.st_mode)) type_ch = 'l';

                    /* Build list: (size mtime type-string) */
                    char type_str[2] = { type_ch, 0 };
                    VmString* ts = vm_string_from_cstr(&vm->heap.regions, type_str);
                    if (!ts) { vm_push(vm, BOOL_VAL(0)); break; }

                    /* Build cons cells: (type . nil) → (mtime . ...) → (size . ...) */
                    int32_t t_sp = heap_alloc(&vm->heap); if (t_sp < 0) { vm_push(vm, BOOL_VAL(0)); break; }
                    vm->heap.objects[t_sp]->type = HEAP_STRING;
                    vm->heap.objects[t_sp]->opaque.ptr = ts;

                    int32_t c3 = heap_alloc(&vm->heap); if (c3 < 0) { vm_push(vm, BOOL_VAL(0)); break; }
                    vm->heap.objects[c3]->type = HEAP_CONS;
                    vm->heap.objects[c3]->cons.car = (Value){.type = VAL_STRING, .as.ptr = t_sp};
                    vm->heap.objects[c3]->cons.cdr = NIL_VAL;

                    int32_t c2 = heap_alloc(&vm->heap); if (c2 < 0) { vm_push(vm, BOOL_VAL(0)); break; }
                    vm->heap.objects[c2]->type = HEAP_CONS;
                    vm->heap.objects[c2]->cons.car = INT_VAL((int64_t)st.st_mtime);
                    vm->heap.objects[c2]->cons.cdr = PAIR_VAL(c3);

                    int32_t c1 = heap_alloc(&vm->heap); if (c1 < 0) { vm_push(vm, BOOL_VAL(0)); break; }
                    vm->heap.objects[c1]->type = HEAP_CONS;
                    vm->heap.objects[c1]->cons.car = INT_VAL((int64_t)st.st_size);
                    vm->heap.objects[c1]->cons.cdr = PAIR_VAL(c2);

                    vm_push(vm, PAIR_VAL(c1));
                    break;
                }
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1742: { /* file-rename(old, new) → #t or #f */
        Value new_val = vm_pop(vm), old_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (old_val.type == VAL_STRING && new_val.type == VAL_STRING) {
            VmString* os = (VmString*)vm->heap.objects[old_val.as.ptr]->opaque.ptr;
            VmString* ns = (VmString*)vm->heap.objects[new_val.as.ptr]->opaque.ptr;
            if (os && ns && rename(os->data, ns->data) == 0) {
                vm_push(vm, BOOL_VAL(1)); break;
            }
        }
#else
        (void)new_val; (void)old_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1743: { /* file-copy(src, dst) → #t or #f */
        Value dst_val = vm_pop(vm), src_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (src_val.type == VAL_STRING && dst_val.type == VAL_STRING) {
            VmString* ss = (VmString*)vm->heap.objects[src_val.as.ptr]->opaque.ptr;
            VmString* ds = (VmString*)vm->heap.objects[dst_val.as.ptr]->opaque.ptr;
            if (ss && ds) {
                FILE* fin = fopen(ss->data, "rb");
                FILE* fout = fin ? fopen(ds->data, "wb") : NULL;
                if (fin && fout) {
                    char cbuf[8192];
                    size_t n;
                    while ((n = fread(cbuf, 1, sizeof(cbuf), fin)) > 0)
                        fwrite(cbuf, 1, n, fout);
                    fclose(fin); fclose(fout);
                    vm_push(vm, BOOL_VAL(1)); break;
                }
                if (fin) fclose(fin);
                if (fout) fclose(fout);
            }
        }
#else
        (void)dst_val; (void)src_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1744: { /* mkdir-recursive(path) → #t or #f */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                char buf[4096];
                strncpy(buf, ps->data, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
                /* Create each component */
                int ok = 1;
                for (char* p = buf + 1; *p; p++) {
                    if (*p == '/') {
                        *p = 0;
                        if (mkdir(buf, 0755) != 0 && errno != EEXIST) { ok = 0; break; }
                        *p = '/';
                    }
                }
                if (ok && mkdir(buf, 0755) != 0 && errno != EEXIST) ok = 0;
                if (ok || errno == EEXIST) { vm_push(vm, BOOL_VAL(1)); break; }
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1745: { /* file-chmod(path, mode) → #t or #f (mode is integer, e.g. 0o755 = 493) */
        Value mode_val = vm_pop(vm), path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps && chmod(ps->data, (mode_t)as_number(mode_val)) == 0) {
                vm_push(vm, BOOL_VAL(1)); break;
            }
        }
#else
        (void)mode_val; (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1746: { /* symlink-create(target, linkpath) → #t or #f */
        Value link_val = vm_pop(vm), target_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (target_val.type == VAL_STRING && link_val.type == VAL_STRING) {
            VmString* ts = (VmString*)vm->heap.objects[target_val.as.ptr]->opaque.ptr;
            VmString* ls = (VmString*)vm->heap.objects[link_val.as.ptr]->opaque.ptr;
            if (ts && ls && symlink(ts->data, ls->data) == 0) {
                vm_push(vm, BOOL_VAL(1)); break;
            }
        }
#else
        (void)link_val; (void)target_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1747: { /* symlink-read(linkpath) → target string or #f */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                char buf[4096];
                ssize_t len = readlink(ps->data, buf, sizeof(buf) - 1);
                if (len > 0) {
                    buf[len] = 0;
                    VmString* s = vm_string_from_cstr(&vm->heap.regions, buf);
                    if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
                }
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1748: { /* directory-walk(path) → flat list of all file paths (recursive) */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                /* BFS using a simple stack of directories to visit */
                Value result = NIL_VAL;
                char dirs[256][4096];
                int dir_count = 0;
                strncpy(dirs[0], ps->data, 4095); dirs[0][4095] = 0;
                dir_count = 1;
                int dir_idx = 0;
                while (dir_idx < dir_count && dir_count < 256) {
                    DIR* d = opendir(dirs[dir_idx]);
                    dir_idx++;
                    if (!d) continue;
                    struct dirent* ent;
                    while ((ent = readdir(d)) != NULL) {
                        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
                        char full[4096];
                        snprintf(full, sizeof(full), "%s/%s", dirs[dir_idx - 1], ent->d_name);
                        struct stat st;
                        if (stat(full, &st) == 0 && S_ISDIR(st.st_mode) && dir_count < 256) {
                            strncpy(dirs[dir_count], full, 4095);
                            dirs[dir_count][4095] = 0;
                            dir_count++;
                        }
                        /* Add to result list */
                        VmString* s = vm_string_from_cstr(&vm->heap.regions, full);
                        if (!s) continue;
                        int32_t sp = heap_alloc(&vm->heap); if (sp < 0) continue;
                        vm->heap.objects[sp]->type = HEAP_STRING;
                        vm->heap.objects[sp]->opaque.ptr = s;
                        int32_t cp = heap_alloc(&vm->heap); if (cp < 0) continue;
                        vm->heap.objects[cp]->type = HEAP_CONS;
                        vm->heap.objects[cp]->cons.car = (Value){.type = VAL_STRING, .as.ptr = sp};
                        vm->heap.objects[cp]->cons.cdr = result;
                        result = PAIR_VAL(cp);
                    }
                    closedir(d);
                }
                vm_push(vm, result);
                break;
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1749: { /* directory-delete-recursive(path) → #t or #f */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                /* Use system rm -rf (safe: path validated) */
                char cmd[4096];
                /* Basic path safety: reject if contains ; | & ` $ */
                int safe = 1;
                for (const char* c = ps->data; *c; c++) {
                    if (*c == ';' || *c == '|' || *c == '&' || *c == '`' || *c == '$') {
                        safe = 0; break;
                    }
                }
                if (safe) {
                    snprintf(cmd, sizeof(cmd), "rm -rf '%s'", ps->data);
                    if (system(cmd) == 0) { vm_push(vm, BOOL_VAL(1)); break; }
                }
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1750: { /* mkstemp(template) → (fd . path) or #f
                  * Template should end with XXXXXX */
        Value tmpl_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (tmpl_val.type == VAL_STRING) {
            VmString* ts = (VmString*)vm->heap.objects[tmpl_val.as.ptr]->opaque.ptr;
            if (ts) {
                char buf[4096];
                strncpy(buf, ts->data, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
                int fd = mkstemp(buf);
                if (fd >= 0) {
                    VmString* ps = vm_string_from_cstr(&vm->heap.regions, buf);
                    if (ps) {
                        int32_t sp = heap_alloc(&vm->heap); if (sp >= 0) {
                            vm->heap.objects[sp]->type = HEAP_STRING;
                            vm->heap.objects[sp]->opaque.ptr = ps;
                            int32_t cp = heap_alloc(&vm->heap); if (cp >= 0) {
                                vm->heap.objects[cp]->type = HEAP_CONS;
                                vm->heap.objects[cp]->cons.car = INT_VAL(fd);
                                vm->heap.objects[cp]->cons.cdr = (Value){.type = VAL_STRING, .as.ptr = sp};
                                vm_push(vm, PAIR_VAL(cp)); break;
                            }
                        }
                    }
                    close(fd);
                }
            }
        }
#else
        (void)tmpl_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1751: { /* mkdtemp(template) → path string or #f */
        Value tmpl_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (tmpl_val.type == VAL_STRING) {
            VmString* ts = (VmString*)vm->heap.objects[tmpl_val.as.ptr]->opaque.ptr;
            if (ts) {
                char buf[4096];
                strncpy(buf, ts->data, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
                if (mkdtemp(buf)) {
                    VmString* s = vm_string_from_cstr(&vm->heap.regions, buf);
                    if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
                }
            }
        }
#else
        (void)tmpl_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Shell Utilities (1770-1779)
     * ══════════════════════════════════════════════════════════════════════ */

    case 1770: { /* shell-quote(str) → single-quoted shell-safe string */
        Value str_val = vm_pop(vm);
        if (str_val.type == VAL_STRING) {
            VmString* ss = (VmString*)vm->heap.objects[str_val.as.ptr]->opaque.ptr;
            if (ss) {
                /* POSIX shell quoting: wrap in single quotes, escape internal ' as '\'' */
                char buf[8192];
                int pos = 0;
                buf[pos++] = '\'';
                for (const char* c = ss->data; *c && pos < 8180; c++) {
                    if (*c == '\'') {
                        buf[pos++] = '\''; buf[pos++] = '\\';
                        buf[pos++] = '\''; buf[pos++] = '\'';
                    } else {
                        buf[pos++] = *c;
                    }
                }
                buf[pos++] = '\'';
                buf[pos] = 0;
                VmString* s = vm_string_from_cstr(&vm->heap.regions, buf);
                if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1771: { /* shell-split(str) → list of strings (basic word splitting) */
        Value str_val = vm_pop(vm);
        if (str_val.type == VAL_STRING) {
            VmString* ss = (VmString*)vm->heap.objects[str_val.as.ptr]->opaque.ptr;
            if (ss) {
                Value result = NIL_VAL;
                char buf[4096];
                strncpy(buf, ss->data, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
                /* Collect words in reverse, then reverse the list */
                char* words[256]; int nwords = 0;
                char* p = buf;
                while (*p && nwords < 256) {
                    while (*p == ' ' || *p == '\t') p++;
                    if (!*p) break;
                    char quote = 0;
                    if (*p == '\'' || *p == '"') { quote = *p; p++; }
                    char* start = p;
                    if (quote) {
                        while (*p && *p != quote) p++;
                        if (*p) *p++ = 0;
                    } else {
                        while (*p && *p != ' ' && *p != '\t') p++;
                        if (*p) *p++ = 0;
                    }
                    words[nwords++] = start;
                }
                /* Build list in order */
                for (int i = nwords - 1; i >= 0; i--) {
                    VmString* ws = vm_string_from_cstr(&vm->heap.regions, words[i]);
                    if (!ws) continue;
                    int32_t sp = heap_alloc(&vm->heap); if (sp < 0) continue;
                    vm->heap.objects[sp]->type = HEAP_STRING;
                    vm->heap.objects[sp]->opaque.ptr = ws;
                    int32_t cp = heap_alloc(&vm->heap); if (cp < 0) continue;
                    vm->heap.objects[cp]->type = HEAP_CONS;
                    vm->heap.objects[cp]->cons.car = (Value){.type = VAL_STRING, .as.ptr = sp};
                    vm->heap.objects[cp]->cons.cdr = result;
                    result = PAIR_VAL(cp);
                }
                vm_push(vm, result);
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Process Management (1780-1799)
     * ══════════════════════════════════════════════════════════════════════ */

    /* ══════════════════════════════════════════════════════════════════════
     * Knowledge Base Extensions (1800-1809)
     * ══════════════════════════════════════════════════════════════════════ */

    case 1800: { /* kb-count(kb) → integer (number of facts) */
        Value kb_val = vm_pop(vm);
        if (kb_val.as.ptr >= 0 && vm->heap.objects[kb_val.as.ptr]->type == HEAP_KB) {
            VmKnowledgeBase* kb = (VmKnowledgeBase*)vm->heap.objects[kb_val.as.ptr]->opaque.ptr;
            if (kb) { vm_push(vm, INT_VAL(kb->n_facts)); break; }
        }
        vm_push(vm, INT_VAL(0));
        break;
    }

    case 1801: { /* kb-retract!(kb, fact) → #t or #f
                  * Remove first matching fact from KB */
        Value fact_val = vm_pop(vm), kb_val = vm_pop(vm);
        if (kb_val.as.ptr >= 0 && vm->heap.objects[kb_val.as.ptr]->type == HEAP_KB &&
            fact_val.as.ptr >= 0 && vm->heap.objects[fact_val.as.ptr]->type == HEAP_FACT) {
            VmKnowledgeBase* kb = (VmKnowledgeBase*)vm->heap.objects[kb_val.as.ptr]->opaque.ptr;
            VmFact* target = (VmFact*)vm->heap.objects[fact_val.as.ptr]->opaque.ptr;
            if (kb && target) {
                for (int i = 0; i < kb->n_facts; i++) {
                    if (kb->facts[i] == target) {
                        /* Shift remaining facts down */
                        for (int j = i; j < kb->n_facts - 1; j++)
                            kb->facts[j] = kb->facts[j + 1];
                        kb->n_facts--;
                        vm_push(vm, BOOL_VAL(1));
                        goto done_1801;
                    }
                }
            }
        }
        vm_push(vm, BOOL_VAL(0));
        done_1801:
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Factor Graph Extensions (1810-1819)
     * ══════════════════════════════════════════════════════════════════════ */

    case 1810: { /* fg-marginal(fg, var-idx) → tensor of belief probabilities */
        Value idx_val = vm_pop(vm), fg_val = vm_pop(vm);
        if (fg_val.as.ptr >= 0 && vm->heap.objects[fg_val.as.ptr]->type == HEAP_FACTOR_GRAPH) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            int var = (int)as_number(idx_val);
            if (fg && var >= 0 && var < fg->num_vars) {
                int dim = fg->var_dims[var];
                /* Convert log-beliefs to probabilities via softmax */
                double probs[64];
                double max_b = fg->beliefs[var][0];
                for (int i = 1; i < dim && i < 64; i++)
                    if (fg->beliefs[var][i] > max_b) max_b = fg->beliefs[var][i];
                double sum = 0;
                for (int i = 0; i < dim && i < 64; i++) {
                    probs[i] = exp(fg->beliefs[var][i] - max_b);
                    sum += probs[i];
                }
                if (sum > 0) for (int i = 0; i < dim && i < 64; i++) probs[i] /= sum;
                int64_t shape[1] = { dim };
                VmTensor* t = vm_tensor_from_data(&vm->heap.regions, probs, shape, 1);
                if (t) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1811: { /* fg-entropy(fg, var-idx) → scalar entropy H = -Σ p*log(p) */
        Value idx_val = vm_pop(vm), fg_val = vm_pop(vm);
        if (fg_val.as.ptr >= 0 && vm->heap.objects[fg_val.as.ptr]->type == HEAP_FACTOR_GRAPH) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            int var = (int)as_number(idx_val);
            if (fg && var >= 0 && var < fg->num_vars) {
                int dim = fg->var_dims[var];
                /* Softmax to get probabilities */
                double max_b = fg->beliefs[var][0];
                for (int i = 1; i < dim; i++)
                    if (fg->beliefs[var][i] > max_b) max_b = fg->beliefs[var][i];
                double sum = 0;
                double probs[64];
                for (int i = 0; i < dim && i < 64; i++) {
                    probs[i] = exp(fg->beliefs[var][i] - max_b);
                    sum += probs[i];
                }
                /* H = -Σ p*log(p) */
                double entropy = 0;
                for (int i = 0; i < dim && i < 64; i++) {
                    double p = probs[i] / sum;
                    if (p > 1e-15) entropy -= p * log(p);
                }
                vm_push(vm, FLOAT_VAL(entropy));
                break;
            }
        }
        vm_push(vm, FLOAT_VAL(0));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Tensor/KB Persistence (1820-1829)
     * Binary format: [magic:4][version:4][ndims:4][shape:ndims*8][data:total*8]
     * ══════════════════════════════════════════════════════════════════════ */

#define TENSOR_FILE_MAGIC 0x45534B54 /* "ESKT" */

    case 1820: { /* tensor-save(path, tensor) → #t or #f */
        Value tensor_val = vm_pop(vm), path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING && tensor_val.type == VAL_TENSOR) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            VmTensor* t = (VmTensor*)vm->heap.objects[tensor_val.as.ptr]->opaque.ptr;
            if (ps && t && t->data) {
                FILE* f = fopen(ps->data, "wb");
                if (f) {
                    uint32_t magic = TENSOR_FILE_MAGIC;
                    uint32_t version = 1;
                    uint32_t ndims = (uint32_t)t->n_dims;
                    fwrite(&magic, 4, 1, f);
                    fwrite(&version, 4, 1, f);
                    fwrite(&ndims, 4, 1, f);
                    for (int i = 0; i < t->n_dims; i++) {
                        int64_t dim = t->shape[i];
                        fwrite(&dim, 8, 1, f);
                    }
                    fwrite(t->data, sizeof(double), (size_t)t->total, f);
                    fclose(f);
                    vm_push(vm, BOOL_VAL(1));
                    break;
                }
            }
        }
#else
        (void)tensor_val; (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1821: { /* tensor-load(path) → tensor or #f */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                FILE* f = fopen(ps->data, "rb");
                if (f) {
                    uint32_t magic, version, ndims;
                    if (fread(&magic, 4, 1, f) == 1 && magic == TENSOR_FILE_MAGIC &&
                        fread(&version, 4, 1, f) == 1 && version == 1 &&
                        fread(&ndims, 4, 1, f) == 1 && ndims > 0 && ndims <= 8) {
                        int64_t shape[8];
                        int ok = 1;
                        for (uint32_t i = 0; i < ndims; i++) {
                            if (fread(&shape[i], 8, 1, f) != 1) { ok = 0; break; }
                        }
                        if (ok) {
                            int64_t total = 1;
                            for (uint32_t i = 0; i < ndims; i++) total *= shape[i];
                            VmTensor* t = vm_tensor_new(&vm->heap.regions, shape, (int)ndims);
                            if (t && t->data && (int64_t)fread(t->data, sizeof(double), (size_t)total, f) == total) {
                                fclose(f);
                                VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t);
                                break;
                            }
                        }
                    }
                    fclose(f);
                }
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1822: { /* kb-save(path, kb) → #t or #f
                  * Serializes KB: writes fact count + predicate hashes + arities as binary.
                  * For facts with datum (list), writes the list repr.
                  * Format: [magic:4][version:4][n_facts:4][per fact: predicate_hash:8, arity:4, datum_ptr:4] */
        Value kb_val = vm_pop(vm), path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING && kb_val.as.ptr >= 0 &&
            vm->heap.objects[kb_val.as.ptr]->type == HEAP_KB) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            VmKnowledgeBase* kb = (VmKnowledgeBase*)vm->heap.objects[kb_val.as.ptr]->opaque.ptr;
            if (ps && kb) {
                FILE* f = fopen(ps->data, "wb");
                if (f) {
                    uint32_t magic = 0x45534B42; /* "ESKB" */
                    uint32_t version = 1;
                    uint32_t nf = (uint32_t)kb->n_facts;
                    fwrite(&magic, 4, 1, f);
                    fwrite(&version, 4, 1, f);
                    fwrite(&nf, 4, 1, f);
                    for (int i = 0; i < kb->n_facts; i++) {
                        VmFact* fact = kb->facts[i];
                        if (!fact) { uint64_t z = 0; fwrite(&z, 8, 1, f); uint32_t za = 0; fwrite(&za, 4, 1, f); continue; }
                        fwrite(&fact->predicate, 8, 1, f);
                        uint32_t ar = (uint32_t)fact->arity;
                        fwrite(&ar, 4, 1, f);
                        /* Write each arg's type byte + data */
                        for (int j = 0; j < fact->arity; j++) {
                            fwrite(&fact->args[j].type, 1, 1, f);
                            fwrite(&fact->args[j].data, 8, 1, f);
                        }
                    }
                    fclose(f);
                    vm_push(vm, BOOL_VAL(1));
                    break;
                }
            }
        }
#else
        (void)kb_val; (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

#undef TENSOR_FILE_MAGIC

    /* ══════════════════════════════════════════════════════════════════════
     * Image I/O (1850-1859) — stb_image based
     * ══════════════════════════════════════════════════════════════════════ */

    case 1850: { /* image-read(path) → tensor (H,W,C) or #f */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                int w, h, c;
                extern double* eshkol_image_read(const char*, int*, int*, int*);
                double* data = eshkol_image_read(ps->data, &w, &h, &c);
                if (data) {
                    int64_t shape[3] = { h, w, c };
                    int ndims = (c == 1) ? 2 : 3;
                    if (c == 1) { shape[0] = h; shape[1] = w; }
                    VmTensor* t = vm_tensor_from_data(&vm->heap.regions, data, shape, ndims);
                    free(data);
                    if (t) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t); break; }
                }
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1851: { /* image-write(path, tensor, format) → #t or #f */
        Value fmt_val = vm_pop(vm), tensor_val = vm_pop(vm), path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING && tensor_val.type == VAL_TENSOR) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            VmTensor* t = (VmTensor*)vm->heap.objects[tensor_val.as.ptr]->opaque.ptr;
            const char* fmt = "png";
            if (fmt_val.type == VAL_STRING) {
                VmString* fs = (VmString*)vm->heap.objects[fmt_val.as.ptr]->opaque.ptr;
                if (fs) fmt = fs->data;
            }
            if (ps && t && t->data && t->n_dims >= 2) {
                int h = (int)t->shape[0], w = (int)t->shape[1];
                int c = (t->n_dims >= 3) ? (int)t->shape[2] : 1;
                extern int eshkol_image_write(const char*, const double*, int, int, int, const char*);
                if (eshkol_image_write(ps->data, t->data, w, h, c, fmt) == 0) {
                    vm_push(vm, BOOL_VAL(1)); break;
                }
            }
        }
#else
        (void)fmt_val; (void)tensor_val; (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1852: { /* image-to-grayscale(tensor) → tensor (H,W) or #f */
        Value tensor_val = vm_pop(vm);
        if (tensor_val.type == VAL_TENSOR) {
            VmTensor* t = (VmTensor*)vm->heap.objects[tensor_val.as.ptr]->opaque.ptr;
            if (t && t->data && t->n_dims >= 2) {
                int h = (int)t->shape[0], w = (int)t->shape[1];
                int c = (t->n_dims >= 3) ? (int)t->shape[2] : 1;
                extern double* eshkol_image_to_grayscale(const double*, int, int, int);
                double* gray = eshkol_image_to_grayscale(t->data, w, h, c);
                if (gray) {
                    int64_t shape[2] = { h, w };
                    VmTensor* gt = vm_tensor_from_data(&vm->heap.regions, gray, shape, 2);
                    free(gray);
                    if (gt) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, gt); break; }
                }
            }
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1853: { /* image-resize(tensor, new-h, new-w) → tensor or #f */
        Value nw_val = vm_pop(vm), nh_val = vm_pop(vm), tensor_val = vm_pop(vm);
        if (tensor_val.type == VAL_TENSOR) {
            VmTensor* t = (VmTensor*)vm->heap.objects[tensor_val.as.ptr]->opaque.ptr;
            int new_h = (int)as_number(nh_val), new_w = (int)as_number(nw_val);
            if (t && t->data && t->n_dims >= 2 && new_h > 0 && new_w > 0) {
                int h = (int)t->shape[0], w = (int)t->shape[1];
                int c = (t->n_dims >= 3) ? (int)t->shape[2] : 1;
                extern double* eshkol_image_resize(const double*, int, int, int, int, int);
                double* resized = eshkol_image_resize(t->data, w, h, c, new_w, new_h);
                if (resized) {
                    int64_t shape[3] = { new_h, new_w, c };
                    int ndims = (c == 1) ? 2 : 3;
                    if (c == 1) { shape[0] = new_h; shape[1] = new_w; }
                    VmTensor* rt = vm_tensor_from_data(&vm->heap.regions, resized, shape, ndims);
                    free(resized);
                    if (rt) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, rt); break; }
                }
            }
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1840: { /* reverse-gradient(f, point) → tensor of gradients
                  * Uses reverse-mode AD via Wengert tape tracing.
                  * Activates the tape, calls f with traced inputs,
                  * runs backward pass, returns gradient tensor.
                  * Single backward pass → O(1) regardless of input dimension. */
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);

        double point[64];
        int n = 0;
        if (x_val.type == VAL_PAIR) {
            Value cur = x_val;
            while (cur.type == VAL_PAIR && n < 64) {
                point[n++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
        } else if (x_val.type == VAL_TENSOR && x_val.as.ptr >= 0) {
            VmTensor* t = (VmTensor*)vm->heap.objects[x_val.as.ptr]->opaque.ptr;
            if (t && t->data) {
                n = (int)(t->total < 64 ? t->total : 64);
                for (int i = 0; i < n; i++) point[i] = t->data[i];
            }
        } else {
            point[0] = as_number(x_val);
            n = 1;
        }

        if (n == 0) { vm_push(vm, FLOAT_VAL(0)); break; }

        /* Create tape and variable nodes */
        AdTape* tape = ad_tape_new(&vm->heap.regions);
        if (!tape) { vm_push(vm, FLOAT_VAL(0)); break; }

        int var_nodes[64];
        Value args[64];
        for (int i = 0; i < n; i++) {
            var_nodes[i] = ad_var(tape, point[i]);
            args[i] = FLOAT_VAL(point[i]);
        }

        /* Activate tape tracing on VM */
        void* saved_tape = vm->active_tape;
        vm->active_tape = tape;

        /* Set up ad_node_map for the argument slots.
         * The closure bridge pushes: closure, arg0, arg1, ..., argN-1
         * at stack positions sp, sp+1, ..., sp+N.
         * After frame setup, locals start at fp = sp+N-N = sp.
         * So arg[i] is at stack position (current_sp + 1 + i). */
        int base_sp = vm->sp + 1; /* +1 for closure push */
        for (int i = 0; i < n; i++) {
            if (base_sp + i < STACK_SIZE)
                vm->ad_node_map[base_sp + i] = var_nodes[i];
        }

        /* Call f(x1, x2, ..., xn) — arithmetic will record on tape */
        Value result = vm_call_closure_from_native(vm, f_val, args, n);

        /* Capture result's tape node (it's at the return value position) */
        /* The closure bridge captures result from stack[sp-1] before restoring sp.
         * At that point, ad_node_map[sp-1] holds the result node. But since sp
         * was already restored by the bridge, we need to find the output node.
         * The last node on the tape IS the output (tape nodes are appended in order). */
        int output_node = tape->len - 1;

        /* Deactivate tape */
        vm->active_tape = saved_tape;

        if (output_node < 0) {
            /* Function didn't produce any tape operations — constant function */
            int64_t shape[1] = { n };
            VmTensor* zt = vm_tensor_zeros(&vm->heap.regions, shape, 1);
            if (zt) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, zt); }
            else { vm_push(vm, NIL_VAL); }
            break;
        }

        /* Run backward pass */
        ad_backward(tape, output_node);

        /* Collect gradients from variable nodes */
        if (n == 1) {
            vm_push(vm, FLOAT_VAL(ad_gradient(tape, var_nodes[0])));
        } else {
            double grads[64];
            for (int i = 0; i < n; i++)
                grads[i] = ad_gradient(tape, var_nodes[i]);
            int64_t shape[1] = { n };
            VmTensor* t = vm_tensor_from_data(&vm->heap.regions, grads, shape, 1);
            if (t) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t); }
            else { vm_push(vm, NIL_VAL); }
        }
        break;
    }

    case 1830: { /* tensor-from-stack(count, v1, v2, ..., vN) → tensor
                  * Internal: compiler emits this for all-numeric #(...) literals.
                  * Stack: [count, val0, val1, ..., valN-1] (count pushed first) */
        /* Pop values in reverse (valN-1 is TOS) */
        int n = 0;
        /* We need to find the count on the stack. The compiler pushes:
         * CONST(count), CONST(v0), CONST(v1), ..., CONST(vN-1), NATIVE_CALL
         * So TOS-0 through TOS-(N-1) are values, TOS-N is count. */
        /* Strategy: peek backwards to find the int count */
        double vals[1024];
        int found = 0;
        for (int try_n = 0; try_n < 1024 && try_n < vm->sp; try_n++) {
            int count_pos = vm->sp - try_n - 1;
            if (count_pos >= 0 && vm->stack[count_pos].type == VAL_INT) {
                int candidate = (int)vm->stack[count_pos].as.i;
                if (candidate == try_n && candidate >= 0 && candidate < 1024) {
                    n = candidate;
                    found = 1;
                    break;
                }
            }
        }
        if (found && n > 0) {
            /* Pop the n values */
            for (int i = n - 1; i >= 0; i--)
                vals[i] = as_number(vm_pop(vm));
            vm_pop(vm); /* pop count */
            int64_t shape[1] = { n };
            VmTensor* t = vm_tensor_from_data(&vm->heap.regions, vals, shape, 1);
            if (t) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t); break; }
        } else if (found && n == 0) {
            vm_pop(vm); /* pop count */
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1780: { /* process-spawn(cmd, args-list, env-alist) → pid or #f
                  * cmd: string, args: list of strings, env: alist of (name . value) or #f
                  * Returns child PID on success, #f on failure */
        Value env_val = vm_pop(vm), args_val = vm_pop(vm), cmd_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (cmd_val.type == VAL_STRING) {
            VmString* cs = (VmString*)vm->heap.objects[cmd_val.as.ptr]->opaque.ptr;
            if (cs) {
                /* Build argv array */
                char* argv[256];
                int argc_local = 0;
                argv[argc_local++] = cs->data;
                Value acur = args_val;
                while (acur.type == VAL_PAIR && argc_local < 255) {
                    Value elem = vm->heap.objects[acur.as.ptr]->cons.car;
                    if (elem.type == VAL_STRING) {
                        VmString* es = (VmString*)vm->heap.objects[elem.as.ptr]->opaque.ptr;
                        if (es) argv[argc_local++] = es->data;
                    }
                    acur = vm->heap.objects[acur.as.ptr]->cons.cdr;
                }
                argv[argc_local] = NULL;

                /* Build environment if provided */
                char* envp_buf[256];
                char env_strs[256][512];
                char** envp = NULL;
                int envc = 0;
                if (env_val.type == VAL_PAIR) {
                    Value ecur = env_val;
                    while (ecur.type == VAL_PAIR && envc < 255) {
                        Value pair = vm->heap.objects[ecur.as.ptr]->cons.car;
                        if (pair.type == VAL_PAIR) {
                            Value key = vm->heap.objects[pair.as.ptr]->cons.car;
                            Value val = vm->heap.objects[pair.as.ptr]->cons.cdr;
                            if (key.type == VAL_STRING && val.type == VAL_STRING) {
                                VmString* ks = (VmString*)vm->heap.objects[key.as.ptr]->opaque.ptr;
                                VmString* vs = (VmString*)vm->heap.objects[val.as.ptr]->opaque.ptr;
                                if (ks && vs) {
                                    snprintf(env_strs[envc], 512, "%s=%s", ks->data, vs->data);
                                    envp_buf[envc] = env_strs[envc];
                                    envc++;
                                }
                            }
                        }
                        ecur = vm->heap.objects[ecur.as.ptr]->cons.cdr;
                    }
                    envp_buf[envc] = NULL;
                    envp = envp_buf;
                }

                pid_t pid = fork();
                if (pid == 0) {
                    /* Child */
                    if (envp) execve(argv[0], argv, envp);
                    else execvp(argv[0], argv);
                    _exit(127);
                } else if (pid > 0) {
                    vm_push(vm, INT_VAL((int64_t)pid));
                    break;
                }
            }
        }
#else
        (void)env_val; (void)args_val; (void)cmd_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1781: { /* process-wait(pid) → exit-status integer */
        Value pid_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        int status = 0;
        pid_t pid = (pid_t)as_number(pid_val);
        if (waitpid(pid, &status, 0) >= 0) {
            vm_push(vm, INT_VAL(WIFEXITED(status) ? WEXITSTATUS(status) : -1));
        } else
#else
        (void)pid_val;
#endif
        { vm_push(vm, INT_VAL(-1)); }
        break;
    }

    case 1782: { /* process-kill(pid, signal) → #t or #f */
        Value sig_val = vm_pop(vm), pid_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        pid_t pid = (pid_t)as_number(pid_val);
        int sig = (int)as_number(sig_val);
        if (kill(pid, sig) == 0) { vm_push(vm, BOOL_VAL(1)); break; }
#else
        (void)sig_val; (void)pid_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1783: { /* io-poll(fd-list, timeout-ms) → list of ready fds or empty
                  * fd-list: list of integer file descriptors
                  * timeout-ms: integer (-1 = block forever, 0 = non-blocking) */
        Value timeout_val = vm_pop(vm), fds_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        /* Count fds */
        int nfds = 0;
        Value cur = fds_val;
        while (cur.type == VAL_PAIR && nfds < 256) {
            nfds++;
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        if (nfds > 0) {
            struct pollfd pfds[256];
            cur = fds_val;
            for (int i = 0; i < nfds; i++) {
                pfds[i].fd = (int)as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                pfds[i].events = POLLIN;
                pfds[i].revents = 0;
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
            int timeout_ms = (int)as_number(timeout_val);
            int ret = poll(pfds, (nfds_t)nfds, timeout_ms);
            if (ret > 0) {
                Value result = NIL_VAL;
                for (int i = nfds - 1; i >= 0; i--) {
                    if (pfds[i].revents & (POLLIN | POLLHUP | POLLERR)) {
                        int32_t cp = heap_alloc(&vm->heap); if (cp < 0) continue;
                        vm->heap.objects[cp]->type = HEAP_CONS;
                        vm->heap.objects[cp]->cons.car = INT_VAL(pfds[i].fd);
                        vm->heap.objects[cp]->cons.cdr = result;
                        result = PAIR_VAL(cp);
                    }
                }
                vm_push(vm, result);
                break;
            }
        }
#else
        (void)timeout_val; (void)fds_val;
#endif
        vm_push(vm, NIL_VAL);
        break;
    }

    default:
        /* Check geometric manifold operations (804-843) */
        if (fid >= 804 && fid <= 843) {
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
