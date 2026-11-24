define %eshkol_tagged_value @gram(%eshkol_tagged_value %x) {
entry:
  %raise_msg_buf = alloca i8, i64 256, align 1
  call void @eshkol_stack_guard_check()
  %0 = extractvalue %eshkol_tagged_value %x, 0
  %1 = icmp uge i8 %0, 8
  %2 = and i8 %0, 15
  %3 = select i1 %1, i8 %0, i8 %2
  %4 = icmp eq i8 %3, 9
  br i1 %4, label %transpose_callable, label %transpose_plain

transpose_callable:                               ; preds = %entry
  %5 = extractvalue %eshkol_tagged_value %x, 4
  %data_ptr = inttoptr i64 %5 to ptr
  %header_ptr = getelementptr i8, ptr %data_ptr, i64 -8
  %subtype = load i8, ptr %header_ptr, align 1
  %callable_subtype_match = icmp eq i8 %subtype, 2
  br i1 %callable_subtype_match, label %transpose_ad, label %transpose_plain

transpose_ad:                                     ; preds = %transpose_callable
  %6 = extractvalue %eshkol_tagged_value %x, 4
  %7 = inttoptr i64 %6 to ptr
  %8 = getelementptr inbounds nuw %ad_node, ptr %7, i32 0, i32 6
  %9 = load ptr, ptr %8, align 8
  %10 = icmp ne ptr %9, null
  br i1 %10, label %transpose_dense, label %transpose_plain

transpose_dense:                                  ; preds = %transpose_ad
  %11 = getelementptr inbounds nuw %ad_node, ptr %7, i32 0, i32 13
  %12 = load ptr, ptr %11, align 8
  %13 = getelementptr inbounds nuw %ad_node, ptr %7, i32 0, i32 14
  %14 = load i64, ptr %13, align 8
  %transpose_ad_total = call i64 @eshkol_ad_node_total_elements(ptr %7)
  %arena = call ptr @eshkol_current_arena()
  %ad_home_arena = call ptr @eshkol_ad_home_arena(ptr %arena)
  %transpose_ad_view = call ptr @arena_allocate_tensor_with_header(ptr %ad_home_arena)
  %15 = getelementptr inbounds nuw %tensor, ptr %transpose_ad_view, i32 0, i32 0
  store ptr %12, ptr %15, align 8
  %16 = getelementptr inbounds nuw %tensor, ptr %transpose_ad_view, i32 0, i32 1
  store i64 %14, ptr %16, align 8
  %17 = getelementptr inbounds nuw %tensor, ptr %transpose_ad_view, i32 0, i32 2
  store ptr %9, ptr %17, align 8
  %18 = getelementptr inbounds nuw %tensor, ptr %transpose_ad_view, i32 0, i32 3
  store i64 %transpose_ad_total, ptr %18, align 8
  %19 = getelementptr inbounds nuw %tensor, ptr %transpose_ad_view, i32 0, i32 4
  store i64 0, ptr %19, align 8
  br label %transpose_input_join

transpose_plain:                                  ; preds = %transpose_ad, %transpose_callable, %entry
  call void @eshkol_set_error_location(ptr @.str.32, i32 2, i32 29)
  %tensor_operand_slot = alloca %eshkol_tagged_value, align 8
  store %eshkol_tagged_value %x, ptr %tensor_operand_slot, align 8
  %tensor_ptr_checked = call ptr @eshkol_tensor_operand_checked(ptr %tensor_operand_slot, ptr @.str.33)
  br label %transpose_input_join

transpose_input_join:                             ; preds = %transpose_plain, %transpose_dense
  %transpose_input = phi ptr [ %transpose_ad_view, %transpose_dense ], [ %tensor_ptr_checked, %transpose_plain ]
  %transpose_parent = phi ptr [ %7, %transpose_dense ], [ null, %transpose_plain ]
  %trans_result = alloca %eshkol_tagged_value, align 8
  br label %transpose_tensor

transpose_tensor:                                 ; preds = %transpose_input_join
  %20 = getelementptr inbounds nuw %tensor, ptr %transpose_input, i32 0, i32 0
  %21 = load ptr, ptr %20, align 8
  %22 = getelementptr inbounds nuw %tensor, ptr %transpose_input, i32 0, i32 1
  %23 = load i64, ptr %22, align 8
  %24 = icmp ne i64 %23, 2
  br i1 %24, label %transpose_dims_err, label %transpose_dims_ok

transpose_exit:                                   ; preds = %trans_row_exit
  %25 = icmp ne ptr %transpose_parent, null
  br i1 %25, label %transpose_record, label %transpose_plain_result

transpose_dims_ok:                                ; preds = %transpose_tensor
  %26 = getelementptr inbounds nuw %tensor, ptr %transpose_input, i32 0, i32 2
  %27 = load ptr, ptr %26, align 8
  %28 = load i64, ptr %21, align 8
  %29 = getelementptr i64, ptr %21, i64 1
  %30 = load i64, ptr %29, align 8
  %arena1 = call ptr @eshkol_current_arena()
  %ad_home_arena2 = call ptr @eshkol_ad_home_arena(ptr %arena1)
  %new_tensor = call ptr @arena_allocate_tensor_with_header(ptr %ad_home_arena2)
  %new_dims = call ptr @arena_allocate(ptr %ad_home_arena2, i64 16)
  %31 = getelementptr i64, ptr %new_dims, i64 0
  store i64 %30, ptr %31, align 8
  %32 = getelementptr i64, ptr %new_dims, i64 1
  store i64 %28, ptr %32, align 8
  %checked_total = call i64 @eshkol_tensor_shape_total(ptr %new_dims, i64 2)
  %33 = icmp slt i64 %checked_total, 0
  br i1 %33, label %tensor_shape_err, label %tensor_shape_ok

transpose_dims_err:                               ; preds = %transpose_tensor
  %34 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %raise_msg_buf, i64 256, ptr @44, i64 %23)
  %35 = call ptr @eshkol_make_exception_with_header(i32 0, ptr %raise_msg_buf)
  call void @eshkol_raise(ptr %35)
  unreachable

tensor_shape_ok:                                  ; preds = %transpose_dims_ok
  call void @eshkol_enforce_tensor_elements(i64 %checked_total)
  %36 = mul i64 %checked_total, 8
  %new_elems = call ptr @arena_allocate(ptr %ad_home_arena2, i64 %36)
  %37 = getelementptr inbounds nuw %tensor, ptr %new_tensor, i32 0, i32 0
  store ptr %new_dims, ptr %37, align 8
  %38 = getelementptr inbounds nuw %tensor, ptr %new_tensor, i32 0, i32 1
  store i64 2, ptr %38, align 8
  %39 = getelementptr inbounds nuw %tensor, ptr %new_tensor, i32 0, i32 2
  store ptr %new_elems, ptr %39, align 8
  %40 = getelementptr inbounds nuw %tensor, ptr %new_tensor, i32 0, i32 3
  store i64 %checked_total, ptr %40, align 8
  %41 = getelementptr inbounds nuw %tensor, ptr %new_tensor, i32 0, i32 2
  %42 = load ptr, ptr %41, align 8
  %trans_i = alloca i64, align 8
  %trans_j = alloca i64, align 8
  store i64 0, ptr %trans_i, align 8
  br label %trans_row_cond

tensor_shape_err:                                 ; preds = %transpose_dims_ok
  %43 = call ptr @eshkol_make_exception_with_header(i32 0, ptr @45)
  call void @eshkol_raise(ptr %43)
  unreachable

trans_row_cond:                                   ; preds = %trans_col_exit, %tensor_shape_ok
  %44 = load i64, ptr %trans_i, align 8
  %45 = icmp ult i64 %44, %28
  br i1 %45, label %trans_row_body, label %trans_row_exit

trans_row_body:                                   ; preds = %trans_row_cond
  store i64 0, ptr %trans_j, align 8
  br label %trans_col_cond

trans_col_cond:                                   ; preds = %trans_col_body, %trans_row_body
  %46 = load i64, ptr %trans_j, align 8
  %47 = icmp ult i64 %46, %30
  br i1 %47, label %trans_col_body, label %trans_col_exit

trans_col_body:                                   ; preds = %trans_col_cond
  %48 = mul i64 %44, %30
  %49 = add i64 %48, %46
  %50 = mul i64 %46, %28
  %51 = add i64 %50, %44
  %52 = getelementptr i64, ptr %27, i64 %49
  %53 = load i64, ptr %52, align 8
  %54 = getelementptr i64, ptr %42, i64 %51
  store i64 %53, ptr %54, align 8
  %55 = add i64 %46, 1
  store i64 %55, ptr %trans_j, align 8
  br label %trans_col_cond

trans_col_exit:                                   ; preds = %trans_col_cond
  %56 = add i64 %44, 1
  store i64 %56, ptr %trans_i, align 8
  br label %trans_row_cond

trans_row_exit:                                   ; preds = %trans_row_cond
  %57 = ptrtoint ptr %new_tensor to i64
  %58 = insertvalue %eshkol_tagged_value { i8 8, i8 0, i16 0, i32 0, i64 undef }, i64 %57, 4
  store %eshkol_tagged_value %58, ptr %trans_result, align 8
  br label %transpose_exit

transpose_record:                                 ; preds = %transpose_exit
  %59 = getelementptr inbounds nuw %tensor, ptr %new_tensor, i32 0, i32 2
  %60 = load ptr, ptr %59, align 8
  %61 = getelementptr inbounds nuw %tensor, ptr %new_tensor, i32 0, i32 0
  %62 = load ptr, ptr %61, align 8
  %arena3 = call ptr @eshkol_current_arena()
  %ad_home_arena4 = call ptr @eshkol_ad_home_arena(ptr %arena3)
  %63 = call ptr @arena_allocate_ad_node_with_header(ptr %ad_home_arena4)
  %64 = getelementptr inbounds nuw %ad_node, ptr %63, i32 0, i32 0
  store i32 25, ptr %64, align 4
  %65 = getelementptr inbounds nuw %ad_node, ptr %63, i32 0, i32 1
  store double 0.000000e+00, ptr %65, align 8
  %66 = getelementptr inbounds nuw %ad_node, ptr %63, i32 0, i32 2
  store double 0.000000e+00, ptr %66, align 8
  %67 = getelementptr inbounds nuw %ad_node, ptr %63, i32 0, i32 3
  store ptr %transpose_parent, ptr %67, align 8
  %68 = getelementptr inbounds nuw %ad_node, ptr %63, i32 0, i32 4
  store ptr null, ptr %68, align 8
  %69 = getelementptr inbounds nuw %ad_node, ptr %63, i32 0, i32 5
  store i64 116, ptr %69, align 8
  %70 = getelementptr inbounds nuw %ad_node, ptr %63, i32 0, i32 6
  store ptr %60, ptr %70, align 8
  %71 = getelementptr inbounds nuw %ad_node, ptr %63, i32 0, i32 7
  store ptr null, ptr %71, align 8
  %72 = getelementptr inbounds nuw %ad_node, ptr %63, i32 0, i32 8
  store ptr null, ptr %72, align 8
  %73 = getelementptr inbounds nuw %ad_node, ptr %63, i32 0, i32 9
  store ptr null, ptr %73, align 8
  %74 = getelementptr inbounds nuw %ad_node, ptr %63, i32 0, i32 10
  store ptr null, ptr %74, align 8
  %75 = getelementptr inbounds nuw %ad_node, ptr %63, i32 0, i32 11
  store i64 0, ptr %75, align 8
  %76 = getelementptr inbounds nuw %ad_node, ptr %63, i32 0, i32 12
  %77 = getelementptr [6 x i64], ptr %76, i32 0, i32 0
  store i64 0, ptr %77, align 8
  %78 = getelementptr [6 x i64], ptr %76, i32 0, i32 1
  store i64 0, ptr %78, align 8
  %79 = getelementptr [6 x i64], ptr %76, i32 0, i32 2
  store i64 0, ptr %79, align 8
  %80 = getelementptr [6 x i64], ptr %76, i32 0, i32 3
  store i64 0, ptr %80, align 8
  %81 = getelementptr [6 x i64], ptr %76, i32 0, i32 4
  store i64 0, ptr %81, align 8
  %82 = getelementptr [6 x i64], ptr %76, i32 0, i32 5
  store i64 0, ptr %82, align 8
  %83 = getelementptr inbounds nuw %ad_node, ptr %63, i32 0, i32 13
  store ptr %62, ptr %83, align 8
  %84 = getelementptr inbounds nuw %ad_node, ptr %63, i32 0, i32 14
  store i64 2, ptr %84, align 8
  %85 = load ptr, ptr @__current_ad_tape, align 8
  %86 = ptrtoint ptr %85 to i64
  %87 = icmp ne i64 %86, 0
  br i1 %87, label %add_tensor_to_tape, label %skip_tensor_tape

transpose_plain_result:                           ; preds = %transpose_exit
  %88 = load %eshkol_tagged_value, ptr %trans_result, align 8
  br label %transpose_result_join

transpose_result_join:                            ; preds = %transpose_plain_result, %skip_tensor_tape
  %transpose_result = phi %eshkol_tagged_value [ %99, %skip_tensor_tape ], [ %88, %transpose_plain_result ]
  %89 = extractvalue %eshkol_tagged_value %transpose_result, 0
  %90 = icmp uge i8 %89, 8
  %91 = and i8 %89, 15
  %92 = select i1 %90, i8 %89, i8 %91
  %93 = icmp eq i8 %92, 9
  br i1 %93, label %mm_callable, label %mm_plain

add_tensor_to_tape:                               ; preds = %transpose_record
  %94 = call i64 @arena_tape_add_node(ptr %85, ptr %63)
  br label %skip_tensor_tape

skip_tensor_tape:                                 ; preds = %add_tensor_to_tape, %transpose_record
  %95 = getelementptr inbounds nuw %ad_node, ptr %63, i32 0, i32 12
  %96 = getelementptr [6 x i64], ptr %95, i32 0, i32 0
  store i64 %28, ptr %96, align 8
  %97 = getelementptr [6 x i64], ptr %95, i32 0, i32 1
  store i64 %30, ptr %97, align 8
  %98 = ptrtoint ptr %63 to i64
  %99 = insertvalue %eshkol_tagged_value { i8 9, i8 0, i16 0, i32 0, i64 undef }, i64 %98, 4
  br label %transpose_result_join

mm_callable:                                      ; preds = %transpose_result_join
  %100 = extractvalue %eshkol_tagged_value %transpose_result, 4
  %101 = inttoptr i64 %100 to ptr
  %102 = extractvalue %eshkol_tagged_value %transpose_result, 4
  %data_ptr5 = inttoptr i64 %102 to ptr
  %header_ptr6 = getelementptr i8, ptr %data_ptr5, i64 -8
  %subtype7 = load i8, ptr %header_ptr6, align 1
  %callable_subtype_match8 = icmp eq i8 %subtype7, 2
  br i1 %callable_subtype_match8, label %mm_ad_input, label %mm_not_ad

mm_plain:                                         ; preds = %mm_ad_input, %transpose_result_join
  call void @eshkol_set_error_location(ptr @.str.32, i32 2, i32 32)
  %tensor_operand_slot10 = alloca %eshkol_tagged_value, align 8
  store %eshkol_tagged_value %transpose_result, ptr %tensor_operand_slot10, align 8
  %tensor_ptr_checked11 = call ptr @eshkol_tensor_operand_carrier_checked(ptr %tensor_operand_slot10, ptr @.str.34)
  br label %mm_input_merge

mm_input_merge:                                   ; preds = %mm_plain, %mm_not_ad, %mm_ad_tensor
  %tensor_ptr = phi ptr [ %mm_ad_view, %mm_ad_tensor ], [ %101, %mm_not_ad ], [ %tensor_ptr_checked11, %mm_plain ]
  %ad_node_ptr = phi ptr [ %101, %mm_ad_tensor ], [ null, %mm_not_ad ], [ null, %mm_plain ]
  %103 = extractvalue %eshkol_tagged_value %x, 0
  %104 = icmp uge i8 %103, 8
  %105 = and i8 %103, 15
  %106 = select i1 %104, i8 %103, i8 %105
  %107 = icmp eq i8 %106, 9
  br i1 %107, label %mm_callable12, label %mm_plain13

mm_ad_input:                                      ; preds = %mm_callable
  %108 = getelementptr inbounds nuw %ad_node, ptr %101, i32 0, i32 6
  %109 = load ptr, ptr %108, align 8
  %110 = icmp ne ptr %109, null
  br i1 %110, label %mm_ad_tensor, label %mm_plain

mm_not_ad:                                        ; preds = %mm_callable
  br label %mm_input_merge

mm_ad_tensor:                                     ; preds = %mm_ad_input
  %111 = getelementptr inbounds nuw %ad_node, ptr %101, i32 0, i32 13
  %112 = load ptr, ptr %111, align 8
  %113 = getelementptr inbounds nuw %ad_node, ptr %101, i32 0, i32 14
  %114 = load i64, ptr %113, align 8
  %mm_ad_total = call i64 @eshkol_ad_node_total_elements(ptr %101)
  %cur_arena = call ptr @eshkol_current_arena()
  %ad_home_arena9 = call ptr @eshkol_ad_home_arena(ptr %cur_arena)
  %mm_ad_view = call ptr @arena_allocate_tensor_with_header(ptr %ad_home_arena9)
  %115 = getelementptr inbounds nuw %tensor, ptr %mm_ad_view, i32 0, i32 0
  store ptr %112, ptr %115, align 8
  %116 = getelementptr inbounds nuw %tensor, ptr %mm_ad_view, i32 0, i32 1
  store i64 %114, ptr %116, align 8
  %117 = getelementptr inbounds nuw %tensor, ptr %mm_ad_view, i32 0, i32 2
  store ptr %109, ptr %117, align 8
  %118 = getelementptr inbounds nuw %tensor, ptr %mm_ad_view, i32 0, i32 3
  store i64 %mm_ad_total, ptr %118, align 8
  %119 = getelementptr inbounds nuw %tensor, ptr %mm_ad_view, i32 0, i32 4
  store i64 0, ptr %119, align 8
  br label %mm_input_merge

mm_callable12:                                    ; preds = %mm_input_merge
  %120 = extractvalue %eshkol_tagged_value %x, 4
  %121 = inttoptr i64 %120 to ptr
  %122 = extractvalue %eshkol_tagged_value %x, 4
  %data_ptr15 = inttoptr i64 %122 to ptr
  %header_ptr16 = getelementptr i8, ptr %data_ptr15, i64 -8
  %subtype17 = load i8, ptr %header_ptr16, align 1
  %callable_subtype_match18 = icmp eq i8 %subtype17, 2
  br i1 %callable_subtype_match18, label %mm_ad_input19, label %mm_not_ad20

mm_plain13:                                       ; preds = %mm_ad_input19, %mm_input_merge
  call void @eshkol_set_error_location(ptr @.str.32, i32 2, i32 32)
  %tensor_operand_slot26 = alloca %eshkol_tagged_value, align 8
  store %eshkol_tagged_value %x, ptr %tensor_operand_slot26, align 8
  %tensor_ptr_checked27 = call ptr @eshkol_tensor_operand_carrier_checked(ptr %tensor_operand_slot26, ptr @.str.34)
  br label %mm_input_merge14

mm_input_merge14:                                 ; preds = %mm_plain13, %mm_not_ad20, %mm_ad_tensor21
  %tensor_ptr28 = phi ptr [ %mm_ad_view25, %mm_ad_tensor21 ], [ %121, %mm_not_ad20 ], [ %tensor_ptr_checked27, %mm_plain13 ]
  %ad_node_ptr29 = phi ptr [ %121, %mm_ad_tensor21 ], [ null, %mm_not_ad20 ], [ null, %mm_plain13 ]
  %123 = getelementptr inbounds nuw %tensor, ptr %tensor_ptr, i32 0, i32 4
  %124 = load i64, ptr %123, align 8
  %125 = getelementptr inbounds nuw %tensor, ptr %tensor_ptr28, i32 0, i32 4
  %126 = load i64, ptr %125, align 8
  %127 = icmp eq i64 %124, 64
  %128 = icmp eq i64 %126, 64
  %129 = or i1 %127, %128
  br i1 %129, label %matmul_dual, label %matmul_normal

mm_ad_input19:                                    ; preds = %mm_callable12
  %130 = getelementptr inbounds nuw %ad_node, ptr %121, i32 0, i32 6
  %131 = load ptr, ptr %130, align 8
  %132 = icmp ne ptr %131, null
  br i1 %132, label %mm_ad_tensor21, label %mm_plain13

mm_not_ad20:                                      ; preds = %mm_callable12
  br label %mm_input_merge14

mm_ad_tensor21:                                   ; preds = %mm_ad_input19
  %133 = getelementptr inbounds nuw %ad_node, ptr %121, i32 0, i32 13
  %134 = load ptr, ptr %133, align 8
  %135 = getelementptr inbounds nuw %ad_node, ptr %121, i32 0, i32 14
  %136 = load i64, ptr %135, align 8
  %mm_ad_total22 = call i64 @eshkol_ad_node_total_elements(ptr %121)
  %cur_arena23 = call ptr @eshkol_current_arena()
  %ad_home_arena24 = call ptr @eshkol_ad_home_arena(ptr %cur_arena23)
  %mm_ad_view25 = call ptr @arena_allocate_tensor_with_header(ptr %ad_home_arena24)
  %137 = getelementptr inbounds nuw %tensor, ptr %mm_ad_view25, i32 0, i32 0
  store ptr %134, ptr %137, align 8
  %138 = getelementptr inbounds nuw %tensor, ptr %mm_ad_view25, i32 0, i32 1
  store i64 %136, ptr %138, align 8
  %139 = getelementptr inbounds nuw %tensor, ptr %mm_ad_view25, i32 0, i32 2
  store ptr %131, ptr %139, align 8
  %140 = getelementptr inbounds nuw %tensor, ptr %mm_ad_view25, i32 0, i32 3
  store i64 %mm_ad_total22, ptr %140, align 8
  %141 = getelementptr inbounds nuw %tensor, ptr %mm_ad_view25, i32 0, i32 4
  store i64 0, ptr %141, align 8
  br label %mm_input_merge14

matmul_dual:                                      ; preds = %mm_input_merge14
  %142 = icmp ne ptr %ad_node_ptr29, null
  %143 = icmp ne ptr %ad_node_ptr, null
  %144 = or i1 %143, %142
  call void @eshkol_set_error_location(ptr @.str.32, i32 2, i32 32)
  %145 = zext i1 %144 to i32
  %arena30 = call ptr @eshkol_current_arena()
  %jet_tensor_matmul = call ptr @eshkol_jet_tensor_matmul(ptr %arena30, ptr %tensor_ptr, ptr %tensor_ptr28, i32 %145, ptr @.str.34)
  %146 = ptrtoint ptr %jet_tensor_matmul to i64
  %147 = insertvalue %eshkol_tagged_value { i8 8, i8 0, i16 0, i32 0, i64 undef }, i64 %146, 4
  br label %matmul_dual_done

matmul_normal:                                    ; preds = %mm_input_merge14
  %148 = getelementptr inbounds nuw %tensor, ptr %tensor_ptr, i32 0, i32 0
  %149 = load ptr, ptr %148, align 8
  %150 = getelementptr inbounds nuw %tensor, ptr %tensor_ptr, i32 0, i32 1
  %151 = load i64, ptr %150, align 8
  %152 = getelementptr inbounds nuw %tensor, ptr %tensor_ptr, i32 0, i32 3
  %153 = load i64, ptr %152, align 8
  %154 = getelementptr inbounds nuw %tensor, ptr %tensor_ptr28, i32 0, i32 0
  %155 = load ptr, ptr %154, align 8
  %156 = getelementptr inbounds nuw %tensor, ptr %tensor_ptr28, i32 0, i32 1
  %157 = load i64, ptr %156, align 8
  %158 = icmp eq i64 %151, 2
  %159 = icmp eq i64 %151, 1
  %160 = or i1 %159, %158
  %161 = icmp eq i64 %157, 2
  %162 = icmp eq i64 %157, 1
  %163 = or i1 %162, %161
  %164 = and i1 %160, %163
  br i1 %164, label %mm_rank_ok, label %mm_rank_err

matmul_dual_done:                                 ; preds = %matmul_dense_done, %matmul_dual
  %matmul_result = phi %eshkol_tagged_value [ %147, %matmul_dual ], [ %matmul_dense_result, %matmul_dense_done ]
  ret %eshkol_tagged_value %matmul_result

mm_rank_ok:                                       ; preds = %matmul_normal
  %165 = icmp eq i64 %151, 1
  br i1 %165, label %mm_a_1d, label %mm_a_2d

mm_rank_err:                                      ; preds = %matmul_normal
  %166 = call ptr @eshkol_make_exception_with_header(i32 0, ptr @46)
  call void @eshkol_raise(ptr %166)
  unreachable

mm_a_1d:                                          ; preds = %mm_rank_ok
  br label %mm_a_merge

mm_a_2d:                                          ; preds = %mm_rank_ok
  %167 = load i64, ptr %149, align 8
  %168 = getelementptr i64, ptr %149, i64 1
  %169 = load i64, ptr %168, align 8
  br label %mm_a_merge

mm_a_merge:                                       ; preds = %mm_a_2d, %mm_a_1d
  %M = phi i64 [ 1, %mm_a_1d ], [ %167, %mm_a_2d ]
  %K = phi i64 [ %153, %mm_a_1d ], [ %169, %mm_a_2d ]
  %170 = load i64, ptr %155, align 8
  %171 = icmp eq i64 %157, 1
  br i1 %171, label %mm_b_1d, label %mm_b_2d

mm_b_1d:                                          ; preds = %mm_a_merge
  br label %mm_b_merge

mm_b_2d:                                          ; preds = %mm_a_merge
  %172 = getelementptr i64, ptr %155, i64 1
  %173 = load i64, ptr %172, align 8
  br label %mm_b_merge

mm_b_merge:                                       ; preds = %mm_b_2d, %mm_b_1d
  %N = phi i64 [ 1, %mm_b_1d ], [ %173, %mm_b_2d ]
  %174 = icmp eq i64 %K, %170
  %175 = call i64 @eshkol_matmul_shape_valid(i64 %M, i64 %K, i64 %N)
  %176 = icmp ne i64 %175, 0
  %177 = and i1 %174, %176
  br i1 %177, label %mm_shape_ok, label %mm_shape_err, !eshkol.verificatum.matmul !35

mm_shape_ok:                                      ; preds = %mm_b_merge
  %178 = getelementptr inbounds nuw %tensor, ptr %tensor_ptr, i32 0, i32 2
  %179 = load ptr, ptr %178, align 8
  %180 = getelementptr inbounds nuw %tensor, ptr %tensor_ptr28, i32 0, i32 2
  %181 = load ptr, ptr %180, align 8
  %182 = icmp eq i64 %157, 2
  %183 = icmp eq i64 %151, 2
  %184 = and i1 %183, %182
  br i1 %184, label %mm_res_2d, label %mm_res_1d

mm_shape_err:                                     ; preds = %mm_b_merge
  %185 = call i32 (ptr, ...) @printf(ptr @47, i64 %K, i64 %170)
  call void @exit(i32 1)
  unreachable

mm_res_2d:                                        ; preds = %mm_shape_ok
  %arena31 = call ptr @eshkol_current_arena()
  %ad_home_arena32 = call ptr @eshkol_ad_home_arena(ptr %arena31)
  %new_tensor33 = call ptr @arena_allocate_tensor_with_header(ptr %ad_home_arena32)
  %new_dims34 = call ptr @arena_allocate(ptr %ad_home_arena32, i64 16)
  %186 = getelementptr i64, ptr %new_dims34, i64 0
  store i64 %M, ptr %186, align 8
  %187 = getelementptr i64, ptr %new_dims34, i64 1
  store i64 %N, ptr %187, align 8
  %checked_total35 = call i64 @eshkol_tensor_shape_total(ptr %new_dims34, i64 2)
  %188 = icmp slt i64 %checked_total35, 0
  br i1 %188, label %tensor_shape_err37, label %tensor_shape_ok36

mm_res_1d:                                        ; preds = %mm_shape_ok
  %189 = select i1 %165, i64 %N, i64 %M
  %arena39 = call ptr @eshkol_current_arena()
  %ad_home_arena40 = call ptr @eshkol_ad_home_arena(ptr %arena39)
  %new_tensor41 = call ptr @arena_allocate_tensor_with_header(ptr %ad_home_arena40)
  %new_dims42 = call ptr @arena_allocate(ptr %ad_home_arena40, i64 8)
  %190 = getelementptr i64, ptr %new_dims42, i64 0
  store i64 %189, ptr %190, align 8
  %checked_total43 = call i64 @eshkol_tensor_shape_total(ptr %new_dims42, i64 1)
  %191 = icmp slt i64 %checked_total43, 0
  br i1 %191, label %tensor_shape_err45, label %tensor_shape_ok44

mm_res_merge:                                     ; preds = %tensor_shape_ok44, %tensor_shape_ok36
  %result_ptr = phi ptr [ %new_tensor33, %tensor_shape_ok36 ], [ %new_tensor41, %tensor_shape_ok44 ]
  %192 = call ptr @eshkol_tensor_result_dtype_binary(ptr %result_ptr, ptr %tensor_ptr, ptr %tensor_ptr28)
  %193 = getelementptr inbounds nuw %tensor, ptr %result_ptr, i32 0, i32 2
  %194 = load ptr, ptr %193, align 8
  %195 = load i1, ptr @__ad_mode_active, align 1
  br i1 %195, label %matmul_dense_pack, label %matmul_dense_plain

tensor_shape_ok36:                                ; preds = %mm_res_2d
  call void @eshkol_enforce_tensor_elements(i64 %checked_total35)
  %196 = mul i64 %checked_total35, 8
  %new_elems38 = call ptr @arena_allocate(ptr %ad_home_arena32, i64 %196)
  %197 = call ptr @memset(ptr %new_elems38, i32 0, i64 %196)
  %198 = getelementptr inbounds nuw %tensor, ptr %new_tensor33, i32 0, i32 0
  store ptr %new_dims34, ptr %198, align 8
  %199 = getelementptr inbounds nuw %tensor, ptr %new_tensor33, i32 0, i32 1
  store i64 2, ptr %199, align 8
  %200 = getelementptr inbounds nuw %tensor, ptr %new_tensor33, i32 0, i32 2
  store ptr %new_elems38, ptr %200, align 8
  %201 = getelementptr inbounds nuw %tensor, ptr %new_tensor33, i32 0, i32 3
  store i64 %checked_total35, ptr %201, align 8
  br label %mm_res_merge

tensor_shape_err37:                               ; preds = %mm_res_2d
  %202 = call ptr @eshkol_make_exception_with_header(i32 0, ptr @48)
  call void @eshkol_raise(ptr %202)
  unreachable

tensor_shape_ok44:                                ; preds = %mm_res_1d
  call void @eshkol_enforce_tensor_elements(i64 %checked_total43)
  %203 = mul i64 %checked_total43, 8
  %new_elems46 = call ptr @arena_allocate(ptr %ad_home_arena40, i64 %203)
  %204 = call ptr @memset(ptr %new_elems46, i32 0, i64 %203)
  %205 = getelementptr inbounds nuw %tensor, ptr %new_tensor41, i32 0, i32 0
  store ptr %new_dims42, ptr %205, align 8
  %206 = getelementptr inbounds nuw %tensor, ptr %new_tensor41, i32 0, i32 1
  store i64 1, ptr %206, align 8
  %207 = getelementptr inbounds nuw %tensor, ptr %new_tensor41, i32 0, i32 2
  store ptr %new_elems46, ptr %207, align 8
  %208 = getelementptr inbounds nuw %tensor, ptr %new_tensor41, i32 0, i32 3
  store i64 %checked_total43, ptr %208, align 8
  br label %mm_res_merge

tensor_shape_err45:                               ; preds = %mm_res_1d
  %209 = call ptr @eshkol_make_exception_with_header(i32 0, ptr @49)
  call void @eshkol_raise(ptr %209)
  unreachable

matmul_dense_pack:                                ; preds = %mm_res_merge
  %210 = getelementptr inbounds nuw %tensor, ptr %tensor_ptr, i32 0, i32 0
  %211 = load ptr, ptr %210, align 8
  %212 = getelementptr inbounds nuw %tensor, ptr %tensor_ptr, i32 0, i32 1
  %213 = load i64, ptr %212, align 8
  %214 = getelementptr inbounds nuw %tensor, ptr %tensor_ptr28, i32 0, i32 0
  %215 = load ptr, ptr %214, align 8
  %216 = getelementptr inbounds nuw %tensor, ptr %tensor_ptr28, i32 0, i32 1
  %217 = load i64, ptr %216, align 8
  %218 = mul i64 %M, %K
  %219 = mul i64 %K, %N
  %arena47 = call ptr @eshkol_current_arena()
  %ad_home_arena48 = call ptr @eshkol_ad_home_arena(ptr %arena47)
  %220 = icmp ne ptr %ad_node_ptr, null
  br i1 %220, label %mm_a_reuse, label %mm_a_pack

matmul_dense_plain:                               ; preds = %mm_res_merge
  br label %matmul_dense_join

matmul_dense_join:                                ; preds = %matmul_dense_plain, %mm_b_join
  %mm_a_use = phi ptr [ %mm_a_buf, %mm_b_join ], [ %179, %matmul_dense_plain ]
  %mm_b_use = phi ptr [ %mm_b_buf, %mm_b_join ], [ %181, %matmul_dense_plain ]
  %mm_parent_a = phi ptr [ %mm_a_node54, %mm_b_join ], [ null, %matmul_dense_plain ]
  %mm_parent_b = phi ptr [ %mm_b_node63, %mm_b_join ], [ null, %matmul_dense_plain ]
  %221 = getelementptr inbounds nuw %tensor, ptr %result_ptr, i32 0, i32 4
  %222 = load i64, ptr %221, align 8
  %223 = trunc i64 %222 to i32
  call void @eshkol_matmul_dispatch(ptr %mm_a_use, ptr %mm_b_use, ptr %194, i64 %M, i64 %K, i64 %N, i32 %223)
  br i1 %195, label %matmul_dense_record, label %matmul_dense_skip

mm_a_reuse:                                       ; preds = %matmul_dense_pack
  %224 = getelementptr inbounds nuw %ad_node, ptr %ad_node_ptr, i32 0, i32 6
  %225 = load ptr, ptr %224, align 8
  %226 = icmp ne ptr %225, null
  br i1 %226, label %mm_a_reuse_ok, label %mm_a_pack

mm_a_pack:                                        ; preds = %mm_a_reuse, %matmul_dense_pack
  %227 = mul i64 %218, 8
  %mm_a_dense = call ptr @arena_allocate(ptr %ad_home_arena48, i64 %227)
  %mm_a_slots = call ptr @arena_allocate(ptr %ad_home_arena48, i64 %227)
  %mm_a_i = alloca i64, align 8
  store i64 0, ptr %mm_a_i, align 8
  br label %mm_a_cond

mm_a_join:                                        ; preds = %skip_tensor_tape53, %mm_a_reuse_ok
  %mm_a_buf = phi ptr [ %225, %mm_a_reuse_ok ], [ %mm_a_dense, %skip_tensor_tape53 ]
  %mm_a_node54 = phi ptr [ %ad_node_ptr, %mm_a_reuse_ok ], [ %234, %skip_tensor_tape53 ]
  %arena55 = call ptr @eshkol_current_arena()
  %ad_home_arena56 = call ptr @eshkol_ad_home_arena(ptr %arena55)
  %228 = icmp ne ptr %ad_node_ptr29, null
  br i1 %228, label %mm_b_reuse, label %mm_b_pack

mm_a_reuse_ok:                                    ; preds = %mm_a_reuse
  br label %mm_a_join

mm_a_cond:                                        ; preds = %mm_a_store, %mm_a_pack
  %229 = load i64, ptr %mm_a_i, align 8
  %230 = icmp ult i64 %229, %218
  br i1 %230, label %mm_a_body, label %mm_a_exit

mm_a_body:                                        ; preds = %mm_a_cond
  %231 = getelementptr i64, ptr %179, i64 %229
  %232 = load i64, ptr %231, align 8
  %233 = icmp ult i64 %232, 1000
  br i1 %233, label %mm_a_small, label %mm_a_probe

mm_a_exit:                                        ; preds = %mm_a_cond
  %mm_a_shape_home = call ptr @eshkol_ad_copy_shape_to_home(ptr %211, i64 %213)
  %arena50 = call ptr @eshkol_current_arena()
  %ad_home_arena51 = call ptr @eshkol_ad_home_arena(ptr %arena50)
  %234 = call ptr @arena_allocate_ad_node_with_header(ptr %ad_home_arena51)
  %235 = getelementptr inbounds nuw %ad_node, ptr %234, i32 0, i32 0
  store i32 83, ptr %235, align 4
  %236 = getelementptr inbounds nuw %ad_node, ptr %234, i32 0, i32 1
  store double 0.000000e+00, ptr %236, align 8
  %237 = getelementptr inbounds nuw %ad_node, ptr %234, i32 0, i32 2
  store double 0.000000e+00, ptr %237, align 8
  %238 = getelementptr inbounds nuw %ad_node, ptr %234, i32 0, i32 3
  store ptr null, ptr %238, align 8
  %239 = getelementptr inbounds nuw %ad_node, ptr %234, i32 0, i32 4
  store ptr null, ptr %239, align 8
  %240 = getelementptr inbounds nuw %ad_node, ptr %234, i32 0, i32 5
  store i64 117, ptr %240, align 8
  %241 = getelementptr inbounds nuw %ad_node, ptr %234, i32 0, i32 6
  store ptr %mm_a_dense, ptr %241, align 8
  %242 = getelementptr inbounds nuw %ad_node, ptr %234, i32 0, i32 7
  store ptr null, ptr %242, align 8
  %243 = getelementptr inbounds nuw %ad_node, ptr %234, i32 0, i32 8
  store ptr null, ptr %243, align 8
  %244 = getelementptr inbounds nuw %ad_node, ptr %234, i32 0, i32 9
  store ptr null, ptr %244, align 8
  %245 = getelementptr inbounds nuw %ad_node, ptr %234, i32 0, i32 10
  store ptr %mm_a_slots, ptr %245, align 8
  %246 = getelementptr inbounds nuw %ad_node, ptr %234, i32 0, i32 11
  store i64 %218, ptr %246, align 8
  %247 = getelementptr inbounds nuw %ad_node, ptr %234, i32 0, i32 12
  %248 = getelementptr [6 x i64], ptr %247, i32 0, i32 0
  store i64 0, ptr %248, align 8
  %249 = getelementptr [6 x i64], ptr %247, i32 0, i32 1
  store i64 0, ptr %249, align 8
  %250 = getelementptr [6 x i64], ptr %247, i32 0, i32 2
  store i64 0, ptr %250, align 8
  %251 = getelementptr [6 x i64], ptr %247, i32 0, i32 3
  store i64 0, ptr %251, align 8
  %252 = getelementptr [6 x i64], ptr %247, i32 0, i32 4
  store i64 0, ptr %252, align 8
  %253 = getelementptr [6 x i64], ptr %247, i32 0, i32 5
  store i64 0, ptr %253, align 8
  %254 = getelementptr inbounds nuw %ad_node, ptr %234, i32 0, i32 13
  store ptr %mm_a_shape_home, ptr %254, align 8
  %255 = getelementptr inbounds nuw %ad_node, ptr %234, i32 0, i32 14
  store i64 %213, ptr %255, align 8
  %256 = load ptr, ptr @__current_ad_tape, align 8
  %257 = ptrtoint ptr %256 to i64
  %258 = icmp ne i64 %257, 0
  br i1 %258, label %add_tensor_to_tape52, label %skip_tensor_tape53

mm_a_small:                                       ; preds = %mm_a_body
  %259 = sitofp i64 %232 to double
  br label %mm_a_store

mm_a_probe:                                       ; preds = %mm_a_body
  %arena49 = call ptr @eshkol_current_arena()
  %ad_node_probe = call i32 @eshkol_ad_node_probe(ptr %arena49, i64 %232, i32 -1)
  %260 = icmp ne i32 %ad_node_probe, 0
  br i1 %260, label %mm_a_node, label %mm_a_double

mm_a_node:                                        ; preds = %mm_a_probe
  %261 = inttoptr i64 %232 to ptr
  %262 = getelementptr inbounds nuw %ad_node, ptr %261, i32 0, i32 1
  %263 = load double, ptr %262, align 8
  br label %mm_a_store

mm_a_double:                                      ; preds = %mm_a_probe
  %264 = bitcast i64 %232 to double
  br label %mm_a_store

mm_a_store:                                       ; preds = %mm_a_double, %mm_a_node, %mm_a_small
  %mm_a_val = phi double [ %259, %mm_a_small ], [ %263, %mm_a_node ], [ %264, %mm_a_double ]
  %mm_a_slot = phi ptr [ null, %mm_a_small ], [ %261, %mm_a_node ], [ null, %mm_a_double ]
  %265 = getelementptr double, ptr %mm_a_dense, i64 %229
  store double %mm_a_val, ptr %265, align 8
  %266 = getelementptr ptr, ptr %mm_a_slots, i64 %229
  store ptr %mm_a_slot, ptr %266, align 8
  %267 = add i64 %229, 1
  store i64 %267, ptr %mm_a_i, align 8
  br label %mm_a_cond

add_tensor_to_tape52:                             ; preds = %mm_a_exit
  %268 = call i64 @arena_tape_add_node(ptr %256, ptr %234)
  br label %skip_tensor_tape53

skip_tensor_tape53:                               ; preds = %add_tensor_to_tape52, %mm_a_exit
  %269 = getelementptr inbounds nuw %ad_node, ptr %234, i32 0, i32 12
  %270 = getelementptr [6 x i64], ptr %269, i32 0, i32 0
  store i64 %218, ptr %270, align 8
  br label %mm_a_join

mm_b_reuse:                                       ; preds = %mm_a_join
  %271 = getelementptr inbounds nuw %ad_node, ptr %ad_node_ptr29, i32 0, i32 6
  %272 = load ptr, ptr %271, align 8
  %273 = icmp ne ptr %272, null
  br i1 %273, label %mm_b_reuse_ok, label %mm_b_pack

mm_b_pack:                                        ; preds = %mm_b_reuse, %mm_a_join
  %274 = mul i64 %219, 8
  %mm_b_dense = call ptr @arena_allocate(ptr %ad_home_arena56, i64 %274)
  %mm_b_slots = call ptr @arena_allocate(ptr %ad_home_arena56, i64 %274)
  %mm_b_i = alloca i64, align 8
  store i64 0, ptr %mm_b_i, align 8
  br label %mm_b_cond

mm_b_join:                                        ; preds = %skip_tensor_tape62, %mm_b_reuse_ok
  %mm_b_buf = phi ptr [ %272, %mm_b_reuse_ok ], [ %mm_b_dense, %skip_tensor_tape62 ]
  %mm_b_node63 = phi ptr [ %ad_node_ptr29, %mm_b_reuse_ok ], [ %280, %skip_tensor_tape62 ]
  br label %matmul_dense_join

mm_b_reuse_ok:                                    ; preds = %mm_b_reuse
  br label %mm_b_join

mm_b_cond:                                        ; preds = %mm_b_store, %mm_b_pack
  %275 = load i64, ptr %mm_b_i, align 8
  %276 = icmp ult i64 %275, %219
  br i1 %276, label %mm_b_body, label %mm_b_exit

mm_b_body:                                        ; preds = %mm_b_cond
  %277 = getelementptr i64, ptr %181, i64 %275
  %278 = load i64, ptr %277, align 8
  %279 = icmp ult i64 %278, 1000
  br i1 %279, label %mm_b_small, label %mm_b_probe

mm_b_exit:                                        ; preds = %mm_b_cond
  %mm_b_shape_home = call ptr @eshkol_ad_copy_shape_to_home(ptr %215, i64 %217)
  %arena59 = call ptr @eshkol_current_arena()
  %ad_home_arena60 = call ptr @eshkol_ad_home_arena(ptr %arena59)
  %280 = call ptr @arena_allocate_ad_node_with_header(ptr %ad_home_arena60)
  %281 = getelementptr inbounds nuw %ad_node, ptr %280, i32 0, i32 0
  store i32 83, ptr %281, align 4
  %282 = getelementptr inbounds nuw %ad_node, ptr %280, i32 0, i32 1
  store double 0.000000e+00, ptr %282, align 8
  %283 = getelementptr inbounds nuw %ad_node, ptr %280, i32 0, i32 2
  store double 0.000000e+00, ptr %283, align 8
  %284 = getelementptr inbounds nuw %ad_node, ptr %280, i32 0, i32 3
  store ptr null, ptr %284, align 8
  %285 = getelementptr inbounds nuw %ad_node, ptr %280, i32 0, i32 4
  store ptr null, ptr %285, align 8
  %286 = getelementptr inbounds nuw %ad_node, ptr %280, i32 0, i32 5
  store i64 118, ptr %286, align 8
  %287 = getelementptr inbounds nuw %ad_node, ptr %280, i32 0, i32 6
  store ptr %mm_b_dense, ptr %287, align 8
  %288 = getelementptr inbounds nuw %ad_node, ptr %280, i32 0, i32 7
  store ptr null, ptr %288, align 8
  %289 = getelementptr inbounds nuw %ad_node, ptr %280, i32 0, i32 8
  store ptr null, ptr %289, align 8
  %290 = getelementptr inbounds nuw %ad_node, ptr %280, i32 0, i32 9
  store ptr null, ptr %290, align 8
  %291 = getelementptr inbounds nuw %ad_node, ptr %280, i32 0, i32 10
  store ptr %mm_b_slots, ptr %291, align 8
  %292 = getelementptr inbounds nuw %ad_node, ptr %280, i32 0, i32 11
  store i64 %219, ptr %292, align 8
  %293 = getelementptr inbounds nuw %ad_node, ptr %280, i32 0, i32 12
  %294 = getelementptr [6 x i64], ptr %293, i32 0, i32 0
  store i64 0, ptr %294, align 8
  %295 = getelementptr [6 x i64], ptr %293, i32 0, i32 1
  store i64 0, ptr %295, align 8
  %296 = getelementptr [6 x i64], ptr %293, i32 0, i32 2
  store i64 0, ptr %296, align 8
  %297 = getelementptr [6 x i64], ptr %293, i32 0, i32 3
  store i64 0, ptr %297, align 8
  %298 = getelementptr [6 x i64], ptr %293, i32 0, i32 4
  store i64 0, ptr %298, align 8
  %299 = getelementptr [6 x i64], ptr %293, i32 0, i32 5
  store i64 0, ptr %299, align 8
  %300 = getelementptr inbounds nuw %ad_node, ptr %280, i32 0, i32 13
  store ptr %mm_b_shape_home, ptr %300, align 8
  %301 = getelementptr inbounds nuw %ad_node, ptr %280, i32 0, i32 14
  store i64 %217, ptr %301, align 8
  %302 = load ptr, ptr @__current_ad_tape, align 8
  %303 = ptrtoint ptr %302 to i64
  %304 = icmp ne i64 %303, 0
  br i1 %304, label %add_tensor_to_tape61, label %skip_tensor_tape62

mm_b_small:                                       ; preds = %mm_b_body
  %305 = sitofp i64 %278 to double
  br label %mm_b_store

mm_b_probe:                                       ; preds = %mm_b_body
  %arena57 = call ptr @eshkol_current_arena()
  %ad_node_probe58 = call i32 @eshkol_ad_node_probe(ptr %arena57, i64 %278, i32 -1)
  %306 = icmp ne i32 %ad_node_probe58, 0
  br i1 %306, label %mm_b_node, label %mm_b_double

mm_b_node:                                        ; preds = %mm_b_probe
  %307 = inttoptr i64 %278 to ptr
  %308 = getelementptr inbounds nuw %ad_node, ptr %307, i32 0, i32 1
  %309 = load double, ptr %308, align 8
  br label %mm_b_store

mm_b_double:                                      ; preds = %mm_b_probe
  %310 = bitcast i64 %278 to double
  br label %mm_b_store

mm_b_store:                                       ; preds = %mm_b_double, %mm_b_node, %mm_b_small
  %mm_b_val = phi double [ %305, %mm_b_small ], [ %309, %mm_b_node ], [ %310, %mm_b_double ]
  %mm_b_slot = phi ptr [ null, %mm_b_small ], [ %307, %mm_b_node ], [ null, %mm_b_double ]
  %311 = getelementptr double, ptr %mm_b_dense, i64 %275
  store double %mm_b_val, ptr %311, align 8
  %312 = getelementptr ptr, ptr %mm_b_slots, i64 %275
  store ptr %mm_b_slot, ptr %312, align 8
  %313 = add i64 %275, 1
  store i64 %313, ptr %mm_b_i, align 8
  br label %mm_b_cond

add_tensor_to_tape61:                             ; preds = %mm_b_exit
  %314 = call i64 @arena_tape_add_node(ptr %302, ptr %280)
  br label %skip_tensor_tape62

skip_tensor_tape62:                               ; preds = %add_tensor_to_tape61, %mm_b_exit
  %315 = getelementptr inbounds nuw %ad_node, ptr %280, i32 0, i32 12
  %316 = getelementptr [6 x i64], ptr %315, i32 0, i32 0
  store i64 %219, ptr %316, align 8
  br label %mm_b_join

matmul_dense_record:                              ; preds = %matmul_dense_join
  %cur_arena64 = call ptr @eshkol_current_arena()
  %ad_home_arena65 = call ptr @eshkol_ad_home_arena(ptr %cur_arena64)
  %317 = call ptr @arena_allocate(ptr %ad_home_arena65, i64 16)
  %318 = getelementptr ptr, ptr %317, i64 0
  store ptr %mm_a_use, ptr %318, align 8
  %319 = getelementptr ptr, ptr %317, i64 1
  store ptr %mm_b_use, ptr %319, align 8
  %320 = getelementptr inbounds nuw %tensor, ptr %result_ptr, i32 0, i32 0
  %321 = load ptr, ptr %320, align 8
  %322 = getelementptr inbounds nuw %tensor, ptr %result_ptr, i32 0, i32 1
  %323 = load i64, ptr %322, align 8
  %arena66 = call ptr @eshkol_current_arena()
  %ad_home_arena67 = call ptr @eshkol_ad_home_arena(ptr %arena66)
  %324 = call ptr @arena_allocate_ad_node_with_header(ptr %ad_home_arena67)
  %325 = getelementptr inbounds nuw %ad_node, ptr %324, i32 0, i32 0
  store i32 24, ptr %325, align 4
  %326 = getelementptr inbounds nuw %ad_node, ptr %324, i32 0, i32 1
  store double 0.000000e+00, ptr %326, align 8
  %327 = getelementptr inbounds nuw %ad_node, ptr %324, i32 0, i32 2
  store double 0.000000e+00, ptr %327, align 8
  %328 = getelementptr inbounds nuw %ad_node, ptr %324, i32 0, i32 3
  store ptr %mm_parent_a, ptr %328, align 8
  %329 = getelementptr inbounds nuw %ad_node, ptr %324, i32 0, i32 4
  store ptr %mm_parent_b, ptr %329, align 8
  %330 = getelementptr inbounds nuw %ad_node, ptr %324, i32 0, i32 5
  store i64 119, ptr %330, align 8
  %331 = getelementptr inbounds nuw %ad_node, ptr %324, i32 0, i32 6
  store ptr %194, ptr %331, align 8
  %332 = getelementptr inbounds nuw %ad_node, ptr %324, i32 0, i32 7
  store ptr null, ptr %332, align 8
  %333 = getelementptr inbounds nuw %ad_node, ptr %324, i32 0, i32 8
  store ptr null, ptr %333, align 8
  %334 = getelementptr inbounds nuw %ad_node, ptr %324, i32 0, i32 9
  store ptr null, ptr %334, align 8
  %335 = getelementptr inbounds nuw %ad_node, ptr %324, i32 0, i32 10
  store ptr %317, ptr %335, align 8
  %336 = getelementptr inbounds nuw %ad_node, ptr %324, i32 0, i32 11
  store i64 2, ptr %336, align 8
  %337 = getelementptr inbounds nuw %ad_node, ptr %324, i32 0, i32 12
  %338 = getelementptr [6 x i64], ptr %337, i32 0, i32 0
  store i64 0, ptr %338, align 8
  %339 = getelementptr [6 x i64], ptr %337, i32 0, i32 1
  store i64 0, ptr %339, align 8
  %340 = getelementptr [6 x i64], ptr %337, i32 0, i32 2
  store i64 0, ptr %340, align 8
  %341 = getelementptr [6 x i64], ptr %337, i32 0, i32 3
  store i64 0, ptr %341, align 8
  %342 = getelementptr [6 x i64], ptr %337, i32 0, i32 4
  store i64 0, ptr %342, align 8
  %343 = getelementptr [6 x i64], ptr %337, i32 0, i32 5
  store i64 0, ptr %343, align 8
  %344 = getelementptr inbounds nuw %ad_node, ptr %324, i32 0, i32 13
  store ptr %321, ptr %344, align 8
  %345 = getelementptr inbounds nuw %ad_node, ptr %324, i32 0, i32 14
  store i64 %323, ptr %345, align 8
  %346 = load ptr, ptr @__current_ad_tape, align 8
  %347 = ptrtoint ptr %346 to i64
  %348 = icmp ne i64 %347, 0
  br i1 %348, label %add_tensor_to_tape68, label %skip_tensor_tape69

matmul_dense_skip:                                ; preds = %matmul_dense_join
  %349 = ptrtoint ptr %result_ptr to i64
  %350 = insertvalue %eshkol_tagged_value { i8 8, i8 0, i16 0, i32 0, i64 undef }, i64 %349, 4
  br label %matmul_dense_done

matmul_dense_done:                                ; preds = %matmul_dense_skip, %skip_tensor_tape69
  %matmul_dense_result = phi %eshkol_tagged_value [ %357, %skip_tensor_tape69 ], [ %350, %matmul_dense_skip ]
  br label %matmul_dual_done

add_tensor_to_tape68:                             ; preds = %matmul_dense_record
  %351 = call i64 @arena_tape_add_node(ptr %346, ptr %324)
  br label %skip_tensor_tape69

skip_tensor_tape69:                               ; preds = %add_tensor_to_tape68, %matmul_dense_record
  %352 = getelementptr inbounds nuw %ad_node, ptr %324, i32 0, i32 12
  %353 = getelementptr [6 x i64], ptr %352, i32 0, i32 0
  store i64 %M, ptr %353, align 8
  %354 = getelementptr [6 x i64], ptr %352, i32 0, i32 1
  store i64 %K, ptr %354, align 8
  %355 = getelementptr [6 x i64], ptr %352, i32 0, i32 2
  store i64 %N, ptr %355, align 8
  %356 = ptrtoint ptr %324 to i64
  %357 = insertvalue %eshkol_tagged_value { i8 9, i8 0, i16 0, i32 0, i64 undef }, i64 %356, 4
  br label %matmul_dense_done
}
