@parameter
fn base_case_div_bw(depth: Int, a: Tensor, b: Tensor) -> Bool:
    return strides_a(depth,a,b)*shape_a(depth,a,b) == strides_b(depth,a,b)*shape_b(depth,a,b)

@parameter
fn kernel_div_bw_a(c: Tensor, inout a: Tensor, inout b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int) -> None:
    let a_shape = shape_a(depth,a,b)
    let b_shape = shape_b(depth,a,b)
    let c_shape = c.shape[depth]

    let a_strides = strides_a(depth,a,b)
    let b_strides = strides_b(depth,a,b)
    let c_strides = c.strides[depth]

    @parameter
    fn v_div_grad_a[nelts: Int](i: Int):
        a.grad.simd_store[nelts](
            a_index*a_strides*a_shape + i, a.grad.simd_load[nelts](a_index*a_strides*a_shape + i) + c.grad.simd_load[nelts](c_index*c_strides*c_shape + i) / b.data.simd_load[nelts](b_index*b_strides*b_shape + i)
        )
    vectorize[nelts, v_div_grad_a](c_strides*c_shape)

@parameter
fn kernel_div_bw_b(c: Tensor, inout a: Tensor, inout b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int) -> None:
    let a_shape = shape_a(depth,a,b)
    let b_shape = shape_b(depth,a,b)
    let c_shape = c.shape[depth]

    let a_strides = strides_a(depth,a,b)
    let b_strides = strides_b(depth,a,b)
    let c_strides = c.strides[depth]

    @parameter
    fn v_div_grad_b[nelts: Int](i: Int):
        b.grad.simd_store[nelts](
            b_index*a_strides*b_shape + i, b.grad.simd_load[nelts](b_index*b_strides*b_shape + i) - a.data.simd_load[nelts](a_index*a_strides*a_shape + i) * c.grad.simd_load[nelts](c_index*c_strides*c_shape + i)  / pow(b.data.simd_load[nelts](b_index*b_strides*b_shape + i),2)
        )
    vectorize[nelts, v_div_grad_b](c_strides*c_shape)

@always_inline
fn e_div_grad(c: Tensor, inout a: Tensor, inout b: Tensor):
    recursive_broadcast_bw[kernel_div_bw_a, base_case_div_bw](c,a,b)
    recursive_broadcast_bw[kernel_div_bw_b, base_case_div_bw](c,a,b)