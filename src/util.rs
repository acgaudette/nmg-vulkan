use std;
use alg;

pub unsafe fn aligned_buffer(
    alignment: usize, // Bytes
    matrices:  &[alg::Mat],
) -> Vec<usize> {
    let count = matrices.len();

    debug_assert!(alignment != 0);
    debug_assert!(count != 0);

    let ptr_len = std::mem::size_of::<usize>();
    let mat_len = std::mem::size_of::<alg::Mat>() / ptr_len;
    let alignment = alignment / ptr_len;

    let size = count * mat_len + alignment; // Over-allocate

    eprintln!(
        "desired size = {} ({}B); expanded to {} ({}B)\n\
        desired alignment = {} ({}B)",
        count * mat_len,
        count * mat_len * ptr_len,
        size,
        size * ptr_len,
        alignment,
        alignment * ptr_len,
    );

    // Waiting for std::heap...
    let mut memory = Vec::<usize>::with_capacity(size);
    let mut ptr = memory.as_mut_ptr();
    std::mem::forget(memory);

    // Align
    let mut iterations = 0;
    loop {
        let address = ptr as *const usize as usize;

        if address % alignment == 0 {
            eprintln!(
                "\talignment found in {} iterations", iterations,
            );

            break;
        }

        ptr = ptr.offset(1);
        iterations += 1;
    }

    let start = ptr.clone();

    // Copy matrices to aligned buffer
    for matrix in matrices {
        std::ptr::copy_nonoverlapping(
            matrix as *const alg::Mat,
            ptr as *mut alg::Mat,
            mat_len,
        );

        ptr = ptr.offset(alignment as isize);
    }

    let shrinked = size - alignment;

    eprintln!(
        "\tfinal size = {} ({}B)",
        shrinked, shrinked * ptr_len,
    );

    Vec::<usize>::from_raw_parts(start, shrinked, shrinked)
}
