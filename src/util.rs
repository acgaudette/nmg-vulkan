use std;

pub unsafe fn aligned_buffer<T>(
    alignment: usize, // Bytes
    data:      &[T],
) -> Vec<usize> {
    let count = data.len();
    debug_assert!(count != 0);

    let ptr_len = std::mem::size_of::<usize>();
    debug_assert!(alignment >= ptr_len);

    let alignment = alignment / ptr_len;
    let size = count * alignment;

    // Waiting for std::heap...
    let mut memory = Vec::<usize>::with_capacity(size);
    let mut ptr = memory.as_mut_ptr();
    std::mem::forget(memory);

    let start = ptr.clone();

    // Copy data to aligned buffer
    for entry in data {
        std::ptr::copy_nonoverlapping(
            entry as *const T,
            ptr as *mut T,
            1,
        );

        ptr = ptr.offset(alignment as isize);
    }

    Vec::<usize>::from_raw_parts(start, size, size)
}

#[cfg(test)]
mod tests {
    use util::*;
    use alg;

    #[test]
    fn create_aligned_buffers() {
        let matrices = [
            alg::Mat::identity(),
            alg::Mat::translation(-1., 2., 5.),
            alg::Mat::translation(8., 3., 3.),
        ];

        compare_aligned_buffer(64, &matrices);
        compare_aligned_buffer(128, &matrices);
        compare_aligned_buffer(256, &matrices);
    }

    fn compare_aligned_buffer(alignment: usize, matrices: &[alg::Mat]) {
        let mut raw = unsafe {
            aligned_buffer(alignment, matrices)
        };

        let aligned = {
            let mut result = Vec::<alg::Mat>::with_capacity(matrices.len());
            let ptr_len = std::mem::size_of::<usize>();

            unsafe {
                let mut ptr = raw.as_mut_ptr();
                let offset = (alignment / ptr_len) as isize;
                let mut start = ptr as usize;

                for i in 0..matrices.len() {
                    let matrix = *(ptr as *const alg::Mat);
                    result.push(matrix);

                    let diff = {
                        let end = ptr as usize;
                        let diff = (end - start) / ptr_len;
                        start = end;

                        diff
                    };

                    eprintln!(
                        "\n\tmatrix[{}] diff = {} ({}B)\n{}",
                        i,
                        diff,
                        diff * ptr_len,
                        matrix,
                    );

                    if i > 0 {
                        assert!(diff as isize == offset);
                    }

                    ptr = ptr.offset(offset);
                }
            }

            result
        };

        // Compare matrices
        for i in 0..aligned.len() {
            assert!(aligned[i] == matrices[i]);
        }
    }
}
