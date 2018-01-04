use std;

pub struct AlignedBuffer<T> {
    alignment: usize,
    pub size:  usize, // Length in usizes
    start:     *const T,
    ptr:       *mut usize,
}

impl<T> AlignedBuffer<T> {
    pub fn new(
        alignment: usize, // Bytes
        count:     usize,
    ) -> AlignedBuffer<T> {
        debug_assert!(count != 0);

        let ptr_len = std::mem::size_of::<usize>();
        debug_assert!(alignment >= ptr_len);
        debug_assert!(std::mem::size_of::<T>() <= alignment);

        let alignment = alignment / ptr_len;
        let size = count * alignment;

        // Waiting for std::heap...
        let mut memory = Vec::<usize>::with_capacity(size);
        let ptr = memory.as_mut_ptr();
        std::mem::forget(memory);

        let start = ptr as *const T;

        AlignedBuffer {
            alignment,
            size,
            start,
            ptr,
        }
    }

    pub fn push(&mut self, entry: T) {
        assert!(
            (self.ptr as usize - self.start as usize)
                / std::mem::size_of::<usize>()
                < self.size
        );

        unsafe {
            std::ptr::copy_nonoverlapping(
                &entry as *const T,
                self.ptr as *mut T,
                1,
            );

            self.ptr = self.ptr.offset(self.alignment as isize);
        }
    }

    pub unsafe fn finalize(&self) -> Vec<usize> {
        Vec::from_raw_parts(
            self.start as *mut usize,
            self.size,
            self.size,
        )
    }
}

#[cfg(test)]
mod tests {
    use alg;
    use util::*;

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
        let mut raw = {
            let mut buffer = AlignedBuffer::new(alignment, matrices.len());

            for &matrix in matrices {
                buffer.push(matrix);
            }

            unsafe {
                buffer.finalize()
            }
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
