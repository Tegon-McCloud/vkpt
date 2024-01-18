use ash::vk;
use nalgebra::Matrix3x4;


pub fn align_up(size: u64, align: u64) -> u64 {
    ((size + align - 1) / align) * align 
}

pub fn region_offsets<const N: usize>(region_sizes: [u64; N], region_align: u64) -> ([u64; N], u64) {

    let mut offset = 0;
    let mut offsets = [0; N];

    for i in 0..N {
        offsets[i] = offset;
        offset = align_up(offset + region_sizes[i], region_align);
    }

    let total_size = offset;
    (offsets, total_size)
}

pub unsafe fn as_u8_slice<'a, T>(reference: &'a T) -> &'a [u8] {
    std::slice::from_raw_parts(reference as *const T as *const u8, std::mem::size_of::<T>())
}

pub unsafe fn as_u8_slice_mut<'a, T>(reference: &'a mut T) -> &'a mut [u8] {
    std::slice::from_raw_parts_mut(reference as *mut T as *mut u8, std::mem::size_of::<T>())
}

pub unsafe fn slice_as_u8_slice<'a, T>(slice: &'a [T]) -> &'a [u8] {
    std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * std::mem::size_of::<T>())
}

pub unsafe fn slice_as_mut_u8_slice<'a, T>(slice: &'a mut [T]) -> &'a mut [u8] {
    std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u8, slice.len() * std::mem::size_of::<T>())
}

pub fn matrix_to_vk_transform(matrix: Matrix3x4<f32>) -> vk::TransformMatrixKHR {
    vk::TransformMatrixKHR {
        matrix: [
            matrix[(0, 0)], matrix[(0, 1)], matrix[(0, 2)], matrix[(0, 3)],
            matrix[(1, 0)], matrix[(1, 1)], matrix[(1, 2)], matrix[(1, 3)],
            matrix[(2, 0)], matrix[(2, 1)], matrix[(2, 2)], matrix[(2, 3)],
        ]
    }
}