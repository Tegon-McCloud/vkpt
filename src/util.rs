
pub fn align_up(size: u64, align: u64) -> u64 {
    ((size + align - 1) / align) * align 
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

