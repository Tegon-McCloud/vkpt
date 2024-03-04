use std::ops::Range;

use ash::{vk, prelude::VkResult};

use crate::{context::DeviceContext, util};

pub struct Buffer<'a> {
    context: &'a DeviceContext,
    inner: vk::Buffer,
    allocation: gpu_allocator::vulkan::Allocation,
}

impl<'a> Buffer<'a> {
    
    pub fn new(context: &'a DeviceContext, size: u64, usage: vk::BufferUsageFlags, location: gpu_allocator::MemoryLocation) -> VkResult<Self> {
        unsafe {
        
            let create_info = vk::BufferCreateInfo::builder()
                .flags(vk::BufferCreateFlags::empty())
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let inner = context.create_buffer(&create_info)?;

            let requirements = context.get_buffer_memory_requirements(inner);
            
            let allocation = context.allocator().lock()
                .expect("failed to lock allocator mutex")                
                .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                    name: "buffer allocation",
                    requirements,
                    location,
                    linear: true,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                })
                .expect("failed to allocate buffer memory");

            let bind_info = vk::BindBufferMemoryInfo::builder()
                .buffer(inner)
                .memory(allocation.memory())
                .memory_offset(allocation.offset())
                .build();
            
            context.bind_buffer_memory2(&[bind_info])?;

            Ok(Buffer {
                context,
                inner,
                allocation,
            })
        }
    }
    
    pub unsafe fn get_device_address(&self) -> vk::DeviceAddress {
        let info = vk::BufferDeviceAddressInfo::builder()
            .buffer(self.inner);

        self.context.get_buffer_device_address(&info)
    }

    pub unsafe fn get_device_or_host_address_const(&self) -> vk::DeviceOrHostAddressConstKHR {
        vk::DeviceOrHostAddressConstKHR { device_address: self.get_device_address() }
    }

    pub unsafe fn get_device_or_host_address(&self) -> vk::DeviceOrHostAddressKHR {
        vk::DeviceOrHostAddressKHR { device_address: self.get_device_address() }
    }
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        unsafe {
            self.context.destroy_buffer(self.inner);
            
            let mut allocation = gpu_allocator::vulkan::Allocation::default();
            std::mem::swap(&mut allocation, &mut self.allocation);
            self.context.allocator().lock()
                .expect("failed to lock allocator mutex")
                .free(allocation)
                .expect("failed to free buffer memory");
        }
    }
}


pub struct DeviceBuffer<'a> {
    buffer: Buffer<'a>,
}

impl<'a> DeviceBuffer<'a> {
    pub fn new(context: &'a DeviceContext, size: u64, usage: vk::BufferUsageFlags) -> VkResult<Self> {
        Ok(Self {
            buffer: Buffer::new(context, size, usage, gpu_allocator::MemoryLocation::GpuOnly)?,
        })
    }

    pub unsafe fn handle(&self) -> vk::Buffer {
        self.buffer.inner
    }

    pub unsafe fn get_device_address(&self) -> vk::DeviceAddress {
        self.buffer.get_device_address()
    }

    pub unsafe fn get_device_or_host_address_const(&self) -> vk::DeviceOrHostAddressConstKHR {
        self.buffer.get_device_or_host_address_const()
    }

    pub unsafe fn get_device_or_host_address(&self) -> vk::DeviceOrHostAddressKHR {
        self.buffer.get_device_or_host_address()
    }
}

pub struct UploadBuffer<'a> {
    buffer: Buffer<'a>,
}

impl<'a> UploadBuffer<'a> {
    pub fn new(context: &'a DeviceContext, size: u64, usage: vk::BufferUsageFlags) -> VkResult<Self> {
        Ok(Self {
            buffer: Buffer::new(context, size, usage, gpu_allocator::MemoryLocation::CpuToGpu)?,
        })
    }

    pub unsafe fn handle(&self) -> vk::Buffer {
        self.buffer.inner
    }

    pub unsafe fn get_device_address(&self) -> vk::DeviceAddress {
        self.buffer.get_device_address()
    }

    pub unsafe fn get_device_or_host_address_const(&self) -> vk::DeviceOrHostAddressConstKHR {
        self.buffer.get_device_or_host_address_const()
    }

    pub unsafe fn get_device_or_host_address(&self) -> vk::DeviceOrHostAddressKHR {
        self.buffer.get_device_or_host_address()
    }

    pub unsafe fn mapped_ptr(&self) -> *mut u8 {
        self.buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut u8
    }
   
    pub unsafe fn flush(&self) -> VkResult<()> {

        if !self.buffer.allocation.memory_properties().contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
            let non_coherent_atom_size = self.buffer.context.physical_device().properties.limits.non_coherent_atom_size;

            let range = vk::MappedMemoryRange::builder()
                .memory(self.buffer.allocation.memory())
                .offset(self.buffer.allocation.offset())
                .size(util::align_up(self.buffer.allocation.size(), non_coherent_atom_size))
                .build();
            
            self.buffer.context.flush_mapped_memory_ranges(&[range])?;
        }

        Ok(())
    }

    pub unsafe fn write_u8_slice(&mut self, src: &[u8], offset: usize) {
        debug_assert!(offset + src.len() <= self.buffer.allocation.size() as usize);
        
        let ptr = self.mapped_ptr();
        let dst = std::slice::from_raw_parts_mut(ptr.add(offset), src.len());

        dst.copy_from_slice(src);
    }

    // &T must be transmutable to a byte slice
    pub unsafe fn write<T>(&mut self, value: T, offset: usize) where
        T: Copy + 'static,
    {
        self.write_u8_slice(util::as_u8_slice(&value), offset)
    }

    // &[T] must be transmutable to a byte slice
    pub unsafe fn write_slice<T>(&mut self, slice: &[T], offset: usize) where
        T: Copy + 'static,
    {
        self.write_u8_slice(util::slice_as_u8_slice(slice), offset)
    }

    pub unsafe fn write_from_iter<I, T>(&mut self, iter: I, mut offset: usize) where
        T: Copy + 'static,
        I:  Iterator<Item = T>,
    {
        for item in iter {
            self.write(item, offset);
            offset += std::mem::size_of::<T>()
        }
    }
}

pub struct ReadBackBuffer<'a> {
    buffer: Buffer<'a>,
}

impl<'a> ReadBackBuffer<'a> {
    pub fn new(context: &'a DeviceContext, size: u64, usage: vk::BufferUsageFlags) -> VkResult<Self> {
        Ok(Self {
            buffer: Buffer::new(context, size, usage, gpu_allocator::MemoryLocation::GpuToCpu)?,
        })
    }

    pub unsafe fn handle(&self) -> vk::Buffer {
        self.buffer.inner
    }

    pub unsafe fn get_device_address(&self) -> vk::DeviceAddress {
        self.buffer.get_device_address()
    }

    pub unsafe fn get_device_or_host_address_const(&self) -> vk::DeviceOrHostAddressConstKHR {
        self.buffer.get_device_or_host_address_const()
    }

    pub unsafe fn get_device_or_host_address(&self) -> vk::DeviceOrHostAddressKHR {
        self.buffer.get_device_or_host_address()
    }

    pub unsafe fn mapped_ptr(&self) -> *const u8 {
        self.buffer.allocation.mapped_ptr().unwrap().as_ptr() as *const u8
    }
   
    pub unsafe fn invalidate(&self) -> VkResult<()> {

        if !self.buffer.allocation.memory_properties().contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
            let non_coherent_atom_size = self.buffer.context.physical_device().properties.limits.non_coherent_atom_size;

            let range = vk::MappedMemoryRange::builder()
                .memory(self.buffer.allocation.memory())
                .offset(self.buffer.allocation.offset())
                .size(util::align_up(self.buffer.allocation.size(), non_coherent_atom_size))
                .build();
            
            self.buffer.context.invalidate_mapped_memory_ranges(&[range])?;
        }

        Ok(())
    }

    pub unsafe fn read_u8_slice(&self, dst: &mut [u8], offset: usize) {
        debug_assert!(offset + dst.len() <= self.buffer.allocation.size() as usize);
        
        let ptr = self.mapped_ptr();
        let src = std::slice::from_raw_parts(ptr.add(offset), dst.len());

        dst.copy_from_slice(src);
    }

    pub unsafe fn read<T>(&self, reference: &mut T, offset: usize) {
        self.read_u8_slice(util::as_u8_slice_mut(reference), offset)
    }

    pub unsafe fn read_slice<T>(&self, slice: &mut [T], offset: usize) {
        self.read_u8_slice(util::slice_as_mut_u8_slice(slice), offset)
    }


}

// pub struct DeviceArray<'a, T> {
//     buffer: Buffer<'a>,
//     length: usize,
//     marker: std::marker::PhantomData<T>,
// }

// pub struct DeviceSlice<'a, 'b, T> {
//     buffer: &'b Buffer<'a>,
//     length: usize,
//     marker: std::marker::PhantomData<T>,
// }

pub struct Image<'ctx> {
    context: &'ctx DeviceContext,
    resolution: (u32, u32),
    inner: vk::Image,
    allocation: gpu_allocator::vulkan::Allocation,
}

impl<'a> Image<'a> {

    pub fn new(context: &'a DeviceContext, create_info: &vk::ImageCreateInfo, location: gpu_allocator::MemoryLocation) -> VkResult<Self> {

        let inner = unsafe { context.device().create_image(&create_info, None)? };

        let memory_requirements = unsafe { context.device().get_image_memory_requirements(inner) };

        let allocation = context.allocator().lock()
            .expect("failed to lock allocator mutex")
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "image allocation",
                requirements: memory_requirements,
                location,
                linear: create_info.tiling == vk::ImageTiling::LINEAR,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .expect("failed to allocate memory for image");

        unsafe {
            context.device().bind_image_memory(inner, allocation.memory(), allocation.offset()).unwrap();
        }

        let resolution = (create_info.extent.width, create_info.extent.height);

        Ok(Self {
            context,
            resolution,
            inner,
            allocation
        })
    }

    pub fn resolution(&self) -> (u32, u32) {
        self.resolution
    }

    pub unsafe fn inner(&self) -> vk::Image {
        self.inner
    }

}

impl Drop for Image<'_> {
    fn drop(&mut self) {
        unsafe {
            self.context.device().destroy_image(self.inner, None);
            
            let mut allocation = gpu_allocator::vulkan::Allocation::default();
            std::mem::swap(&mut allocation, &mut self.allocation);
            self.context.allocator().lock()
                .expect("failed to lock allocator mutex")
                .free(allocation)
                .expect("failed to free image memory");
        }
    }
}


pub struct ImageView<'ctx> {
    context: &'ctx DeviceContext,
    inner: vk::ImageView,
}

impl<'ctx> ImageView<'ctx> {
    pub fn new(image: &Image<'ctx>, format: vk::Format, mip_levels: Range<u32>, array_layers: Range<u32>) -> VkResult<Self> {
        let info = vk::ImageViewCreateInfo::builder()
        .image(unsafe { image.inner() })
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .components(vk::ComponentMapping {
            r: vk::ComponentSwizzle::R,
            g: vk::ComponentSwizzle::G,
            b: vk::ComponentSwizzle::B,
            a: vk::ComponentSwizzle::A,
        })
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: mip_levels.start,
            level_count: mip_levels.end - mip_levels.start,
            base_array_layer: array_layers.start,
            layer_count: array_layers.end - array_layers.start,
        });

        let inner = unsafe { image.context.device().create_image_view(&info, None)? };

        Ok(Self {
            context: image.context,
            inner,
        })
    }

    pub unsafe fn inner(&self) -> vk::ImageView {
        self.inner
    }

}

impl<'ctx> Drop for ImageView<'ctx> {
    fn drop(&mut self) {
        unsafe {
            self.context.device().destroy_image_view(self.inner, None);
        }
    }
}

