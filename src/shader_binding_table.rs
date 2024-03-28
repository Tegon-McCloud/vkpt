
use std::ops::Range;

use ash::{prelude::VkResult, vk::{self, StridedDeviceAddressRegionKHR}};
use crate::{context::DeviceContext, pipeline::Pipeline, resource::UploadBuffer, util};

// struct PipelineReference {
//     pipeline: vk::Pipeline,
//     shader_group_count: u32,
// }

#[derive(Debug, Clone, Copy)]
struct AlignmentInfo {
    handle_size: u64,
    handle_align: u64,
    region_align: u64,
}

impl AlignmentInfo {
    fn for_context(context: &DeviceContext) -> Self {
        Self {
            handle_size: context.physical_device().ray_tracing_pipeline_properties.shader_group_handle_size as u64,
            handle_align: context.physical_device().ray_tracing_pipeline_properties.shader_group_handle_alignment as u64,
            region_align: context.physical_device().ray_tracing_pipeline_properties.shader_group_base_alignment as u64,
        }
    }

    fn aligned_size_of_entry(&self, entry: &Entry) -> u64 {
        let entry_size = self.handle_size + entry.data_range.len() as u64;
        util::align_up(entry_size, self.handle_align)
    }

    fn calculate_stride_and_size(&self, entries: &[Entry]) -> (u64, u64) {
        let stride = entries
            .iter()
            .fold(0, |stride, entry| stride.max(self.aligned_size_of_entry(entry)));

        let size = util::align_up(stride * entries.len() as u64, self.region_align);

        (stride, size)
    }
}

struct Entry {
    shader_group: u32,
    data_range: Range<usize>,
}

impl Entry {
    fn get_handle<'a>(&self, align_info: AlignmentInfo, handle_data: &'a [u8]) -> &'a [u8] {
        
        let begin = self.shader_group as usize * align_info.handle_size as usize;
        let end = (self.shader_group + 1) as usize * align_info.handle_size as usize;

        &handle_data[begin..end]
    }

    fn get_data<'a>(&self, data: &'a [u8]) -> &'a [u8] {
        &data[self.data_range.clone()]
    }
}

pub struct ShaderBindingTableDescription {
    entries: [Vec<Entry>; 4],
    data: Vec<u8>,
}

impl<'ctx> ShaderBindingTableDescription {

    pub fn new() -> Self {
        Self {
            entries: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            data: Vec::new(),
        }
    }

    pub fn push_raygen_entry(&mut self, shader_group: u32, data: &[u8]) {
        self.push_entry(0, shader_group, data);
    }

    pub fn push_miss_entry(&mut self, shader_group: u32, data: &[u8]) {
        self.push_entry(1, shader_group, data);
    }

    pub fn push_hit_group_entry(&mut self, shader_group: u32, data: &[u8]) {
        self.push_entry(2, shader_group, data);
    }

    pub fn push_callable_entry(&mut self, shader_group: u32, data: &[u8]) {
        self.push_entry(3, shader_group, data);
    }

    fn push_entry(&mut self, index: usize, shader_group: u32, data: &[u8]) {
        let data_start = self.data.len();
        let data_end = data_start + data.len();

        self.data.extend(data);

        self.entries[index].push(Entry { shader_group, data_range: data_start..data_end });
    }

    pub unsafe fn build(&self, context: &'ctx DeviceContext, handle_data: &[u8]) -> VkResult<ShaderBindingTable<'ctx>> {

        let align_info = AlignmentInfo::for_context(context);

        let mut strides = [0; 4];
        let mut sizes = [0; 4];
        let mut offsets = [0; 4];

        let mut curr_offset = 0;

        for i in 0..4 {
            
            let (stride, size) = align_info.calculate_stride_and_size(&self.entries[i]);
            
            strides[i] = stride;
            sizes[i] = size;
            offsets[i] = curr_offset;

            curr_offset += size;
        }

        let buffer_size = curr_offset;

        let mut buffer = UploadBuffer::new(
            context, 
            buffer_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
        )?;
        
        for (i, entries) in self.entries.iter().enumerate() {
            for (j, entry) in entries.iter().enumerate() {
                
                let entry_handle = entry.get_handle(align_info, handle_data);
                let entry_data = entry.get_data(&self.data);
                
                let handle_offset = offsets[i] as usize + j * strides[i] as usize;
                let data_offset = handle_offset + align_info.handle_size as usize;
 

                unsafe {
                    buffer.write_u8_slice(entry_handle, handle_offset);
                    buffer.write_u8_slice(entry_data, data_offset)
                }
            }
        }
        unsafe {
            buffer.flush()?;
        }
        
        let buffer_address = unsafe { buffer.get_device_address() };

        Ok(ShaderBindingTable {
            buffer,

            raygen_region: StridedDeviceAddressRegionKHR {
                device_address: buffer_address + offsets[0],
                stride: strides[0],
                size: strides[0],
            },

            miss_region: StridedDeviceAddressRegionKHR {
                device_address: buffer_address + offsets[1],
                stride: strides[1],
                size: strides[1],
            },

            hit_group_region: StridedDeviceAddressRegionKHR {
                device_address: buffer_address + offsets[2],
                stride: strides[2],
                size: strides[2],
            },
            
            callable_region: StridedDeviceAddressRegionKHR {
                device_address: buffer_address + offsets[3],
                stride: strides[3],
                size: strides[3],
            },
        })
    }
}

pub struct ShaderBindingTable<'a> {
    #[allow(unused)]
    buffer: UploadBuffer<'a>,
    
    raygen_region: vk::StridedDeviceAddressRegionKHR,
    miss_region: vk::StridedDeviceAddressRegionKHR,
    hit_group_region: vk::StridedDeviceAddressRegionKHR,
    callable_region: vk::StridedDeviceAddressRegionKHR,
}

impl<'a> ShaderBindingTable<'a> {

    pub unsafe fn raygen_region(&self) -> vk::StridedDeviceAddressRegionKHR {
        self.raygen_region
    }

    pub unsafe fn miss_region(&self) -> vk::StridedDeviceAddressRegionKHR {
        self.miss_region
    }

    pub unsafe fn hit_group_region(&self) -> vk::StridedDeviceAddressRegionKHR {
        self.hit_group_region
    }

    pub unsafe fn callable_region(&self) -> vk::StridedDeviceAddressRegionKHR {
        self.callable_region
    }
}
