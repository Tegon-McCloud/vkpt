use ash::{prelude::VkResult, vk::{self, StridedDeviceAddressRegionKHR}};

use crate::{context::DeviceContext, resource::UploadBuffer, util};

// struct PipelineLibrary {
//     pipeline: vk::Pipeline,
//     shader_group_count: u64,
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
        let entry_size = self.handle_size + entry.data_size as u64;
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
    data_offset: usize,
    data_size: usize,
}

pub struct ShaderBindingTableBuilder<'a> {
    context: &'a DeviceContext,
    
    entries: [Vec<Entry>; 4],
    data: Vec<u8>,
}

impl<'a> ShaderBindingTableBuilder<'a> {

    pub fn new(context: &'a DeviceContext) -> Self {
        Self {
            context,
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

    pub fn build(&self, pipeline: vk::Pipeline, group_count: u32) -> VkResult<ShaderBindingTable> {
    
        let align_info = AlignmentInfo::for_context(self.context);

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
            self.context, 
            buffer_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
        )?;

        let handle_data_size = group_count as u64 * align_info.handle_size;
        let handle_data = unsafe { self.context.extensions()
                .ray_tracing_pipeline
                .get_ray_tracing_shader_group_handles(pipeline, 0, group_count, handle_data_size as usize)? };

        let get_handle = |shader_group: u32| -> &[u8] {
            let handle_begin = shader_group as usize * align_info.handle_size as usize;
            let handle_end = handle_begin + align_info.handle_size as usize;
            
            &handle_data[handle_begin..handle_end]
        };
        
        for (i, entries) in self.entries.iter().enumerate() {
            for (j, entry) in entries.iter().enumerate() {
                
                let entry_data = &self.data[entry.data_offset..entry.data_offset + entry.data_size];
                
                let handle_offset = offsets[i] as usize + j * strides[i] as usize;
                let data_offset = handle_offset + align_info.handle_size as usize;

                unsafe {
                    buffer.write_u8_slice(get_handle(entry.shader_group), handle_offset);
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

    fn push_entry(&mut self, index: usize, shader_group: u32, data: &[u8]) {
        let data_offset = self.data.len();
        let data_size = data.len();

        self.data.extend(data);

        self.entries[index].push(Entry { shader_group, data_offset, data_size });
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
