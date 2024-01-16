
use ash::{vk::{self, AccelerationStructureGeometryTrianglesDataKHR}, prelude::VkResult};

use crate::{context::DeviceContext, resource::{DeviceBuffer, UploadBuffer}};

// pub struct MeshFactory<'a> {
//     context: &'a DeviceContext,



//     staging_buffer: Buffer<'a>,
//     scratch_buffer: Buffer<'a>,
// }

// impl<'a> MeshFactory<'a> {

// }

pub struct Mesh<'a> {
    context: &'a DeviceContext,
    #[allow(unused)]
    vertex_buffer: DeviceBuffer<'a>,
    #[allow(unused)]
    index_buffer: DeviceBuffer<'a>,
    #[allow(unused)]
    accel_buffer: DeviceBuffer<'a>,
    accel_structure: vk::AccelerationStructureKHR,
}

impl<'a> Mesh<'a> {

    pub unsafe fn new<PI, FI>(
        position_iter: PI,
        face_iter: FI,
        context: &'a DeviceContext
    ) -> VkResult<Self> where
        PI: ExactSizeIterator<Item=[f32; 3]>,
        FI: ExactSizeIterator<Item=[u32; 3]>
    {

        let face_count = face_iter.len();
        let vertex_count = position_iter.len();
        
        let (index_buffer, vertex_buffer) = Self::create_and_fill_geometry_buffers(
            position_iter,
            face_iter,
            context
        )?;

        let (accel_buffer, accel_structure) = Self::create_and_build_acceleration_structure(
            &index_buffer,
            &vertex_buffer, 
            face_count,
            vertex_count,
            context,
        )?;


        Ok(Self {
            context,
            vertex_buffer,
            index_buffer,
            accel_buffer,
            accel_structure,
        })
    }

    unsafe fn create_and_fill_geometry_buffers<PI, FI>(
        position_iter: PI,
        face_iter: FI,
        context: &'a DeviceContext,
    ) -> VkResult<(DeviceBuffer, DeviceBuffer)> where
        PI: ExactSizeIterator<Item=[f32; 3]>,
        FI: ExactSizeIterator<Item=[u32; 3]> 
    {
        let index_buffer_size = face_iter.len() * std::mem::size_of::<[u32; 3]>();
        let vertex_buffer_size = position_iter.len() * std::mem::size_of::<[f32; 3]>();

        let index_buffer = DeviceBuffer::new(
            context,
            index_buffer_size as u64,
            vk::BufferUsageFlags::TRANSFER_DST |
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS |
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
        )?;
        let vertex_buffer = DeviceBuffer::new(
            context,
            vertex_buffer_size as u64,
            vk::BufferUsageFlags::TRANSFER_DST |
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS |
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
        )?;
        
        let mut staging_buffer = UploadBuffer::new(
            context,
            (index_buffer_size + vertex_buffer_size) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC
        )?;

        let face_offset = 0;
        let position_offset = index_buffer_size;

        staging_buffer.write_from_iter(face_iter, face_offset);
        staging_buffer.write_from_iter(position_iter, position_offset);

        staging_buffer.flush()?;

        context.execute_commands(|cmd_buffer| {

            let index_region = vk::BufferCopy2::builder()
                .src_offset(0)
                .dst_offset(0)
                .size(index_buffer_size as u64)
                .build();

            let index_copy_info = vk::CopyBufferInfo2::builder()
                .src_buffer(staging_buffer.handle())
                .dst_buffer(index_buffer.handle())
                .regions(std::slice::from_ref(&index_region));

            context.device().cmd_copy_buffer2(cmd_buffer, &index_copy_info);
            
            let vertex_region = vk::BufferCopy2::builder()
                .src_offset(index_buffer_size as u64)
                .dst_offset(0)
                .size(vertex_buffer_size as u64)
                .build();

            let vertex_copy_info = vk::CopyBufferInfo2::builder()
                .src_buffer(staging_buffer.handle())
                .dst_buffer(vertex_buffer.handle())
                .regions(std::slice::from_ref(&vertex_region));
            
            context.device().cmd_copy_buffer2(cmd_buffer, &vertex_copy_info);

            // prevent acceleration structure build until upload has finished
            let index_barrier = vk::BufferMemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .buffer(index_buffer.handle())
                .offset(0)
                .size(vk::WHOLE_SIZE)
                .build();

            let vertex_barrier = vk::BufferMemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .buffer(index_buffer.handle())
                .offset(0)
                .size(vk::WHOLE_SIZE)
                .build();

            let buffer_barriers = [index_barrier, vertex_barrier];

            let dependency_info = vk::DependencyInfo::builder()
                .buffer_memory_barriers(&buffer_barriers);

            context.device().cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
        })?;

        Ok((index_buffer, vertex_buffer))
    }


    unsafe fn create_and_build_acceleration_structure(
        index_buffer: &DeviceBuffer,
        vertex_buffer: &DeviceBuffer,
        face_count: usize,
        vertex_count: usize,
        context: &'a DeviceContext,
    ) -> VkResult<(DeviceBuffer<'a>, vk::AccelerationStructureKHR)> {

        let accel_geometry_data = vk::AccelerationStructureGeometryDataKHR {
            triangles: AccelerationStructureGeometryTrianglesDataKHR::builder()
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .vertex_data(vertex_buffer.get_device_or_host_address_const())
                .vertex_stride(std::mem::size_of::<[f32; 3]>() as u64)
                .max_vertex(vertex_count as u32 - 1)
                .index_type(vk::IndexType::UINT32)
                .index_data(index_buffer.get_device_or_host_address_const())
                .build(),
        };
        
        let accel_geometry = vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(accel_geometry_data);
        
        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&accel_geometry))
            .build();

        let accel_build_sizes = context.extensions().acceleration_structure
            .get_acceleration_structure_build_sizes(vk::AccelerationStructureBuildTypeKHR::DEVICE, &build_info, &[face_count as u32]);
    
        println!("Blas buffer size: {}", accel_build_sizes.acceleration_structure_size);

        let accel_buffer = DeviceBuffer::new(
            context,
            accel_build_sizes.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
        )?;

        let accel_info = vk::AccelerationStructureCreateInfoKHR::builder()
            .buffer(accel_buffer.handle())
            .size(accel_build_sizes.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
        
        let accel_structure = context.extensions().acceleration_structure.create_acceleration_structure(&accel_info, None)?;
        
        let scratch_buffer = DeviceBuffer::new(
            context,
            accel_build_sizes.build_scratch_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;

        build_info.dst_acceleration_structure = accel_structure;
        build_info.scratch_data = scratch_buffer.get_device_or_host_address();

        let accel_build_range = vk::AccelerationStructureBuildRangeInfoKHR::builder()
            .primitive_offset(0)
            .primitive_count(face_count as u32)
            .first_vertex(0)
            .transform_offset(0)
            .build();

        let accel_build_ranges = [std::slice::from_ref(&accel_build_range)];

        context.execute_commands(|cmd_buffer| {
            context.extensions().acceleration_structure.cmd_build_acceleration_structures(
                cmd_buffer,
                std::slice::from_ref(&build_info),
                &accel_build_ranges,
            );

            let buffer_barrier = vk::BufferMemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .buffer(accel_buffer.handle())
                .offset(0)
                .size(vk::WHOLE_SIZE)
                .build();

            let dependency_info = vk::DependencyInfo::builder()
                .buffer_memory_barriers(std::slice::from_ref(&buffer_barrier));
            
            context.device().cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
        })?;
        
        Ok((accel_buffer, accel_structure))
    }

    pub unsafe fn get_accel_structure_device_address(&self) -> vk::DeviceAddress {
        let info = vk::AccelerationStructureDeviceAddressInfoKHR::builder()
            .acceleration_structure(self.accel_structure);

        self.context.extensions().acceleration_structure.get_acceleration_structure_device_address(&info)
    }

}

impl Drop for Mesh<'_> {   
    fn drop(&mut self) {
        unsafe {
            self.context.extensions().acceleration_structure.destroy_acceleration_structure(self.accel_structure, None);
        }
    }
}





