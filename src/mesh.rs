
use ash::{vk::{self, AccelerationStructureGeometryTrianglesDataKHR}, prelude::VkResult};

use crate::{context::DeviceContext, resource::{DeviceBuffer, UploadBuffer}, util};

// pub struct AsyncMeshFactory<'a> {
//     context: &'a DeviceContext,



//     staging_buffer: UploadBuffer<'a>,
//     scratch_buffer: DeviceBuffer<'a>,
// }

// impl<'a> MeshFactory<'a> {

//}

struct MeshGeometry<'a> {
    #[allow(unused)]
    buffer: DeviceBuffer<'a>,
    index_region: vk::DeviceAddress,
    position_region: vk::DeviceAddress,
    normal_region: vk::DeviceAddress,
    
    face_count: u64,
    vertex_count: u64,
}

pub struct Mesh<'a> {
    context: &'a DeviceContext,
    geometry: MeshGeometry<'a>,

    #[allow(unused)]
    accel_buffer: DeviceBuffer<'a>,
    accel_structure: vk::AccelerationStructureKHR,
}

impl<'a> Mesh<'a> {

    pub unsafe fn new<FI, PI, NI>(
        context: &'a DeviceContext,
        face_iter: FI,
        position_iter: PI,
        normal_iter: NI,
    ) -> VkResult<Self> where
        FI: ExactSizeIterator<Item=[u32; 3]>,
        PI: ExactSizeIterator<Item=[f32; 3]>,
        NI: ExactSizeIterator<Item=[f32; 3]>,
    {
        let geometry = Self::create_geometry(
            context,
            face_iter,
            position_iter,
            normal_iter,
        )?;

        let (accel_buffer, accel_structure) = Self::create_and_build_acceleration_structure(
            context,
            &geometry,
        )?;

        Ok(Self {
            context,
            geometry,
            accel_buffer,
            accel_structure,
        })
    }

    unsafe fn create_geometry<FI, PI, NI>(
        context: &'a DeviceContext,
        face_iter: FI,
        position_iter: PI,
        normal_iter: NI,
    ) -> VkResult<MeshGeometry<'a>> where
        FI: ExactSizeIterator<Item=[u32; 3]>,
        PI: ExactSizeIterator<Item=[f32; 3]>,
        NI: ExactSizeIterator<Item=[f32; 3]>,
    {
        let face_count = face_iter.len() as u64;
        let vertex_count = position_iter.len() as u64;

        let position_iter = position_iter.map(|p| [p[0], p[1], p[2], 1.0]);
        let normal_iter = normal_iter.map(|n| [n[0], n[1], n[2], 0.0]);

        let region_sizes = [
            (face_iter.len() * std::mem::size_of::<[u32; 3]>()) as u64,
            (position_iter.len() * std::mem::size_of::<[f32; 4]>()) as u64,
            (normal_iter.len() * std::mem::size_of::<[f32; 4]>()) as u64,
        ];

        let (region_offsets, buffer_size) = util::region_offsets(region_sizes, 16);
        
        let buffer = DeviceBuffer::new(
            context,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST |
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS |
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
        )?;

        let mut staging_buffer = UploadBuffer::new(
            context,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC
        )?;

        staging_buffer.write_from_iter(face_iter, region_offsets[0] as usize);
        staging_buffer.write_from_iter(position_iter, region_offsets[1] as usize);
        staging_buffer.write_from_iter(normal_iter, region_offsets[2] as usize);
        
        staging_buffer.flush()?;

        context.execute_commands(|cmd_buffer| {
            let copy_region = vk::BufferCopy2::builder()
                .src_offset(0)
                .dst_offset(0)
                .size(buffer_size)
                .build();

            let copy_info = vk::CopyBufferInfo2::builder()
                .src_buffer(staging_buffer.handle())
                .dst_buffer(buffer.handle())
                .regions(std::slice::from_ref(&copy_region));

            context.device().cmd_copy_buffer2(cmd_buffer, &copy_info);

            // prevent acceleration structure build until upload has finished
            let barrier = vk::BufferMemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .buffer(buffer.handle())
                .offset(0)
                .size(buffer_size)
                .build();

            let dependency_info = vk::DependencyInfo::builder()
                .buffer_memory_barriers(std::slice::from_ref(&barrier));

            context.device().cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
        })?;

        let device_address = buffer.get_device_address();

        Ok(MeshGeometry {
            buffer,
            index_region: device_address + region_offsets[0],
            position_region: device_address + region_offsets[1],
            normal_region: device_address + region_offsets[2],
            face_count,
            vertex_count,
        })
    }


    unsafe fn create_and_build_acceleration_structure(
        context: &'a DeviceContext,
        geometry: &MeshGeometry<'a>,
    ) -> VkResult<(DeviceBuffer<'a>, vk::AccelerationStructureKHR)> {

        let accel_geometry_data = vk::AccelerationStructureGeometryDataKHR {
            triangles: AccelerationStructureGeometryTrianglesDataKHR::builder()
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: geometry.position_region })
                .vertex_stride(std::mem::size_of::<[f32; 4]>() as u64)
                .max_vertex(geometry.vertex_count as u32 - 1)
                .index_type(vk::IndexType::UINT32)
                .index_data(vk::DeviceOrHostAddressConstKHR { device_address: geometry.index_region })
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
            .get_acceleration_structure_build_sizes(vk::AccelerationStructureBuildTypeKHR::DEVICE, &build_info, &[geometry.face_count as u32]);
    
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
            .primitive_count(geometry.face_count as u32)
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

    pub unsafe fn index_address(&self) -> vk::DeviceAddress {
        self.geometry.index_region
    }

    pub unsafe fn position_address(&self) -> vk::DeviceAddress {
        self.geometry.position_region
    }

    pub unsafe fn normal_address(&self) -> vk::DeviceAddress {
        self.geometry.normal_region
    }

}

impl Drop for Mesh<'_> {   
    fn drop(&mut self) {
        unsafe {
            self.context.extensions().acceleration_structure.destroy_acceleration_structure(self.accel_structure, None);
        }
    }
}





