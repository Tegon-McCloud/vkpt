#![feature(iter_array_chunks)]
#![feature(int_roundings)]
#![feature(once_cell_try)]

use std::{path::Path, fs::File, ffi::CStr, os::windows::fs::MetadataExt, io::Read};

use ash::{vk::{self, Packed24_8}, prelude::VkResult};
use itertools::Itertools;

use context::DeviceContext;
use resource::{Image, ReadBackBuffer};
use mesh::Mesh;
use shader_binding_table::ShaderBindingTableBuilder;

use crate::resource::{UploadBuffer, DeviceBuffer};
pub mod util;
pub mod context;
pub mod resource;
pub mod uploader;
pub mod descriptor;
pub mod mesh;
pub mod acceleration;
pub mod shader_binding_table;
pub mod surface;


pub fn load_scene<P: AsRef<Path>>(path: P, context: &DeviceContext) -> VkResult<()> {

    let (document, buffers, _images) = gltf::import(path).expect("failed to import scene");

    let mut meshes = Vec::new();

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            let face_iter = reader.read_indices()
                .unwrap()
                .into_u32()
                .array_chunks();

            let position_iter = reader.read_positions().unwrap();
            // let normal_iter = reader.read_normals().unwrap();
            unsafe {
                meshes.push(Mesh::new(position_iter, face_iter, context)?);
            }
        }
    }

    if let Some(_scene) = document.default_scene() {

    }

    Ok(())

}

pub fn build_top_level_accel_structure<'a>(context: &'a DeviceContext, mesh: &Mesh<'a>) -> VkResult<(vk::AccelerationStructureKHR, DeviceBuffer<'a>)> {

    unsafe {

        let identity_matrix = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ];

        let instance = vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR { matrix: identity_matrix },
            instance_custom_index_and_mask: Packed24_8::new(0, 0xff),
            instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(0, vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: mesh.get_accel_structure_device_address()
            },
        };  

        let mut instance_buffer = UploadBuffer::new(
            &context,
            std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() as vk::DeviceSize,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
        )?;

        instance_buffer.write(instance, 0);
        instance_buffer.flush()?;

        let accel_geometry_data = vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR::builder()
                .array_of_pointers(false)
                .data(instance_buffer.get_device_or_host_address_const())
                .build(),
        };

        let accel_geometry = vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(accel_geometry_data);

        let mut accel_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&accel_geometry))
            .build();

        let accel_sizes = context.extensions().acceleration_structure.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &accel_build_info,
            &[1],
        );

        println!("Tlas buffer size: {}", accel_sizes.acceleration_structure_size);

        let accel_buffer = DeviceBuffer::new(
            context,
            accel_sizes.acceleration_structure_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        )?;

        let accel_info = vk::AccelerationStructureCreateInfoKHR::builder()
            .buffer(accel_buffer.handle())
            .size(accel_sizes.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);

        let accel_structure = context.extensions().acceleration_structure.create_acceleration_structure(&accel_info, None).unwrap();

        let scratch_buffer = DeviceBuffer::new(
            context,
            accel_sizes.build_scratch_size, 
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;

        accel_build_info.scratch_data = scratch_buffer.get_device_or_host_address();
        accel_build_info.dst_acceleration_structure = accel_structure;

        let accel_build_range = vk::AccelerationStructureBuildRangeInfoKHR::builder()
            .primitive_count(1)
            .build();

        let accel_build_ranges = [std::slice::from_ref(&accel_build_range)];

        context.execute_commands(|cmd_buffer| {
            context.extensions().acceleration_structure.cmd_build_acceleration_structures(
                cmd_buffer,
                std::slice::from_ref(&accel_build_info),
                &accel_build_ranges,
            );

            let barrier = vk::BufferMemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR)
                .buffer(accel_buffer.handle())
                .offset(0)
                .size(vk::WHOLE_SIZE)
                .build();

            let dependency_info = vk::DependencyInfo::builder()
                .buffer_memory_barriers(std::slice::from_ref(&barrier));

            context.device().cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
        })?;

        Ok((accel_structure, accel_buffer))
    }
}

pub fn create_descriptor_set<'a>(
    context: &'a DeviceContext,
    accel_structure: vk::AccelerationStructureKHR,
    output_view: vk::ImageView,
) -> VkResult<(vk::DescriptorSet, vk::DescriptorSetLayout, vk::DescriptorPool)> {
    unsafe {

        let pool_sizes = [
            vk::DescriptorPoolSize::builder()
                .descriptor_count(1)
                .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .build(),

            vk::DescriptorPoolSize::builder()
                .descriptor_count(1)
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .build(),
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(1)
            .pool_sizes(&pool_sizes);

        let pool = context.device().create_descriptor_pool(&pool_info, None)?;

        let layout_bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                .build(),

            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                .build(),
        ];

        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&layout_bindings);

        let set_layout = context.device().create_descriptor_set_layout(&set_layout_info, None)?;

        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(pool)
            .set_layouts(std::slice::from_ref(&set_layout));

        let set = context.device().allocate_descriptor_sets(&alloc_info)?[0];

        let mut accel_structure_write = vk::WriteDescriptorSetAccelerationStructureKHR::builder()
            .acceleration_structures(std::slice::from_ref(&accel_structure))
            .build();

        let mut accel_write = vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .push_next(&mut accel_structure_write)
            .build();

        accel_write.descriptor_count = 1;

        let output_image_info = vk::DescriptorImageInfo::builder()
            .image_view(output_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .sampler(vk::Sampler::null())
            .build();

        let output_image_write = vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&output_image_info))
            .build();

        context.device().update_descriptor_sets(&[accel_write, output_image_write], &[]);

        Ok((set, set_layout, pool))
    }


}

pub unsafe fn create_shader_module<P: AsRef<Path>>(spv_file: P, context: &DeviceContext) -> VkResult<vk::ShaderModule> {

    let mut file = File::open(spv_file).expect("failed to open SPIR-V file");

    let spv_size = file.metadata().unwrap().file_size() as usize;
    let u32_buffer_size = spv_size.div_ceil(std::mem::size_of::<u32>());

    let mut buffer = vec![0u32; u32_buffer_size];

    file.read(util::slice_as_mut_u8_slice(buffer.as_mut_slice())).expect("failed to read SPIR-V file");

    let info = vk::ShaderModuleCreateInfo::builder()
        .code(&buffer);

    context.device().create_shader_module(&info, None)
}

pub fn general_shader_group_info(index: u32) -> vk::RayTracingShaderGroupCreateInfoKHR {
    vk::RayTracingShaderGroupCreateInfoKHR::builder()
        .any_hit_shader(vk::SHADER_UNUSED_KHR)
        .closest_hit_shader(vk::SHADER_UNUSED_KHR)
        .intersection_shader(vk::SHADER_UNUSED_KHR)
        .general_shader(index)
        .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
        .build()
}


// pub fn render(render_target: vk::Image, mesh: &Mesh<'_>, context: &DeviceContext) -> VkResult<()> {
//     unsafe {

//         context.execute_commands(|cmd_buffer| {


//             context.extensions().ray_tracing_pipeline.cmd_trace_rays(
//                 cmd_buffer,
//                 , miss_shader_binding_tables, hit_shader_binding_tables, callable_shader_binding_tables, width, height, depth)

//         })?;


//         let cmd_pool_info = vk::CommandPoolCreateInfo::builder()
//             .queue_family_index(context.queue_family())
//             .flags(vk::CommandPoolCreateFlags::default());
//         let cmd_pool = context.device().create_command_pool(&cmd_pool_info, None)?;

//         let cmd_buffer_info = vk::CommandBufferAllocateInfo::builder()
//             .command_buffer_count(1)
//             .command_pool(cmd_pool)
//             .level(vk::CommandBufferLevel::PRIMARY);
//         let cmd_buffer = context.device().allocate_command_buffers(&cmd_buffer_info)?[0];

//         let fence_info = vk::FenceCreateInfo::builder()
//             .flags(vk::FenceCreateFlags::default());
//         let fence = context.device().create_fence(&fence_info, None)?;

//         let begin_info = vk::CommandBufferBeginInfo::builder()
//             .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

//         context.device().begin_command_buffer(cmd_buffer, &begin_info)?;

//         context.device().end_command_buffer(cmd_buffer)?;

//         context.device().destroy_fence(fence, None);
//         context.device().destroy_command_pool(cmd_pool, None);
//     }

//     Ok(())
// }


struct SampleTarget<'a> {
    #[allow(unused)]
    context: &'a DeviceContext,
    image: Image<'a>,

}

impl<'a> SampleTarget<'a> {

    pub fn new(context: &'a DeviceContext, width: u32, height: u32) -> VkResult<Self> {
        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST);

        let image = Image::new(context, &image_info, gpu_allocator::MemoryLocation::GpuOnly)?;

        Ok(Self {
            context,
            image,
        })

    }

    pub unsafe fn full_view_info(&self) -> vk::ImageViewCreateInfo {

        vk::ImageViewCreateInfo::builder()
        .image(self.image.inner)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(vk::Format::R32G32B32A32_SFLOAT)
        .components(vk::ComponentMapping::builder()
            .r(vk::ComponentSwizzle::R)
            .g(vk::ComponentSwizzle::G)
            .b(vk::ComponentSwizzle::B)
            .a(vk::ComponentSwizzle::A)
            .build()
        )
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .build()
    }


}



// unsafe fn copy_mapped_image_to_vec(image: Image) -> Vec<f32> {


// }


fn main() {

    let context = DeviceContext::new().expect("failed to create device context");

    // load_scene("./resources/bunny.gltf", &context).unwrap();

    let mesh = unsafe {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];

        let indices = [
            [0, 1, 2]
        ];

        Mesh::new(positions.iter().copied(), indices.iter().copied(), &context).unwrap()
    };

    let entry_point_name = unsafe {
        CStr::from_bytes_with_nul_unchecked(b"main\0")
    };

    let (accel_structure, _accel_buffer) = build_top_level_accel_structure(&context, &mesh).unwrap();

    let img_width = 512;
    let img_height = 512;

    let sample_target = SampleTarget::new(&context, img_width, img_height).unwrap();
    let sample_view = unsafe { context.device().create_image_view(&sample_target.full_view_info(), None).unwrap() };
    
    let (descriptor_set, descriptor_set_layout, descriptor_pool) = create_descriptor_set(&context, accel_structure, sample_view).unwrap();

    let raygen_module = unsafe { create_shader_module("shader_bin/raytrace.rgen.spv", &context).unwrap() };
    let miss_module = unsafe { create_shader_module("shader_bin/raytrace.rmiss.spv", &context).unwrap() };
    let closest_hit_module = unsafe { create_shader_module("shader_bin/raytrace.rchit.spv", &context).unwrap() };

    let raygen_shader_stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::RAYGEN_KHR)
        .module(raygen_module)
        .name(entry_point_name)
        .build();

    let miss_shader_stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::MISS_KHR)
        .module(miss_module)
        .name(entry_point_name)
        .build();

    let closest_hit_stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
        .module(closest_hit_module)
        .name(entry_point_name)
        .build();
    
    let stage_infos = [raygen_shader_stage_info, miss_shader_stage_info, closest_hit_stage_info];

    let raygen_group_info = general_shader_group_info(0);
    let miss_group_info = general_shader_group_info(1);
    let hit_group_info = vk::RayTracingShaderGroupCreateInfoKHR::builder()
        .any_hit_shader(vk::SHADER_UNUSED_KHR)
        .closest_hit_shader(2)
        .intersection_shader(vk::SHADER_UNUSED_KHR)
        .general_shader(vk::SHADER_UNUSED_KHR)
        .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
        .build();

    let group_infos = [raygen_group_info, miss_group_info, hit_group_info];

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(std::slice::from_ref(&descriptor_set_layout));

    let pipeline_layout = unsafe { context.device().create_pipeline_layout(&pipeline_layout_info, None).unwrap() };

    let pipeline_info = vk::RayTracingPipelineCreateInfoKHR::builder()
        .stages(&stage_infos)
        .groups(&group_infos)
        .max_pipeline_ray_recursion_depth(2)
        .layout(pipeline_layout)
        .build();

    let pipeline = unsafe {
        context.extensions().ray_tracing_pipeline.create_ray_tracing_pipelines(
            vk::DeferredOperationKHR::null(),
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None
        ).unwrap()[0]
    };


    let mut sbt_builder = ShaderBindingTableBuilder::new(&context);

    sbt_builder.push_raygen_entry(0, &[]);
    sbt_builder.push_miss_entry(1, &[]);
    sbt_builder.push_hit_group_entry(2, &[]);

    let sbt = sbt_builder.build(pipeline, 3).unwrap();

    let readback_buffer = ReadBackBuffer::new(
        &context,
        std::mem::size_of::<f32>() as u64 * 4 * img_width as u64 * img_height as u64,
        vk::BufferUsageFlags::TRANSFER_DST
    ).unwrap();

    unsafe {
        context.execute_commands(|cmd_buffer| {

            let full_subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };

            { // transition image to GENERAL
                let image_barrier = vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                    .dst_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(sample_target.image.inner)
                    .subresource_range(full_subresource_range)
                    .build();

                let dependency_info = vk::DependencyInfo::builder()
                    .image_memory_barriers(std::slice::from_ref(&image_barrier));

                context.device().cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
            }

            { // trace rays
                context.device().cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline);
                context.device().cmd_bind_descriptor_sets(cmd_buffer, vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline_layout, 0, &[descriptor_set], &[]);
                context.extensions().ray_tracing_pipeline.cmd_trace_rays(
                    cmd_buffer,
                    &sbt.raygen_region(),
                    &sbt.miss_region(),
                    &sbt.hit_group_region(),
                    &vk::StridedDeviceAddressRegionKHR::default(),
                    img_width,
                    img_height,
                    1,
                );
            }
            
            { // transition image to TRANSFER_SRC
                let copy_barrier = vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(sample_target.image.inner)
                    .subresource_range(full_subresource_range)
                    .build();

                let dependency_info = vk::DependencyInfo::builder()
                    .image_memory_barriers(std::slice::from_ref(&copy_barrier));

                context.device().cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
            }

            { // copy image to buffer
                let copy_region = vk::BufferImageCopy2::builder()
                    .buffer_offset(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D {
                        x: 0,
                        y: 0,
                        z: 0,
                    })
                    .image_extent(vk::Extent3D {
                        width: img_width,
                        height: img_height,
                        depth: 1,
                    });
                
                let copy_info = vk::CopyImageToBufferInfo2::builder()
                    .src_image(sample_target.image.inner)
                    .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .dst_buffer(readback_buffer.handle())
                    .regions(std::slice::from_ref(&copy_region));

                context.device().cmd_copy_image_to_buffer2(cmd_buffer, &copy_info);
            }

            { // barrier to prevent reading from buffer until transfer has finished
                let buffer_barrier = vk::BufferMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                    .dst_access_mask(vk::AccessFlags2::HOST_READ)
                    .buffer(readback_buffer.handle())
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
                    .build();
                
                let dependency_info = vk::DependencyInfo::builder()
                    .dependency_flags(vk::DependencyFlags::empty())
                    .buffer_memory_barriers(std::slice::from_ref(&buffer_barrier));

                context.device().cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
            }

        }).unwrap();
        

        let mut img_raw_f32 = vec![0.0f32; (4 * img_width * img_height) as usize];

        readback_buffer.invalidate().unwrap();
        readback_buffer.read_slice(&mut img_raw_f32, 0);

        println!("{:?}", &img_raw_f32[0..4]);

        let img_raw_u8 = img_raw_f32.iter()
            .map(|f| (f * 255.0).clamp(0.0, 255.0) as u8)
            .collect_vec();

        let img = image::RgbaImage::from_vec(img_width, img_height, img_raw_u8).unwrap();

        img.save("output.png").unwrap(); 
    }
    
    unsafe {
        context.device().destroy_pipeline(pipeline, None);
        context.device().destroy_pipeline_layout(pipeline_layout, None);

        context.device().destroy_shader_module(raygen_module, None);
        context.device().destroy_shader_module(miss_module, None);
        context.device().destroy_shader_module(closest_hit_module, None);

        //context.device().free_descriptor_sets(descriptor_pool, &[descriptor_set]).unwrap();
        context.device().destroy_descriptor_set_layout(descriptor_set_layout, None);
        context.device().destroy_descriptor_pool(descriptor_pool, None);

        context.device().destroy_image_view(sample_view, None);

        context.extensions().acceleration_structure.destroy_acceleration_structure(accel_structure, None);
    };
}
