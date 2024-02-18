#![feature(iter_array_chunks)]
#![feature(int_roundings)]
#![feature(once_cell_try)]

use ash::{vk, prelude::VkResult};
use itertools::Itertools;

use context::DeviceContext;
use nalgebra::Vector3;
use resource::{Image, ReadBackBuffer};
use scene::{Material, SceneDescription};

pub mod util;
pub mod context;
pub mod resource;
pub mod pipeline;
pub mod shader_binding_table;
pub mod scene;
pub mod output;


pub fn create_descriptor_set<'a>(
    context: &'a DeviceContext,
    set_layout: vk::DescriptorSetLayout,
    accel_structure: vk::AccelerationStructureKHR,
    output_view: vk::ImageView,
) -> VkResult<(vk::DescriptorSet, vk::DescriptorPool)> {
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

        Ok((set, pool))
    }
}

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

fn main() {

    let context = DeviceContext::new().expect("failed to create device context");

    let mut scene_desc = SceneDescription::new();
    scene_desc.add_material(Material { base_color: Vector3::new(1.0, 1.0, 0.0) });
    scene_desc.load("./resources/bunny.gltf", &context).unwrap();
    
    let img_width = 512;
    let img_height = 512;

    let sample_target = SampleTarget::new(&context, img_width, img_height).unwrap();
    let sample_view = unsafe { context.device().create_image_view(&sample_target.full_view_info(), None).unwrap() };
    
    let descriptor_set_layout_bindings = [
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

    let descriptor_set_layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(&descriptor_set_layout_bindings);

    let descriptor_set_layout = unsafe {
        context.device().create_descriptor_set_layout(&descriptor_set_layout_info, None).unwrap()
    };

    let scene = scene_desc.build(&context).unwrap();
    
    let (sbt, pipeline) = unsafe { scene.make_sbt(descriptor_set_layout).unwrap() };

    let (descriptor_set, descriptor_pool) = create_descriptor_set(&context, descriptor_set_layout, scene.tlas(), sample_view).unwrap();
    
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
                context.device().cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline.pipeline());
                context.device().cmd_bind_descriptor_sets(cmd_buffer, vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline.layout(), 0, &[descriptor_set], &[]);
                context.extensions().ray_tracing_pipeline.cmd_trace_rays(
                    cmd_buffer,
                    &sbt.raygen_region(),
                    &sbt.miss_region(),
                    &sbt.hit_group_region(),
                    &sbt.callable_region(),
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

        let px_idx = (img_height / 2 * img_width + img_width / 2) as usize;

        println!("{:?}", &img_raw_f32[4*px_idx..4*px_idx+4]);

        let img_raw_u8 = img_raw_f32.iter()
            .map(|f| (f * 255.0).clamp(0.0, 255.0) as u8)
            .collect_vec();

        let img = image::RgbaImage::from_vec(img_width, img_height, img_raw_u8).unwrap();

        img.save("output.png").unwrap(); 
    }
    
    unsafe {
        //context.device().free_descriptor_sets(descriptor_pool, &[descriptor_set]).unwrap();
        context.device().destroy_descriptor_set_layout(descriptor_set_layout, None);
        context.device().destroy_descriptor_pool(descriptor_pool, None);

        context.device().destroy_image_view(sample_view, None);
    };
}
