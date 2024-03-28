#![feature(iter_array_chunks)]
#![feature(int_roundings)]
#![feature(once_cell_try)]

use ash::{prelude::VkResult, vk};
use gpu_allocator::MemoryLocation;
use itertools::Itertools;

use context::DeviceContext;
use nalgebra::{Matrix3, Point3};
use pipeline::Pipeline;
use resource::{Image, ImageView, ReadBackBuffer, UploadBuffer};
use scene::SceneDescriptorSet;

use crate::{pipeline::{ResourceLayout, ShaderData}, scene::{camera::Camera, light::Environment, Scene}, shader_binding_table::ShaderBindingTableDescription, util::as_u8_slice};

pub mod util;
pub mod context;
pub mod resource;
pub mod descriptor;
pub mod pipeline;
pub mod shader_binding_table;
pub mod scene;
pub mod output;

struct SampleTarget<'ctx> {
    #[allow(unused)]
    context: &'ctx DeviceContext,
    image: Image<'ctx>,
}

impl<'ctx> SampleTarget<'ctx> {

    pub fn new(context: &'ctx DeviceContext, width: u32, height: u32) -> VkResult<Self> {
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
}

// fn create_post_processing_shader<'ctx>(context: &'ctx DeviceContext) -> Shader<'ctx> {

    

// }

unsafe fn render<'ctx>(context: &'ctx DeviceContext, scene: &Scene, image: &Image<'ctx>) -> VkResult<()> {

    const SAMPLES_IN_FLIGHT: u64 = 2;
    const SAMPLE_COUNT: u64 = 512;

    let descriptor_set = scene.create_descriptor_set(ImageView::new(image, vk::Format::R32G32B32A32_SFLOAT, 0..1, 0..1)?)?;

    let mut shader_groups = Vec::new();
    let mut binding_table_desc = ShaderBindingTableDescription::new();

    scene.add_binding_table_entries(&mut shader_groups, &mut binding_table_desc);

    let descriptor_set_layouts = vec![descriptor_set.layout()];

    let push_constant_ranges = vec![
        vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::RAYGEN_KHR,
            offset: 0,
            size: 68,
        },
    ];

    let resource_layout = ResourceLayout::new(descriptor_set_layouts, push_constant_ranges);
    let pipeline = Pipeline::new(context, &resource_layout, &shader_groups, binding_table_desc)?;
    let binding_table = pipeline.binding_table();

    let cmd_pool_info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(context.queue_family());
    
    let cmd_pool = context.device().create_command_pool(&cmd_pool_info, None)?;

    let cmd_buffer_alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(SAMPLES_IN_FLIGHT as u32);

    let cmd_buffers = context.device().allocate_command_buffers(&cmd_buffer_alloc_info)?;

    let mut wait_points = [0; SAMPLES_IN_FLIGHT as usize];
    let semaphore = context.create_timeline_semaphore()?;

    for i in 0..SAMPLE_COUNT {
        let signal_point = i + 1;
        
        let j = (i % SAMPLES_IN_FLIGHT) as usize;
        let cmd_buffer = cmd_buffers[j];
        let wait_point = wait_points[j];

        // host side wait for the command buffer to be finished executing
        let wait_info = vk::SemaphoreWaitInfo::builder()
            .semaphores(std::slice::from_ref(&semaphore))
            .values(std::slice::from_ref(&wait_point));

        context.device().wait_semaphores(&wait_info, u64::MAX)?;
        context.device().reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())?;

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        context.device().begin_command_buffer(cmd_buffer, &begin_info)?;

        // wait for last sample to written before reading the result of it
        let image_barrier = vk::ImageMemoryBarrier2::builder()
            .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(image.inner())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build();

        let dependency_info = vk::DependencyInfo::builder()
            .image_memory_barriers(std::slice::from_ref(&image_barrier));
            
        context.device().cmd_pipeline_barrier2(cmd_buffer, &dependency_info);

        context.device().cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline.pipeline());

        context.device().cmd_bind_descriptor_sets(cmd_buffer, vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline.layout(), 0, &[descriptor_set.inner()], &[]);
        
        context.device().cmd_push_constants(cmd_buffer, pipeline.layout(), vk::ShaderStageFlags::RAYGEN_KHR, 0, scene.camera_data().as_u8_slice());
        context.device().cmd_push_constants(cmd_buffer, pipeline.layout(), vk::ShaderStageFlags::RAYGEN_KHR, 64, as_u8_slice(&(i as u32)));

        context.extensions().ray_tracing_pipeline.cmd_trace_rays(
            cmd_buffer,
            &binding_table.raygen_region(),
            &binding_table.miss_region(),
            &binding_table.hit_group_region(),
            &binding_table.callable_region(),
            image.resolution().0,
            image.resolution().1,
            1,
        );                    

        context.device().end_command_buffer(cmd_buffer)?; 

        let cmd_buffer_info = vk::CommandBufferSubmitInfo::builder()
            .command_buffer(cmd_buffer)
            .build();

        let signal_info = vk::SemaphoreSubmitInfo::builder()
            .semaphore(semaphore)
            .value(signal_point)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS_KHR)
            .build();

        let submit_info = vk::SubmitInfo2::builder()
            .command_buffer_infos(std::slice::from_ref(&cmd_buffer_info))
            .signal_semaphore_infos(std::slice::from_ref(&signal_info))
            .build();

        context.device().queue_submit2(context.queue(), &[submit_info], vk::Fence::null())?;

        wait_points[j] = signal_point;
    }
    
    let wait_info = vk::SemaphoreWaitInfo::builder()
        .semaphores(std::slice::from_ref(&semaphore))
        .values(std::slice::from_ref(&SAMPLE_COUNT));

    context.device().wait_semaphores(&wait_info, u64::MAX)?;

    context.device().destroy_semaphore(semaphore, None);
    context.device().destroy_command_pool(cmd_pool, None);



    Ok(())
}

fn load_environment_map<'ctx>(context: &'ctx DeviceContext) -> VkResult<Image<'ctx>> {

    let host_image = image::io::Reader::open("resources/luxo_pxr_campus.hdr.png")
        .expect("failed to read image")
        .decode()
        .expect("failed to decode image");

    let queue_family = context.queue_family();

    let image_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UINT)
        .extent(vk::Extent3D { width: host_image.width(), height: host_image.height(), depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .queue_family_indices(std::slice::from_ref(&queue_family))
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = Image::new(context, &image_info, MemoryLocation::GpuOnly)?;

    let mut buffer = UploadBuffer::new(
        context,
        (host_image.width() * host_image.height() * 4) as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
    )?;

    let pixels = host_image.as_rgba8().unwrap().as_raw();

    unsafe {
        buffer.write_u8_slice(pixels, 0);
        buffer.flush()?;

        context.execute_commands(|cmd_buffer| {


            // transition image to transfer destination
            let image_barrier = vk::ImageMemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(image.inner())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build();

            let dependency_info = vk::DependencyInfo::builder()
                .image_memory_barriers(std::slice::from_ref(&image_barrier));

            context.device().cmd_pipeline_barrier2(cmd_buffer, &dependency_info);

            let copy_region = vk::BufferImageCopy2::builder()
                .buffer_offset(0)
                .buffer_row_length(host_image.width())
                .buffer_image_height(host_image.height())
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D { width: host_image.width(), height: host_image.height(), depth: 1 })
                .build();
            
            let copy_info = vk::CopyBufferToImageInfo2::builder()
                .src_buffer(buffer.handle())
                .dst_image(image.inner())
                .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .regions(std::slice::from_ref(&copy_region));

            context.device().cmd_copy_buffer_to_image2(cmd_buffer, &copy_info);

            let image_barrier = vk::ImageMemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(image.inner())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build();

            let dependency_info = vk::DependencyInfo::builder()
                .image_memory_barriers(std::slice::from_ref(&image_barrier));

            context.device().cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
        })?;
    }

    Ok(image)
}

// fn as_f32_row_major<const M: usize, const N: usize>(value: &serde_json::Value) -> Option<SMatrix<f32, M, N>> {
//     let array = value.as_array()?;
//     let mut matrix = SMatrix::zeros();

//     for i in 0..M {
//         for j in 0..N {
//             matrix[(i, j)] = array.get(i * N + j)?.as_f64()? as f32;
//         }
//     }

//     Some(matrix)
// }

// fn as_u32_row_major<const M: usize, const N: usize>(value: &serde_json::Value) -> Option<SMatrix<u32, M, N>> {
//     let array = value.as_array()?;
//     let mut matrix = SMatrix::zeros();

//     for i in 0..(M * N) {
//         matrix[i] = array.get(i)?.as_u64()? as u32;
//     }

//     Some(matrix)
// }

// fn get_camera(reference: &serde_json::Value) -> Option<Camera> {

//     let position = Point3::from(as_f32_row_major(reference.get("cam_pos")?)?);

//     let intrinsic = as_f32_row_major::<3, 3>(reference.get("K")?)?;
//     let rotation = as_f32_row_major::<3, 3>(reference.get("R")?)?;
//     let resolution = as_u32_row_major::<2, 1>(reference.get("im_resolution")?)?;
//     let scale = Matrix3::new(
//         resolution.x as f32, 0.0, 0.0,
//         0.0, resolution.y as f32, 0.0,
//         0.0, 0.0, 1.0,
//     );

//     println!("{}", (intrinsic * rotation).try_inverse()? * scale);

//     Some(Camera::new(position, (intrinsic * rotation).try_inverse()? * scale))
// }

// fn get_scene_data() -> Option<Camera> {

//     let scene_estimate: serde_json::Value = serde_json::from_reader(File::open("./resources/scene_estimate.json").unwrap()).unwrap();

//     let reference = scene_estimate.get("photos")?.get("reference_0_0.png")?;

//     get_camera(reference)
// }

fn main() {

    unsafe {
        let context = DeviceContext::new().expect("failed to create device context");

        let img_width = 512;
        let img_height = 512;
    
        let sample_target = SampleTarget::new(&context, img_width, img_height).unwrap();

        let mut scene = Scene::new(&context);
        scene.load("./resources/sphere.gltf", &context).unwrap();

        let camera = Camera::new(
            Point3::new(0.0, 0.0, 5.0),
            Matrix3::new(
                1.0, 0.0, -0.5,
                0.0, -1.0, 0.5,
                0.0, 0.0, -1.0,
            )
        );

        scene.set_camera(camera);
        // scene.set_environment(Environment::constant(&context).unwrap());

        let environment_map = load_environment_map(&context).unwrap();

        let environment_map_handle = scene.add_texture(environment_map);
        scene.set_environment(Environment::spherical(&context, environment_map_handle).unwrap());
        
        let readback_buffer = ReadBackBuffer::new(
            &context,
            std::mem::size_of::<f32>() as u64 * 4 * img_width as u64 * img_height as u64,
            vk::BufferUsageFlags::TRANSFER_DST
        ).unwrap();
        
        context.execute_commands(|cmd_buffer| {
            { // transition image to GENERAL
                let image_barrier = vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                    .dst_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(sample_target.image.inner())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .build();

                let dependency_info = vk::DependencyInfo::builder()
                    .image_memory_barriers(std::slice::from_ref(&image_barrier));

                context.device().cmd_pipeline_barrier2(cmd_buffer, &dependency_info);

                context.device().cmd_clear_color_image(
                    cmd_buffer,
                    sample_target.image.inner(),
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                    &[vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }],
                );

            }
        }).unwrap();

        render(&context, &scene, &sample_target.image).unwrap();
        
        context.execute_commands(|cmd_buffer| {
            { // transition image to TRANSFER_SRC
                let copy_barrier = vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(sample_target.image.inner())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
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
                    .src_image(sample_target.image.inner())
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
        println!("{:?}", &img_raw_f32[img_raw_f32.len()-4..img_raw_f32.len()]);


        // let px_idx = (img_height / 2 * img_width + img_width / 2) as usize;

        // println!("{:?}", &img_raw_f32[4*px_idx..4*px_idx+4]);

        let img_raw_u8 = img_raw_f32.iter()
            .map(|f| (f * 255.0).clamp(0.0, 255.0) as u8)
            .collect_vec();

        let img = image::RgbaImage::from_vec(img_width, img_height, img_raw_u8).unwrap();

        img.save("output.png").unwrap();
    }
}
