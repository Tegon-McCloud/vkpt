#![feature(iter_array_chunks)]
#![feature(int_roundings)]
#![feature(once_cell_try)]

use std::{ffi::CStr, fs, io::Write};

use ash::{prelude::VkResult, vk};
use gpu_allocator::MemoryLocation;
use itertools::Itertools;

use context::DeviceContext;
use nalgebra::{Matrix3, Point3, Vector3};
use pipeline::{Pipeline, Shader};
use resource::{Image, ImageView, ReadBackBuffer, UploadBuffer};
use scene::{material::{Material, MaterialType}, MaterialHandle};

use crate::{pipeline::ResourceLayout, scene::{camera::Camera, light::Environment, Scene}, shader_binding_table::ShaderBindingTableDescription, util::as_u8_slice};

pub mod util;
pub mod context;
pub mod resource;
pub mod pipeline;
pub mod shader_binding_table;
pub mod scene;

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

        unsafe {
            context.execute_commands(|cmd_buffer| {
                { // transition image to GENERAL
                    let image_barrier = vk::ImageMemoryBarrier2::builder()
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                        .dst_access_mask(vk::AccessFlags2::SHADER_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
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
                }
            })?;
        }
        
        Ok(Self {
            context,
            image,
        })
    }

    pub fn download(&self) -> VkResult<Vec<f32>> {
        
        let (width, height) = self.image.resolution();

        let read_back_buffer = ReadBackBuffer::new(
            &self.image.context(),
            std::mem::size_of::<f32>() as u64 * 4 * width as u64 * height as u64,
            vk::BufferUsageFlags::TRANSFER_DST
        ).unwrap();

        unsafe {
            self.context().execute_commands(|cmd_buffer| {
                { // transition image to TRANSFER_SRC
                    let copy_barrier = vk::ImageMemoryBarrier2::builder()
                        .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                        .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .image(self.image.inner())
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
    
                    self.context().device().cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
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
                            width,
                            height,
                            depth: 1,
                        });
                    
                    let copy_info = vk::CopyImageToBufferInfo2::builder()
                        .src_image(self.image.inner())
                        .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .dst_buffer(read_back_buffer.handle())
                        .regions(std::slice::from_ref(&copy_region));
    
                    self.context().device().cmd_copy_image_to_buffer2(cmd_buffer, &copy_info);
                }
    
                { // barrier to prevent reading from buffer until transfer has finished
                    let buffer_barrier = vk::BufferMemoryBarrier2::builder()
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                        .dst_access_mask(vk::AccessFlags2::HOST_READ)
                        .buffer(read_back_buffer.handle())
                        .offset(0)
                        .size(vk::WHOLE_SIZE)
                        .build();

                    // transfer image back to shad
                    let image_barrier = vk::ImageMemoryBarrier2::builder()
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                        .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                        .dst_access_mask(vk::AccessFlags2::SHADER_WRITE)
                        .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .image(self.image.inner())
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .build();


                    let dependency_info = vk::DependencyInfo::builder()
                        .dependency_flags(vk::DependencyFlags::empty())
                        .buffer_memory_barriers(std::slice::from_ref(&buffer_barrier))
                        .image_memory_barriers(std::slice::from_ref(&image_barrier));
    
                    self.context().device().cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
                }
            })?;
            
            let mut img_raw_f32 = vec![0.0f32; (4 * width * height) as usize];

            read_back_buffer.invalidate()?;
            read_back_buffer.read_slice(&mut img_raw_f32, 0);
            
            Ok(img_raw_f32)
        }
    }

    pub fn context(&self) -> &'ctx DeviceContext {
        self.image.context()
    }
}

// fn create_post_processing_shader<'ctx>(context: &'ctx DeviceContext) -> Shader<'ctx> {

    

// }

unsafe fn render<'ctx>(context: &'ctx DeviceContext, scene: &Scene, image: &Image<'ctx>, sample_count: u64, rand_seed: u32) -> VkResult<f32> {

    const SAMPLES_IN_FLIGHT: u64 = 2;

    let descriptor_set = scene.create_descriptor_set(ImageView::new(image, vk::Format::R32G32B32A32_SFLOAT, 0..1, 0..1)?)?;

    let mut shader_groups = Vec::new();
    let mut binding_table_desc = ShaderBindingTableDescription::new();

    scene.add_binding_table_entries(&mut shader_groups, &mut binding_table_desc);

    let descriptor_set_layouts = vec![descriptor_set.layout()];

    let push_constant_ranges = vec![
        vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::RAYGEN_KHR,
            offset: 0,
            size: 8,
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

    let start_time = std::time::Instant::now();

    for i in 0..sample_count {
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

        #[allow(dead_code)]
        struct PushConstantData {
            sample_index: u32,
            rand_seed: u32,
        }
        context.device().cmd_push_constants(
            cmd_buffer,
            pipeline.layout(),
            vk::ShaderStageFlags::RAYGEN_KHR,
            0,
            as_u8_slice(&PushConstantData { 
                sample_index: i as u32,
                rand_seed,
            }),
        );
        
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
        .values(std::slice::from_ref(&sample_count));

    context.device().wait_semaphores(&wait_info, u64::MAX)?;

    let end_time = std::time::Instant::now();
    
    let render_time_seconds = (end_time - start_time).as_secs_f32();

    println!("time: {}s", render_time_seconds);

    context.device().destroy_semaphore(semaphore, None);
    context.device().destroy_command_pool(cmd_pool, None);

    Ok(render_time_seconds)
}

fn load_environment_map<'ctx>(context: &'ctx DeviceContext) -> VkResult<Image<'ctx>> {

    let host_image = image::io::Reader::open("resources/kloppenheim_06_4k.hdr.png")
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

fn load_material_type<'ctx>(context: &'ctx DeviceContext, eval_shader_file: &str, sample_shader_file: &str) -> VkResult<MaterialType<'ctx>> {
    unsafe {
        let entry_point_name = CStr::from_bytes_with_nul_unchecked(b"main\0").to_owned();

        let evaluation_shader = Shader::new(
            context,
            eval_shader_file,
            entry_point_name.to_owned(),
        )?;
    
        let sample_shader = Shader::new(
            context,
            sample_shader_file,
            entry_point_name.to_owned(),
        )?;

        Ok(MaterialType {
            evaluation_shader,
            sample_shader,
        })
    }
}


fn run_refraction_test<'ctx>(context: &'ctx DeviceContext) -> VkResult<()> {

    let img_width = 512;
    let img_height = 512;
    let save_path = "images/refract";

    fs::create_dir_all(save_path).unwrap();

    let environment_map = load_environment_map(&context).unwrap();
    let camera = Camera::look_at(
        Point3::new(-1.0, 2.0, 4.0),
        Point3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        1.0,
        std::f32::consts::PI / 4.0,
    );
    let sample_target = SampleTarget::new(&context, img_width, img_height).unwrap();

    let mut scene = Scene::new(&context);

    scene.load("resources/mitsuba.gltf", context, None)?;

    scene.set_camera(camera);
    let environment_map_handle = scene.add_texture(environment_map);
    scene.set_environment(Environment::spherical(context, environment_map_handle)?);

    let lambertian_mat_type = scene.add_material_type(load_material_type(context, "shader_bin/lambertian_evaluate.rcall.spv", "shader_bin/lambertian_sample.rcall.spv")?);

    let backdrop_mat = scene.add_material(Material {
        ior: 1.54,
        roughness: 0.5,
        material_type: lambertian_mat_type,
    });

    let inside_mat = scene.add_material(Material {
        ior: 1.54,
        roughness: 0.8,
        material_type: lambertian_mat_type,
    });

    scene.set_instance_material(0, backdrop_mat);
    scene.set_instance_material(1, inside_mat);

    let material_types = [
        ("ss_uniform", "shader_bin/ss_evaluate.rcall.spv", "shader_bin/ss_sample_uniform.rcall.spv"),
        ("ss_ndf", "shader_bin/ss_evaluate.rcall.spv", "shader_bin/ss_sample_ndf.rcall.spv"),
        ("ss_vndf", "shader_bin/ss_evaluate.rcall.spv", "shader_bin/ss_sample_vndf.rcall.spv"),
        ("ms_heitz", "shader_bin/ss_evaluate.rcall.spv", "shader_bin/ms_sample_heitz.rcall.spv"),
        ("ms_dupuy", "shader_bin/ss_evaluate.rcall.spv", "shader_bin/ms_sample_dupuy.rcall.spv"),
    ];
    
    let roughnesses = [
        0.01, 0.04, 0.16, 0.64,
    ];

    let mut materials = vec![];

    for material_type in material_types {

        let material_type_handle = scene.add_material_type(load_material_type(context, material_type.1, material_type.2)?);

        for roughness in roughnesses {

            let material_handle = scene.add_material(Material {
                ior: 1.54,
                roughness,
                material_type: material_type_handle,
            });
            
            let file_name = format!("{}/{}_r{}.png", save_path, material_type.0, (100.0 * roughness).round() as u32);

            materials.push((file_name, material_handle));
        }
    }
    unsafe {    
        for material in materials {
    
            scene.set_instance_material(2, material.1);
    
            render(context, &scene, &sample_target.image, 1 << 9, 0)?;
            
            let img_data = sample_target.download()?;

            let img_data_u8 = img_data.iter()
                .map(|lin| lin.powf(1.0 / 2.2))
                .map(|f| (f * 255.0).clamp(0.0, 255.0) as u8)
                .collect_vec();

            let img = image::RgbaImage::from_vec(img_width, img_height, img_data_u8).unwrap();

            img.save(material.0).unwrap();
        }
    }
    Ok(())
}

fn run_ss_convergence_test<'ctx>(context: &'ctx DeviceContext) -> VkResult<()> {
    
    let img_width = 512;
    let img_height = 512;
    let save_path = "images/ss_convergence";

    fs::create_dir_all(save_path).unwrap();

    let environment_map = load_environment_map(&context).unwrap();
    let camera = Camera::look_at(
        Point3::new(-1.0, 2.0, 4.0),
        Point3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        1.0,
        std::f32::consts::PI / 4.0,
    );
    let sample_target = SampleTarget::new(&context, img_width, img_height).unwrap();

    let mut scene = Scene::new(&context);

    scene.load("resources/mitsuba.gltf", context, None)?;

    scene.set_camera(camera);
    let environment_map_handle = scene.add_texture(environment_map);
    scene.set_environment(Environment::spherical(context, environment_map_handle)?);

    let lambertian_mat_type = scene.add_material_type(load_material_type(context, "shader_bin/lambertian_evaluate.rcall.spv", "shader_bin/lambertian_sample.rcall.spv")?);

    let backdrop_mat = scene.add_material(Material {
        ior: 1.54,
        roughness: 0.5,
        material_type: lambertian_mat_type,
    });

    let inside_mat = scene.add_material(Material {
        ior: 1.54,
        roughness: 0.8,
        material_type: lambertian_mat_type,
    });

    scene.set_instance_material(0, backdrop_mat);
    scene.set_instance_material(1, inside_mat);

    let material_types = [
        // ("ss_uniform", scene.add_material_type(load_material_type(context, "shader_bin/ss_evaluate.", sample_shader_file)))
        ("ss_ndf",  scene.add_material_type(load_material_type(context, "shader_bin/ss_evaluate.rcall.spv", "shader_bin/ss_sample_ndf.rcall.spv")?)),
        ("ss_vndf", scene.add_material_type(load_material_type(context, "shader_bin/ss_evaluate.rcall.spv", "shader_bin/ss_sample_vndf.rcall.spv")?)),
    ];
    
    let roughnesses = [
        0.01, 0.04, 0.16, 0.64
    ];

    let ior = 1.54;
    
    for roughness in roughnesses {

        let reference_material = scene.add_material(Material {
            ior, 
            roughness,
            material_type: material_types[0].1,
        });

        scene.set_instance_material(2, reference_material);
        unsafe { render(context, &scene, &sample_target.image, 1 << 20, 12345678)? };

        let reference_rgba = sample_target.download()?;

        let reference_rgba_u8 = reference_rgba.iter()
            .map(|f| (f * 255.0).clamp(0.0, 255.0) as u8)
            .collect_vec();

        let reference = image::RgbaImage::from_vec(img_width, img_height, reference_rgba_u8).unwrap();

        reference.save(format!("{}/reference_r{}.png", save_path, (100.0 * roughness).round() as u32)).unwrap();

        for (material_type_name, material_type) in material_types {

            let material = scene.add_material(Material {
                ior,
                roughness,
                material_type,
            });

            scene.set_instance_material(2, material);

            let file_name = format!("{}/{}_r{}.txt", save_path, material_type_name, (100.0 * roughness).round() as u32);
            let mut file = std::fs::File::create(file_name).unwrap();

            for j in 0.. {

                let time = unsafe { render(context, &scene, &sample_target.image, 1 << j, 987654)? };
                let rgba = sample_target.download()?;
                let mse = mse_rgb(&rgba, &reference_rgba);

                write!(file, "{} {}\n", time, mse).unwrap();
            
                if time > 4.0 {
                    break;
                }
            }
        }
    }

    Ok(())
}

fn run_ss_equal_error_test<'ctx>(context: &'ctx DeviceContext) -> VkResult<()> {

    let img_width = 512;
    let img_height = 512;
    let save_path = "images/ss_equal_error";

    fs::create_dir_all(save_path).unwrap();

    let environment_map = load_environment_map(&context).unwrap();
    let camera = Camera::look_at(
        Point3::new(-1.0, 2.0, 4.0),
        Point3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        1.0,
        std::f32::consts::PI / 4.0,
    );
    let sample_target = SampleTarget::new(&context, img_width, img_height).unwrap();

    let mut scene = Scene::new(&context);

    scene.load("resources/mitsuba.gltf", context, None)?;

    scene.set_camera(camera);
    let environment_map_handle = scene.add_texture(environment_map);
    scene.set_environment(Environment::spherical(context, environment_map_handle)?);

    let lambertian_mat_type = scene.add_material_type(load_material_type(context, "shader_bin/lambertian_evaluate.rcall.spv", "shader_bin/lambertian_sample.rcall.spv")?);

    let backdrop_mat = scene.add_material(Material {
        ior: 1.54,
        roughness: 0.5,
        material_type: lambertian_mat_type,
    });

    let inside_mat = scene.add_material(Material {
        ior: 1.54,
        roughness: 0.8,
        material_type: lambertian_mat_type,
    });

    scene.set_instance_material(0, backdrop_mat);
    scene.set_instance_material(1, inside_mat);

    let material_types = [
        ("ss_ndf",  scene.add_material_type(load_material_type(context, "shader_bin/ss_evaluate.rcall.spv", "shader_bin/ss_sample_ndf.rcall.spv")?)),
        ("ss_vndf", scene.add_material_type(load_material_type(context, "shader_bin/ss_evaluate.rcall.spv", "shader_bin/ss_sample_vndf.rcall.spv")?)),
    ];
    
    let roughness = 0.16;
    let ior = 1.54;

    let ndf_material = scene.add_material(Material {
        ior,
        roughness,
        material_type: material_types[0].1,
    });

    let vndf_material = scene.add_material(Material {
        ior,
        roughness,
        material_type: material_types[1].1,
    });

    let reference_sample_count = 1 << 20;
    let target_mse = 0.0001;
    let tolerance = 10;

    unsafe {

        scene.set_instance_material(2, ndf_material);
        render(context, &scene, &sample_target.image, reference_sample_count, 0)?; 
        let reference_rgba = sample_target.download()?;

        let mut measure = |material: MaterialHandle, sample_count: u64| -> VkResult<(f32, f32, Vec<f32>)> {
            scene.set_instance_material(2, material);
            let time = render(context, &scene, &sample_target.image, sample_count, 0)?;
            let rgba = sample_target.download()?;
            let mse = mse_rgb(&rgba, &reference_rgba);

            Ok((time, mse, rgba))
        };

        let mut bisection = |material: MaterialHandle| -> VkResult<u64> {

            let mut sample_count_a = 1;
            let mut sample_count_b = 10000;

            loop {
                let sample_count = (sample_count_a + sample_count_b) / 2;
                
                if (sample_count_b - sample_count_a) / 2 < tolerance {
                    return Ok(sample_count);
                } 

                let (_, mse, _) = measure(material, sample_count)?;

                if mse > target_mse {   
                    sample_count_a = sample_count;
                } else {
                    sample_count_b = sample_count;
                }
            };
        };

        let ndf_sample_count = bisection(ndf_material)?;
        let vndf_sample_count = bisection(vndf_material)?;

        let (ndf_time, ndf_mse, ndf_rgba) = measure(ndf_material, ndf_sample_count)?;
        let (vndf_time, vndf_mse, vndf_rgba) = measure(vndf_material, vndf_sample_count)?;

        println!("mse: {},\tsamples: {},\ttime: {}", ndf_mse, ndf_sample_count, ndf_time);    
        println!("mse: {},\tsamples: {},\ttime: {}", vndf_mse, vndf_sample_count, vndf_time);    
        
        let save = |rgba: Vec<f32>, file: &str| {
            let rgba_u8 = rgba.iter()
            .copied()
            .array_chunks::<4>()
            .flat_map(post_process_orb)
            .collect_vec();
        
            image::RgbaImage::from_vec(img_width, img_height, rgba_u8).unwrap()
                .save(format!("{}/{}", save_path, file)).unwrap();
        };

        save(reference_rgba, "reference.png");
        save(ndf_rgba, "ndf.png");
        save(vndf_rgba, "vndf.png");
    }

    Ok(())
}

fn run_furnace_test<'ctx>(context: &'ctx DeviceContext) -> VkResult<()> {
    
    let img_width = 512;
    let img_height = 512;
    let save_path = "images/furnace";

    fs::create_dir_all(save_path).unwrap();

    let camera = Camera::new(
        Point3::new(0.0, 0.0, 2.0),
        Matrix3::new(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        )
    );
    let sample_target = SampleTarget::new(context, img_width, img_height).unwrap();

    let mut scene = Scene::new(context);

    scene.load("resources/sphere.gltf", context, None)?;
    
    scene.set_camera(camera);
    scene.set_environment(Environment::constant(context)?);

    let material_types = [
        ("ss_ndf", "shader_bin/ss_evaluate.rcall.spv", "shader_bin/ss_sample_ndf.rcall.spv"),
        ("ss_vndf", "shader_bin/ss_evaluate.rcall.spv", "shader_bin/ss_sample_vndf.rcall.spv"),
        ("ms_heitz", "shader_bin/ss_evaluate.rcall.spv", "shader_bin/ms_sample_heitz.rcall.spv"),
        ("ms_dupuy", "shader_bin/ss_evaluate.rcall.spv", "shader_bin/ms_sample_dupuy.rcall.spv"),
    ];
    
    let roughnesses = [
        0.01, 0.04, 0.16, 0.64,
    ];

    let ior = 1.54;

    let mut materials = vec![];

    for material_type in material_types {

        let material_type_handle = scene.add_material_type(load_material_type(context, material_type.1, material_type.2)?);

        for roughness in roughnesses {

            let material_handle = scene.add_material(Material {
                ior,
                roughness,
                material_type: material_type_handle,
            });
            
            let file_path = format!("{}/{}_r{}.png", save_path, material_type.0, (100.0 * roughness).round() as u32);

            materials.push((file_path, material_handle));
        }
    }
    unsafe {    
        for material in materials {

            scene.set_instance_material(0, material.1);
            
            render(context, &scene, &sample_target.image, 1 << 16, 0)?;
            
            let img_data = sample_target.download()?;

            let img_data_u8 = img_data.iter()
                .map(|lin| lin.powf(1.0 / 2.2))
                .map(|f| (f * 255.0).clamp(0.0, 255.0) as u8)
                .collect_vec();

            let img = image::RgbaImage::from_vec(img_width, img_height, img_data_u8).unwrap();

            img.save(material.0).unwrap();
        }
    }
    Ok(())
}

fn run_ms_ss_comparison_test<'ctx>(context: &'ctx DeviceContext) -> VkResult<()> {

    let img_width = 512;
    let img_height = 512;
    let sample_count = 1 << 16;
    let save_path = "images/ms_ss_comparison";
    
    fs::create_dir_all(save_path).unwrap();

    let environment_map = load_environment_map(&context).unwrap();
    let camera = Camera::look_at(
        Point3::new(-1.0, 2.0, 4.0),
        Point3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        1.0,
        std::f32::consts::PI / 4.0,
    );
    let sample_target = SampleTarget::new(&context, img_width, img_height).unwrap();

    let mut scene = Scene::new(&context);

    scene.load("resources/mitsuba.gltf", context, None)?;

    scene.set_camera(camera);
    let environment_map_handle = scene.add_texture(environment_map);
    scene.set_environment(Environment::spherical(context, environment_map_handle)?);

    let lambertian_mat_type = scene.add_material_type(load_material_type(context, "shader_bin/lambertian_evaluate.rcall.spv", "shader_bin/lambertian_sample.rcall.spv")?);

    let backdrop_mat = scene.add_material(Material {
        ior: 1.54,
        roughness: 0.5,
        material_type: lambertian_mat_type,
    });

    let inside_mat = scene.add_material(Material {
        ior: 1.54,
        roughness: 0.8,
        material_type: lambertian_mat_type,
    });

    scene.set_instance_material(0, backdrop_mat);
    scene.set_instance_material(1, inside_mat);

    let ss_material_type = scene.add_material_type(load_material_type(context, "shader_bin/ss_evaluate.rcall.spv", "shader_bin/ss_sample_vndf.rcall.spv")?);
    let ms_material_type = scene.add_material_type(load_material_type(context, "shader_bin/ss_evaluate.rcall.spv", "shader_bin/ms_sample_heitz.rcall.spv")?);
    
    let roughnesses = [
        0.01, 0.04, 0.16, 0.64
    ];
    
    let ior = 1.54;

    let timings_filename = format!("{}/timings.txt", save_path);
    let mut timings_file = std::fs::File::create(timings_filename).unwrap();

    let warmup_material = scene.add_material(Material {
        ior,
        roughness: 2.0,
        material_type: ss_material_type,
    });

    scene.set_instance_material(2, warmup_material);
    unsafe { render(context, &scene, &sample_target.image, sample_count, 0)? };

    for roughness in roughnesses {

        unsafe {

            let ss_material = scene.add_material(Material {
                ior,
                roughness,
                material_type: ss_material_type,
            });

            let ms_material = scene.add_material(Material {
                ior,
                roughness,
                material_type: ms_material_type,
            });

            scene.set_instance_material(2, ss_material);
            let ss_time = render(context, &scene, &sample_target.image, sample_count, 0)?;
            let ss_rgba = sample_target.download()?;
            
            scene.set_instance_material(2, ms_material);
            let ms_time = render(context, &scene, &sample_target.image, sample_count, 0)?;
            let ms_rgba = sample_target.download()?;

            let ss_rgba_u8 = ss_rgba.iter()
                .copied()
                .array_chunks::<4>()
                .flat_map(post_process_orb)
                .collect_vec();

            let ss_filename = format!("{}/ss_r{}.png", save_path, (100.0 * roughness).round() as u32);

            image::RgbaImage::from_vec(img_width, img_height, ss_rgba_u8).unwrap()
                .save(ss_filename).unwrap();

            let ms_rgba_u8 = ms_rgba.iter()
                .copied()
                .array_chunks::<4>()
                .flat_map(post_process_orb)
                .collect_vec();

            let ms_filename = format!("{}/ms_r{}.png", save_path, (100.0 * roughness).round() as u32);

            image::RgbaImage::from_vec(img_width, img_height, ms_rgba_u8).unwrap()
                .save(ms_filename).unwrap();

            let diff_rgba = ss_rgba.iter().copied()
                .zip(ms_rgba.iter().copied())
                .array_chunks::<4>()
                .flat_map(|[r, g, b, _]| [r.1 - r.0, g.1 - g.0, b.1 - b.0, 1.0])
                .collect_vec();

            let diff_rgba_u8 = diff_rgba.iter()
                .map(|c| (c.clamp(0.0, 1.0) * 255.0) as u8)
                .collect_vec();

            let diff_filename = format!("{}/diff_r{}.png", save_path, (100.0 * roughness).round() as u32);

            image::RgbaImage::from_vec(img_width, img_height, diff_rgba_u8).unwrap()
                .save(diff_filename).unwrap();

            
            write!(timings_file, "{} {}\n", ss_time, ms_time).unwrap();
        }
    }

    Ok(())
} 

fn run_ms_convergence_test<'ctx>(context: &'ctx DeviceContext) -> VkResult<()> {
    
    let img_width = 512;
    let img_height = 512;
    let save_path = "images/ms_convergence";

    fs::create_dir_all(save_path).unwrap();

    let environment_map = load_environment_map(&context).unwrap();
    let camera = Camera::look_at(
        Point3::new(-1.0, 2.0, 4.0),
        Point3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        1.0,
        std::f32::consts::PI / 4.0,
    );
    let sample_target = SampleTarget::new(&context, img_width, img_height).unwrap();

    let mut scene = Scene::new(&context);

    scene.load("resources/mitsuba.gltf", context, None)?;

    scene.set_camera(camera);
    let environment_map_handle = scene.add_texture(environment_map);
    scene.set_environment(Environment::spherical(context, environment_map_handle)?);

    let lambertian_mat_type = scene.add_material_type(load_material_type(context, "shader_bin/lambertian_evaluate.rcall.spv", "shader_bin/lambertian_sample.rcall.spv")?);

    let backdrop_mat = scene.add_material(Material {
        ior: 1.54,
        roughness: 0.5,
        material_type: lambertian_mat_type,
    });

    let inside_mat = scene.add_material(Material {
        ior: 1.54,
        roughness: 0.8,
        material_type: lambertian_mat_type,
    });

    scene.set_instance_material(0, backdrop_mat);
    scene.set_instance_material(1, inside_mat);

    let material_types = [
        ("ss",  scene.add_material_type(load_material_type(context, "shader_bin/ss_evaluate.rcall.spv", "shader_bin/ss_sample_vndf.rcall.spv")?)),
        ("ms", scene.add_material_type(load_material_type(context, "shader_bin/ss_evaluate.rcall.spv", "shader_bin/ms_sample_heitz.rcall.spv")?)),
    ];
    
    let roughnesses = [
        0.01, 0.04, 0.16, 0.64
    ];

    let ior = 1.54;
    
    for roughness in roughnesses {

        for (material_type_name, material_type) in material_types {

            let material = scene.add_material(Material {
                ior,
                roughness,
                material_type,
            });

            scene.set_instance_material(2, material);
            unsafe { render(context, &scene, &sample_target.image, 1 << 20, 12345678)? };

            let reference_rgba = sample_target.download()?;

            let reference_rgba_u8 = reference_rgba.iter()
                .map(|f| (f * 255.0).clamp(0.0, 255.0) as u8)
                .collect_vec();
    
            let reference = image::RgbaImage::from_vec(img_width, img_height, reference_rgba_u8).unwrap();
    
            reference.save(format!("{}/reference_{}_r{}.png", save_path, material_type_name, (100.0 * roughness).round() as u32)).unwrap();

            let file_name = format!("{}/{}_r{}.txt", save_path, material_type_name, (100.0 * roughness).round() as u32);
            let mut file = std::fs::File::create(file_name).unwrap();

            for j in 0.. {

                let time = unsafe { render(context, &scene, &sample_target.image, 1 << j, 98765)? };
                let rgba = sample_target.download()?;
                let mse = mse_rgb(&rgba, &reference_rgba);

                write!(file, "{} {}\n", time, mse).unwrap();
            
                if time > 4.0 {
                    break;
                }
            }
        }
    }

    Ok(())
}

pub fn run_sphere_test<'ctx>(context: &'ctx DeviceContext) -> VkResult<()> {
    let img_width = 512;
    let img_height = 512;
    
    let sample_target = SampleTarget::new(&context, img_width, img_height).unwrap();

    let mut scene = Scene::new(&context);

    let camera = Camera::new(
        Point3::new(0.0, 0.0, 2.0),
        Matrix3::new(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        )
    );
    scene.set_camera(camera);
    
    let environment_map = load_environment_map(&context)?;
    let environment_map_handle = scene.add_texture(environment_map);
    
    scene.set_environment(Environment::spherical(&context, environment_map_handle)?);
    // scene.set_environment(Environment::constant(&context).unwrap());

    let material_type = scene.add_material_type(
        load_material_type(context, "shader_bin/ss_evaluate.rcall.spv", "shader_bin/ss_sample_uniform.rcall.spv")?
    );
    
    let material = scene.add_material(Material {
        ior: 1.54,
        roughness: 0.1,
        material_type,
    });

    scene.load("resources/sphere.gltf", &context, None)?;
    
    scene.set_instance_material(0, material);

    unsafe { render(&context, &scene, &sample_target.image, 1 << 12, 0)? };
    let rgba = sample_target.download()?;
    
    let rgba_u8 = rgba.iter()
        .copied()
        .map(to_u8)
        .collect_vec();

    image::RgbaImage::from_vec(img_width, img_height, rgba_u8).unwrap()
        .save("images/sphere.png").unwrap();

    Ok(())
}


pub fn run_orb_test<'ctx>(context: &'ctx DeviceContext) -> VkResult<()> {

    let img_width = 512;
    let img_height = 512;
    
    let sample_target = SampleTarget::new(&context, img_width, img_height).unwrap();

    let mut scene = Scene::new(&context);

    let camera = Camera::look_at(
        Point3::new(-1.0, 2.0, 4.0),
        Point3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        1.0,
        std::f32::consts::PI / 4.0,
    );
    scene.set_camera(camera);
    
    let environment_map = load_environment_map(&context)?;
    let environment_map_handle = scene.add_texture(environment_map);
    
    scene.set_environment(Environment::spherical(&context, environment_map_handle)?);
    // scene.set_environment(Environment::constant(&context).unwrap());

    let lambertian_mat_type = scene.add_material_type(
        load_material_type(context, "shader_bin/lambertian_evaluate.rcall.spv", "shader_bin/lambertian_sample.rcall.spv")?
    );

    let microfacet_mat_type = scene.add_material_type(
        load_material_type(context, "shader_bin/ss_evaluate.rcall.spv", "shader_bin/ss_sample_uniform.rcall.spv")?
    );

    let backdrop_mat = scene.add_material(Material {
        ior: 1.54,
        roughness: 0.5,
        material_type: lambertian_mat_type,
    });

    let inside_mat = scene.add_material(Material {
        ior: 1.54,
        roughness: 0.8,
        material_type: lambertian_mat_type,
    });
    
    let shell_mat = scene.add_material(Material {
        ior: 1.54,
        roughness: 0.1,
        material_type: microfacet_mat_type,
    });

    scene.load("resources/mitsuba.gltf", &context, None)?;

    scene.set_instance_material(0, backdrop_mat);
    scene.set_instance_material(1, inside_mat);
    scene.set_instance_material(2, shell_mat);

    unsafe { render(&context, &scene, &sample_target.image, 1 << 12, 0)? };
    let rgba = sample_target.download()?;
    
    let rgba_u8 = rgba.iter()
        .copied()
        .map(to_u8)
        .collect_vec();

    image::RgbaImage::from_vec(img_width, img_height, rgba_u8).unwrap()
        .save("images/orb.png").unwrap();

    Ok(())
}

fn mse_rgb(image_rgba: &[f32], reference_rgba: &[f32]) -> f32 {

    let image_rgb = image_rgba.iter()
        .copied()
        .array_chunks::<4>()
        .flat_map(rgba_to_rgb);

    let reference_rgb = reference_rgba.iter()
        .copied()
        .array_chunks::<4>()
        .flat_map(rgba_to_rgb);

    let sse = image_rgb.zip(reference_rgb)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>();

    let count = 3 * image_rgba.len() / 4;

    sse / count as f32
} 

fn rgba_to_rgb([r, g, b, _]: [f32; 4]) -> [f32; 3] {
    [r, g, b]
}

fn to_u8(r: f32) -> u8 {
    (r.clamp(0.0, 1.0) * 255.0) as u8
}

fn post_process_orb([r, g, b, a]: [f32; 4]) -> [u8; 4] {

    // gamma correction
    let [r, g, b] = [r, g, b].map(|c| c.powf(1.0 / 2.2));

    [r, g, b, a].map(to_u8)
}

fn main() {

    let context = DeviceContext::new().expect("failed to create device context");
    
    run_refraction_test(&context).unwrap();
    run_ss_convergence_test(&context).unwrap();
    run_ss_equal_error_test(&context).unwrap();

    run_furnace_test(&context).unwrap();
    run_ms_ss_comparison_test(&context).unwrap();
    run_ms_convergence_test(&context).unwrap();
    
    // run_sphere_test(&context).unwrap();
    // run_orb_test(&context).unwrap();
}
