use std::{ffi::CStr, ops::Range, path::Path};

use ash::{vk, prelude::VkResult};
use nalgebra::{Matrix3x4, Matrix4};

use crate::{
    context::DeviceContext,
    pipeline::{Shader, ShaderData, ShaderGroup},
    resource::{DeviceBuffer, Image, ImageView, UploadBuffer},
    shader_binding_table::ShaderBindingTableDescription,
    util::{self, as_u8_slice},
};

use self::{camera::Camera, light::{LightSource, Environment}, material::{Material, MaterialType}, mesh::Mesh};

pub mod mesh;
pub mod material;
pub mod light;
pub mod camera;

#[derive(Debug, Clone, Copy)]
pub struct MeshHandle {
    index: usize,
}


#[derive(Debug, Clone, Copy)]
pub struct MaterialTypeHandle {
    index: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct TextureHandle {
    index: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct MaterialHandle {
    index: usize,
}

struct GeometryInstance {
    mesh: MeshHandle,
    material: MaterialHandle,
    transform: Matrix3x4<f32>,
}

#[repr(C)]
struct HitGroupSbtData {
    material_index: u32,
    face_address: vk::DeviceAddress,
    position_address: vk::DeviceAddress,
    normal_address: vk::DeviceAddress,
}

#[repr(C)]
struct MicrofacetData {
    ior: f32,
    roughness: f32,
}

pub struct Scene<'ctx> {
    context: &'ctx DeviceContext,
    meshes: Vec<Mesh<'ctx>>,
    material_types: Vec<MaterialType<'ctx>>,
    textures: Vec<Image<'ctx>>,
    materials: Vec<Material>,
    instances: Vec<GeometryInstance>,
    camera: Camera,
    environment: Option<Environment<'ctx>>,
    raygen_shader: Shader<'ctx>, // abstract this into camera
    // miss_shader: Shader<'ctx>, // abstract this into light
    closest_hit_shader: Shader<'ctx>,
}

impl<'ctx> Scene<'ctx> {

    pub fn new(context: &'ctx DeviceContext) -> Self {
        
        let raygen_shader = unsafe {
            Shader::new(
                context,
                "shader_bin/raytrace.rgen.spv",
                CStr::from_bytes_with_nul_unchecked(b"main\0").to_owned(),
            ).unwrap()
        };

        let closest_hit_shader = unsafe {
            Shader::new(
                context,
                "shader_bin/raytrace.rchit.spv",
                CStr::from_bytes_with_nul_unchecked(b"main\0").to_owned(),
            ).unwrap()
        };

        Self {
            context,
            meshes: Vec::new(),
            material_types: Vec::new(),
            textures: Vec::new(),
            materials: Vec::new(),
            instances: Vec::new(),
            camera: Camera::default(),
            environment: None,
            raygen_shader,
            closest_hit_shader,
        }
    }

    pub fn add_mesh(&mut self, mesh: Mesh<'ctx>) -> MeshHandle {
        self.meshes.push(mesh);
        MeshHandle { index: self.meshes.len() - 1 }
    }
    
    pub fn add_material_type(&mut self, material_type: MaterialType<'ctx>) -> MaterialTypeHandle {
        self.material_types.push(material_type);
        MaterialTypeHandle { index: self.material_types.len() - 1 }
    }

    pub fn add_texture(&mut self, texture: Image<'ctx>) ->  TextureHandle {
        self.textures.push(texture);
        TextureHandle { index: self.textures.len() - 1 }
    }

    pub fn add_material(&mut self, material: Material) -> MaterialHandle {
        self.materials.push(material);
        MaterialHandle { index: self.materials.len() - 1 }
    }

    pub fn add_instance(&mut self, mesh: MeshHandle, material: MaterialHandle, transform: Matrix4<f32>) {
        let transform = Matrix3x4::from_fn(|i, j| transform[(i, j)]);
        self.instances.push(GeometryInstance{ mesh, material, transform });
    }

    pub fn set_camera(&mut self, camera: Camera) {
        self.camera = camera;
    }

    pub fn set_environment(&mut self, environment: Environment<'ctx>) {
        self.environment = Some(environment);
    }

    pub fn get_mesh(&self, mesh_id: usize) -> &Mesh<'ctx> {
        &self.meshes[mesh_id]
    }

    pub fn camera_data(&self) -> impl ShaderData {
        self.camera.serialize()
    }

    pub unsafe fn create_descriptor_set<'a>(&'a self, output_view: ImageView<'ctx>) -> VkResult<SceneDescriptorSet<'ctx, 'a>> {

        let layout = self.create_descriptor_set_layout()?;

        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .mip_lod_bias(0.0)
            .anisotropy_enable(false)
            .compare_enable(false)
            .min_lod(0.0)
            .max_lod(vk::LOD_CLAMP_NONE)
            .unnormalized_coordinates(false);

        let sampler = self.context.device().create_sampler(&sampler_info, None)?;

        let texture_views = self.textures.iter()
            .map(|texture| ImageView::new(texture, vk::Format::R8G8B8A8_UINT, 0..1, 0..1))
            .collect::<VkResult<Vec<_>>>()?;

        let (tlas, tlas_buffer) = self.build_tlas()?;


        let pool_sizes = [
            vk::DescriptorPoolSize::builder()
                .descriptor_count(1)
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .build(),

            vk::DescriptorPoolSize::builder()
                .descriptor_count(1)
                .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .build(),

            vk::DescriptorPoolSize::builder()
                .descriptor_count(1)
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .build()
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        
        let pool = self.context.device().create_descriptor_pool(&pool_info, None)?;

        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(pool)
            .set_layouts(std::slice::from_ref(&layout));
  
        let set = self.context.device().allocate_descriptor_sets(&alloc_info)?[0];

        let mut accel_structure_write = vk::WriteDescriptorSetAccelerationStructureKHR::builder()
            .acceleration_structures(std::slice::from_ref(&tlas))
            .build();

        let mut accel_write = vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .push_next(&mut accel_structure_write)
            .build();

        accel_write.descriptor_count = 1;

        let output_image_info = vk::DescriptorImageInfo::builder()
            .sampler(vk::Sampler::null())
            .image_view(output_view.inner())
            .image_layout(vk::ImageLayout::GENERAL)
            .build();

        let output_image_write = vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&output_image_info))
            .build();

        let texture_image_info = vk::DescriptorImageInfo::builder()
            .sampler(sampler)
            .image_view(texture_views[0].inner())
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build();
            
        let texture_image_write = vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&texture_image_info))
            .build();

        self.context.device().update_descriptor_sets(
            &[
                accel_write,
                output_image_write,
                texture_image_write,
            ],
            &[],
        );

        Ok(SceneDescriptorSet {
            scene: self, // ensures that the compiled scene cannot outlive scene resources

            output_view,

            tlas_buffer, // ensures that tlas_buffer will live as long as tlas
            tlas,

            texture_views,

            layout,

            descriptor_pool: pool,
            descriptor_set: set,
        })
    }

    unsafe fn create_descriptor_set_layout(&self) -> VkResult<vk::DescriptorSetLayout> {
        
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

            vk::DescriptorSetLayoutBinding::builder()
                .binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build(),
        ];
    
        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&layout_bindings);
        
        let layout = self.context.device().create_descriptor_set_layout(&layout_info, None).unwrap();
        
        Ok(layout)
    }

    unsafe fn build_tlas(&self) -> VkResult<(vk::AccelerationStructureKHR, DeviceBuffer<'ctx>)> {
        let blas_instance_iter = self.instances.iter()
            .enumerate()
            .map(|(i, instance)|
                vk::AccelerationStructureInstanceKHR {
                    transform: util::matrix_to_vk_transform(instance.transform),
                    instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
                    instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                        i as u32,
                        vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8
                    ),
                    acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                        device_handle: self.meshes[instance.mesh.index].get_accel_structure_device_address()
                    },
                }
            );
            
        let mut instance_buffer = UploadBuffer::new(
            &self.context,
            (self.instances.len() * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>()) as u64,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
        )?;

        instance_buffer.write_from_iter(blas_instance_iter, 0);
        instance_buffer.flush()?;

        let instance_data = vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR::builder()
                .array_of_pointers(false)
                .data(instance_buffer.get_device_or_host_address_const())
                .build(),
        };

        let tlas_geometry = vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(instance_data);

        let mut tlas_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&tlas_geometry))
            .build();

        let tlas_sizes = self.context.extensions().acceleration_structure.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &tlas_build_info,
            &[self.instances.len() as u32],
        );

        println!("Tlas buffer size: {}", tlas_sizes.acceleration_structure_size);

        let tlas_buffer = DeviceBuffer::new(
            self.context,
            tlas_sizes.acceleration_structure_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        )?;

        let tlas_info = vk::AccelerationStructureCreateInfoKHR::builder()
            .buffer(tlas_buffer.handle())
            .size(tlas_sizes.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);

        let tlas = self.context.extensions().acceleration_structure.create_acceleration_structure(&tlas_info, None).unwrap();

        let scratch_buffer = DeviceBuffer::new(
            self.context,
            tlas_sizes.build_scratch_size, 
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;

        tlas_build_info.scratch_data = scratch_buffer.get_device_or_host_address();
        tlas_build_info.dst_acceleration_structure = tlas;

        let build_range = vk::AccelerationStructureBuildRangeInfoKHR::builder()
            .primitive_count(self.instances.len() as u32)
            .build();

        let build_ranges = [std::slice::from_ref(&build_range)];

        self.context.execute_commands(|cmd_buffer| {
            self.context.extensions().acceleration_structure.cmd_build_acceleration_structures(
                cmd_buffer,
                std::slice::from_ref(&tlas_build_info),
                &build_ranges,
            );

            let barrier = vk::BufferMemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR)
                .buffer(tlas_buffer.handle())
                .offset(0)
                .size(vk::WHOLE_SIZE)
                .build();

            let dependency_info = vk::DependencyInfo::builder()
                .buffer_memory_barriers(std::slice::from_ref(&barrier));

            self.context.device().cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
        })?;
        
        Ok((tlas, tlas_buffer))
    }

    // safety: sbt will refer to resources that only lives for 's
    pub unsafe fn add_binding_table_entries<'s, 'a>(
        &'s self,
        shader_groups: &mut Vec<ShaderGroup<'ctx, 'a>>,
        sbt_desc: &mut ShaderBindingTableDescription,
    ) where 's: 'a {
        self.add_raygen_entry(sbt_desc, shader_groups);
        self.add_miss_entry(sbt_desc, shader_groups);
        self.add_instance_entries(sbt_desc, shader_groups);
        self.add_material_entries(sbt_desc, shader_groups);
    }

    // safety: the sbt will refer to resources that only lives as long as this scene.
    unsafe fn add_raygen_entry<'s, 'a>(
        &'s self,
        sbt_desc: &mut ShaderBindingTableDescription,
        shader_groups: &mut Vec<ShaderGroup<'ctx, 'a>>,
    ) where 's: 'a {
        shader_groups.push(ShaderGroup::Raygen { raygen: &self.raygen_shader });
        sbt_desc.push_raygen_entry((shader_groups.len() - 1) as u32, &[]);
    }

    // safety: the sbt will refer to resources that only lives as long as this scene.
    unsafe fn add_miss_entry<'s, 'a>(
        &'s self,
        sbt_desc: &mut ShaderBindingTableDescription,
        shader_groups: &mut Vec<ShaderGroup<'ctx, 'a>>,
    ) where 's: 'a {

        if let Some(environment) = &self.environment {
            shader_groups.push(ShaderGroup::Miss { miss: environment.miss_shader().unwrap() });
            sbt_desc.push_miss_entry((shader_groups.len() - 1) as u32, &[]);
        }
    }

    // safety: the sbt will refer to resources that only lives as long as this scene.
    unsafe fn add_instance_entries<'s, 'a>(
        &'s self, 
        sbt_desc: &mut ShaderBindingTableDescription,
        shader_groups: &mut Vec<ShaderGroup<'ctx, 'a>>,
    ) where 's: 'a {
        shader_groups.push(ShaderGroup::TriangleHit { closest_hit: &self.closest_hit_shader });

        for instance in &self.instances {
            let mesh = &self.meshes[instance.mesh.index];

            let sbt_data = HitGroupSbtData {
                material_index: instance.material.index as u32,
                face_address: mesh.index_address(),
                position_address: mesh.position_address(),
                normal_address: mesh.normal_address(),
            };
            
            sbt_desc.push_hit_group_entry((shader_groups.len() - 1) as u32, as_u8_slice(&sbt_data));
        }
    }

    // safety: the sbt will refer to resources that only lives as long as this scene.
    unsafe fn add_material_entries<'a>(
        &'a self,
        sbt_desc: &mut ShaderBindingTableDescription,
        shader_groups: &mut Vec<ShaderGroup<'ctx, 'a>>,
    ) {

        let shader_group_begin = shader_groups.len();

        for material_type in &self.material_types {
            shader_groups.push(ShaderGroup::Callable { callable: &material_type.evaluation_shader });
            shader_groups.push(ShaderGroup::Callable { callable: &material_type.sample_shader });    
        }

        for material in &self.materials {

            let evaluation_index = shader_group_begin + 2 * material.material_type.index;
            let sample_index = evaluation_index + 1;

            let sbt_data = MicrofacetData {
                ior: material.ior,
                roughness: material.roughness,
            };

            sbt_desc.push_callable_entry(evaluation_index as u32, unsafe { util::as_u8_slice(&sbt_data) });
            sbt_desc.push_callable_entry(sample_index as u32, unsafe { util::as_u8_slice(&sbt_data) });
        }
    }



}


pub struct SceneDescriptorSet<'ctx, 'a> {
    scene: &'a Scene<'ctx>,
    
    #[allow(unused)]
    output_view: ImageView<'ctx>,

    #[allow(unused)]
    tlas_buffer: DeviceBuffer<'ctx>,
    tlas: vk::AccelerationStructureKHR,

    #[allow(unused)]
    texture_views: Vec<ImageView<'ctx>>,

    layout: vk::DescriptorSetLayout,

    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
}

impl<'ctx> SceneDescriptorSet<'ctx, '_> {

    pub fn context(&self) -> &'ctx DeviceContext {
        self.scene.context
    }

    pub fn tlas(&self) -> vk::AccelerationStructureKHR {
        self.tlas
    }

    pub unsafe fn layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }

    pub unsafe fn inner(&self) -> vk::DescriptorSet {
        self.descriptor_set
    }
}

impl<'ctx, 'a> Drop for SceneDescriptorSet<'ctx, 'a> {
    fn drop(&mut self) {
        unsafe {
            self.context().extensions().acceleration_structure.destroy_acceleration_structure(self.tlas, None);
            self.context().device().destroy_descriptor_pool(self.descriptor_pool, None);
            self.context().device().destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

impl<'ctx> Scene<'ctx> {
    pub fn load<P: AsRef<Path>>(&mut self, path: P, context: &'ctx DeviceContext) -> VkResult<()> {
        let (document, buffers, _images) = gltf::import(path).expect("failed to import scene");
    
        let mut primitive_ids = Vec::new();
        let mut mesh_ranges = Vec::new();
        
        for mesh in document.meshes() {

            let mesh_begin = primitive_ids.len();

            for primitive in mesh.primitives() {
    
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                let face_iter = reader.read_indices()
                    .unwrap()
                    .into_u32()
                    .array_chunks();
                
                let position_iter = reader.read_positions().unwrap();
                let normal_iter = reader.read_normals().unwrap();

                unsafe {
                    primitive_ids.push(self.add_mesh(Mesh::new(context, face_iter, position_iter, normal_iter)?));
                }
            }

            let mesh_end = primitive_ids.len();
            mesh_ranges.push(mesh_begin..mesh_end);
        }

        let material_type = unsafe {
            let entry_point_name = CStr::from_bytes_with_nul_unchecked(b"main\0").to_owned();

            let evaluation_shader = Shader::new(
                &self.context,
                "shader_bin/microfacet_evaluate.rcall.spv",
                entry_point_name.to_owned(),
            )?;
    
            let sample_shader = Shader::new(
                &self.context,
                "shader_bin/multiscatter_sample.rcall.spv",
                entry_point_name.to_owned(),
            )?;
            
            self.add_material_type(MaterialType {
                evaluation_shader,
                sample_shader,
            })
        };

        let default_material = self.add_material(Material {
            ior: 1.54,
            roughness: 0.01,
            material_type,
        });

        let mut material_handles = Vec::new();

        for material in document.materials() {
            material_handles.push(self.add_material(Material {
                ior: 1.54,
                roughness: material.pbr_metallic_roughness().roughness_factor(),
                material_type,
            }));
        }

        if let Some(scene) = document.default_scene() {
            for node in scene.nodes() {
                self.add_node(
                    &mesh_ranges,
                    &primitive_ids,
                    &material_handles,
                    default_material,
                    node,
                    Matrix4::identity()
                );
            }
        }

        Ok(())
    }

    fn add_node(
        &mut self,
        mesh_ranges: &[Range<usize>],
        primitive_handles: &[MeshHandle],
        material_handles: &[MaterialHandle],
        default_material: MaterialHandle,
        node: gltf::Node<'_>,
        mut transform: Matrix4<f32>
    ) {
        let local_transform = node.transform().matrix(); 
        let local_transform = Matrix4::from_fn(|i, j| local_transform[j][i]);
    
        transform = local_transform * transform;

        if let Some(gltf_mesh) = node.mesh() {
            
            let mesh_range = mesh_ranges[gltf_mesh.index()].clone();

            for (gltf_primitive, &primitive_handle) in gltf_mesh.primitives().zip(&primitive_handles[mesh_range]) {
                
                let material_handle = gltf_primitive
                    .material()
                    .index()
                    .map(|index| material_handles[index])
                    .unwrap_or(default_material);

                self.add_instance(primitive_handle, material_handle, transform)
            }
        }
        
        for child in node.children() {
            self.add_node(
                mesh_ranges,
                primitive_handles,
                material_handles,
                default_material,
                child,
                transform
            );
        }
    }
}

