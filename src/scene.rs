use std::{ffi::CStr, ops::Range, path::Path};

use ash::{vk, prelude::VkResult};
use nalgebra::{Matrix3x4, Matrix4};

use crate::{context::DeviceContext, pipeline::{Pipeline, Shader, ShaderData, ShaderGroup, ShaderResourceLayout}, resource::{DeviceBuffer, UploadBuffer}, shader_binding_table::{ShaderBindingTable, ShaderBindingTableDescription}, util::{self, as_u8_slice}};

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
pub struct MaterialHandle {
    index: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct MaterialTypeHandle {
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
    materials: Vec<Material>,
    instances: Vec<GeometryInstance>,
    camera: Camera,
    environment: Option<Environment<'ctx>>,
    raygen_shader: Shader<'ctx>, // abstract this into camera
    // miss_shader: Shader<'ctx>, // abstract this into light
    closest_hit_shader: Shader<'ctx>
}

impl<'ctx> Scene<'ctx> {

    pub fn new(context: &'ctx DeviceContext, raygen_desc_set_layout: vk::DescriptorSetLayout) -> Self {

        let raygen_shader = unsafe {
            Shader::new(
                context,
                "shader_bin/raytrace.rgen.spv",
                CStr::from_bytes_with_nul_unchecked(b"main\0").to_owned(),
                ShaderResourceLayout::new(vec![raygen_desc_set_layout], 68),
            ).unwrap()
        };

        let closest_hit_shader = unsafe {
            Shader::new(
                context,
                "shader_bin/raytrace.rchit.spv",
                CStr::from_bytes_with_nul_unchecked(b"main\0").to_owned(),
                ShaderResourceLayout::default(),
            ).unwrap()
        };

        Self {
            context,
            meshes: Vec::new(),
            material_types: Vec::new(),
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

    pub fn compile<'a>(&'a self) -> VkResult<CompiledScene<'ctx, 'a>> {
        
        let (tlas, tlas_buffer) = unsafe { self.build_tlas()? };
        let (sbt, pipeline) = unsafe { self.make_sbt()? };

        Ok(CompiledScene {
            scene: self, // ensures that the compiled scene cannot outlive scene resources

            tlas_buffer, // ensures that tlas_buffer will live as long as tlas
            tlas,
            
            pipeline,
            sbt,
        })
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
    
    // safety: the sbt will refer to resources that only lives as long as this scene.
    unsafe fn make_sbt(&self) -> VkResult<(ShaderBindingTable<'ctx>, Pipeline<'ctx>)> {
        let mut shader_groups = Vec::new();
        let mut sbt_desc = ShaderBindingTableDescription::new();

        self.add_raygen_entry(&mut sbt_desc, &mut shader_groups);
        self.add_miss_entry(&mut sbt_desc, &mut shader_groups);
        self.add_instance_entries(&mut sbt_desc, &mut shader_groups);
        self.add_material_entries(&mut sbt_desc, &mut shader_groups);

        let pipeline = Pipeline::new(self.context, &shader_groups)?;
        let sbt = sbt_desc.build(self.context, &pipeline)?;

        Ok((sbt, pipeline))
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


pub struct CompiledScene<'ctx, 'a> {
    scene: &'a Scene<'ctx>,
    
    #[allow(unused)]
    tlas_buffer: DeviceBuffer<'ctx>,
    tlas: vk::AccelerationStructureKHR,
    
    pipeline: Pipeline<'ctx>,
    sbt: ShaderBindingTable<'ctx>,
}

impl<'ctx> CompiledScene<'ctx, '_> {

    pub fn tlas(&self) -> vk::AccelerationStructureKHR {
        self.tlas
    }

    pub fn sbt(&self) -> &ShaderBindingTable<'ctx> {
        &self.sbt
    } 

    pub fn pipeline(&self) -> &Pipeline<'ctx> {
        &self.pipeline
    }

    pub fn camera_data(&self) -> impl ShaderData {
        self.scene.camera.serialize()
    }

    pub unsafe fn bind(&self, cmd_buffer: vk::CommandBuffer) {

    }
}

impl<'ctx, 'a> Drop for CompiledScene<'ctx, 'a> {
    fn drop(&mut self) {
        unsafe {
            self.scene.context.extensions().acceleration_structure.destroy_acceleration_structure(self.tlas, None);
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
                ShaderResourceLayout::default(),
            )?;
    
            let sample_shader = Shader::new(
                &self.context,
                "shader_bin/microfacet_sample.rcall.spv",
                entry_point_name.to_owned(),
                ShaderResourceLayout::default(),
            )?;
            
            self.add_material_type(MaterialType {
                evaluation_shader,
                sample_shader,
            })
        };

        let default_material = self.add_material(Material {
            ior: 1.54,
            roughness: 0.1,
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

