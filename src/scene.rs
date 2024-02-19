use std::{ffi::CStr, ops::Range, path::Path};

use ash::{vk, prelude::VkResult};
use nalgebra::{Matrix3x4, Matrix4, Vector3};

use crate::{context::DeviceContext, pipeline::{Pipeline, Shader, ShaderGroup}, resource::{DeviceBuffer, UploadBuffer}, shader_binding_table::{ShaderBindingTable, ShaderBindingTableDescription}, util::{self, as_u8_slice}};

use self::{material::Material, mesh::Mesh};

pub mod mesh;
pub mod material;



struct GeometryInstance {
    mesh: MeshHandle,
    material: MaterialHandle,
    transform: Matrix3x4<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct MeshHandle {
    index: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct MaterialHandle {
    index: usize,
}

#[repr(C)]
struct HitGroupSbtData {
    material_index: u32,
    face_address: vk::DeviceAddress,
    position_address: vk::DeviceAddress,
    normal_address: vk::DeviceAddress,
}

#[repr(C)]
struct LambertianData {
    color: [f32; 4],
}

pub struct Scene<'ctx> {
    context: &'ctx DeviceContext,
    
    meshes: Vec<Mesh<'ctx>>,
    materials: Vec<Material>,

    instances: Vec<GeometryInstance>,
}

impl<'ctx> Scene<'ctx> {

    pub fn new(context: &'ctx DeviceContext) -> Self {
        Self {
            context,
            meshes: Vec::new(),
            materials: Vec::new(),
            instances: Vec::new(),
        }
    }

    pub fn add_mesh(&mut self, mesh: Mesh<'ctx>) -> MeshHandle {
        self.meshes.push(mesh);
        MeshHandle { index: self.meshes.len() - 1 }
    }

    pub fn add_material(&mut self, material: Material) -> MaterialHandle {
        self.materials.push(material);
        MaterialHandle { index: self.materials.len() - 1 }
    }

    pub fn add_instance(&mut self, mesh: MeshHandle, material: MaterialHandle, transform: Matrix4<f32>) {
        let transform = Matrix3x4::from_fn(|i, j| transform[(i, j)]);
        self.instances.push(GeometryInstance{ mesh, material, transform });
    }

    pub fn get_mesh(&self, mesh_id: usize) -> &Mesh<'ctx> {
        &self.meshes[mesh_id]
    }

    pub unsafe fn compile<'a>(&'a self, raygen_desc_set_layout: vk::DescriptorSetLayout) -> VkResult<CompiledScene<'ctx, 'a>> {
        
        let (tlas, tlas_buffer) = self.build_tlas()?;
        let (sbt, pipeline) = self.make_sbt(raygen_desc_set_layout)?;

        Ok(CompiledScene {
            scene: self,

            tlas_buffer,
            tlas,
            
            pipeline,
            sbt
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


    unsafe fn make_sbt(&self, raygen_desc_set_layout: vk::DescriptorSetLayout) -> VkResult<(ShaderBindingTable<'ctx>, Pipeline<'ctx>)> {
        let mut shader_groups = Vec::new();
        let mut sbt_desc = ShaderBindingTableDescription::new();

        let entry_point_name = CStr::from_bytes_with_nul_unchecked(b"main\0").to_owned();

        let raygen_shader = Shader::new(
            &self.context,
            "shader_bin/raytrace.rgen.spv",
            entry_point_name.clone(),
            vec![raygen_desc_set_layout],
        )?;
        
        let miss_shader = Shader::new(
            &self.context,
            "shader_bin/raytrace.rmiss.spv",
            entry_point_name.clone(),
            vec![],
        )?;

        let closest_hit_shader = Shader::new(
            &self.context,
            "shader_bin/raytrace.rchit.spv",
            entry_point_name.clone(),
            vec![],
        )?;

        let lambertian_shader = Shader::new(
            &self.context,
            "shader_bin/lambertian.rcall.spv",
            entry_point_name.to_owned(),
            vec![],
        )?;

        shader_groups.push(ShaderGroup::Raygen { raygen: &raygen_shader });
        let raygen_group_index = shader_groups.len() - 1;

        shader_groups.push(ShaderGroup::Miss { miss: &miss_shader });
        let miss_group_index = shader_groups.len() - 1;

        shader_groups.push(ShaderGroup::TriangleHit { closest_hit: &closest_hit_shader });
        let hit_group_index = shader_groups.len() - 1;

        shader_groups.push(ShaderGroup::Callable { callable: &lambertian_shader });
        let lambertian_group_index = shader_groups.len() - 1;
        
        let pipeline = Pipeline::new(self.context, &shader_groups)?;

        sbt_desc.push_raygen_entry(raygen_group_index as u32, &[]);
        sbt_desc.push_miss_entry(miss_group_index as u32, &[]);
        
        for instance in &self.instances {
            let mesh = &self.meshes[instance.mesh.index];

            let sbt_data = HitGroupSbtData {
                material_index: instance.material.index as u32,
                face_address: mesh.index_address(),
                position_address: mesh.position_address(),
                normal_address: mesh.normal_address(),
            };
            
            sbt_desc.push_hit_group_entry(hit_group_index as u32, as_u8_slice(&sbt_data));
        }

        for material in &self.materials {
            let sbt_data = LambertianData {
                color: [material.base_color.x, material.base_color.y, material.base_color.z, 1.0],
            };

            sbt_desc.push_callable_entry(lambertian_group_index as u32, util::as_u8_slice(&sbt_data));
        }

        let sbt = sbt_desc.build(self.context, &pipeline)?;

        Ok((sbt, pipeline))
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

    pub fn pipeline(&self) -> &Pipeline<'ctx>  {
        &self.pipeline
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


        let mut material_ids = Vec::new();

        for material in document.materials() {
            let base_color_factor = material.pbr_metallic_roughness().base_color_factor();

            material_ids.push(self.add_material(Material {
                base_color: Vector3::new(base_color_factor[0], base_color_factor[1], base_color_factor[2]),
            }));
        }

        if let Some(scene) = document.default_scene() {
            for node in scene.nodes() {
                self.add_node(&mesh_ranges, &primitive_ids, &material_ids, node, Matrix4::identity());
            }
        }

        Ok(())
    }

    fn add_node(
        &mut self,
        mesh_ranges: &[Range<usize>],
        primitive_handles: &[MeshHandle],
        material_handles: &[MaterialHandle],
        node: gltf::Node<'_>,
        mut transform: Matrix4<f32>
    ) {
        let local_transform = node.transform().matrix(); 
        let local_transform = Matrix4::from_fn(|i, j| local_transform[j][i]);
    
        transform = local_transform * transform;

        if let Some(gltf_mesh) = node.mesh() {
            
            let mesh_range = mesh_ranges[gltf_mesh.index()].clone();

            for (gltf_primitive, &primitive_handle) in gltf_mesh.primitives().zip(&primitive_handles[mesh_range]) {
                
                let material_id = material_handles[gltf_primitive.material().index().unwrap()];
                self.add_instance(primitive_handle, material_id, transform)
            }
        }
        
        for child in node.children() {
            self.add_node(mesh_ranges, primitive_handles, material_handles, child, transform);
        }
    }
}

