use std::path::Path;

use ash::{vk, prelude::VkResult};
use nalgebra::{Matrix3x4, Matrix4};

use crate::{context::DeviceContext, resource::{DeviceBuffer, UploadBuffer}, util};

use self::mesh::Mesh;

pub mod mesh;

struct GeometryInstance {
    mesh_id: usize,
    transform: Matrix3x4<f32>,
    material_id: usize,
}

pub struct SceneDescription<'a> {
    meshes: Vec<Mesh<'a>>,
    instances: Vec<GeometryInstance>,
}

impl<'a> SceneDescription<'a> {
    pub fn new() -> Self {
        Self {
            meshes: Vec::new(),
            instances: Vec::new(),
        }
    }

    pub fn add_mesh(&mut self, mesh: Mesh<'a>) -> usize {
        self.meshes.push(mesh);
        self.meshes.len() - 1
    }

    pub fn add_instance(&mut self, mesh_id: usize, transform: Matrix4<f32>, material_id: usize) {
        let transform = Matrix3x4::from_fn(|i, j| transform[(i, j)]);

        self.instances.push(GeometryInstance{ mesh_id, transform, material_id });
    }

    pub fn load<P: AsRef<Path>>(&mut self, path: P, context: &'a DeviceContext) -> VkResult<()> {
    
        let (document, buffers, _images) = gltf::import(path).expect("failed to import scene");
    
        let mut mesh_ids = Vec::new();
    
        for mesh in document.meshes() {
            for primitive in mesh.primitives() {
    
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                let face_iter = reader.read_indices()
                    .unwrap()
                    .into_u32()
                    .array_chunks();
                
                let position_iter = reader.read_positions().unwrap();
                let normal_iter = reader.read_normals().unwrap();
                unsafe {
                    let mesh_id = self.add_mesh(Mesh::new(context, face_iter, position_iter, normal_iter)?);
                    mesh_ids.push(mesh_id);
                }
            }
        }
    
        if let Some(scene) = document.default_scene() {
            for node in scene.nodes() {
                self.add_node(&mesh_ids, node, Matrix4::identity());
            }
        }

        Ok(())
    }

    fn add_node(
        &mut self,
        mesh_ids: &[usize],
        node: gltf::Node<'_>,
        mut transform: Matrix4<f32>
    ) {
        let local_transform = node.transform().matrix(); 
        let local_transform = Matrix4::from_fn(|i, j| local_transform[j][i]);
    
        transform = local_transform * transform;
        
        if let Some(gltf_mesh) = node.mesh() {
    
            let mesh_id = mesh_ids[gltf_mesh.index()];
            self.add_instance(mesh_id, transform, 0)
        }
    
        for child in node.children() {
            self.add_node(mesh_ids, child, transform);
        }
    }    

    pub fn build(self, context: &'a DeviceContext) -> VkResult<Scene<'a>> {
        unsafe {

            let (tlas_buffer, tlas) = self.build_tlas(context)?;
            
            Ok(Scene {
                context,
                meshes: self.meshes,
                tlas_buffer,
                tlas,
            })
        }
    }

    unsafe fn build_tlas(&self, context: &'a DeviceContext) -> VkResult<(DeviceBuffer<'a>, vk::AccelerationStructureKHR)> {
        let blas_instance_iter = self.instances.iter()
            .map(|instance|
                vk::AccelerationStructureInstanceKHR {
                    transform: util::matrix_to_vk_transform(instance.transform),
                    instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
                    instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8),
                    acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                        device_handle: self.meshes[instance.mesh_id].get_accel_structure_device_address()
                    },
                }
            );
            
        let mut instance_buffer = UploadBuffer::new(
            &context,
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

        let tlas_sizes = context.extensions().acceleration_structure.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &tlas_build_info,
            &[self.instances.len() as u32],
        );

        println!("Tlas buffer size: {}", tlas_sizes.acceleration_structure_size);

        let tlas_buffer = DeviceBuffer::new(
            context,
            tlas_sizes.acceleration_structure_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        )?;

        let tlas_info = vk::AccelerationStructureCreateInfoKHR::builder()
            .buffer(tlas_buffer.handle())
            .size(tlas_sizes.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);

        let tlas = context.extensions().acceleration_structure.create_acceleration_structure(&tlas_info, None).unwrap();

        let scratch_buffer = DeviceBuffer::new(
            context,
            tlas_sizes.build_scratch_size, 
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;

        tlas_build_info.scratch_data = scratch_buffer.get_device_or_host_address();
        tlas_build_info.dst_acceleration_structure = tlas;

        let build_range = vk::AccelerationStructureBuildRangeInfoKHR::builder()
            .primitive_count(self.instances.len() as u32)
            .build();

        let build_ranges = [std::slice::from_ref(&build_range)];

        context.execute_commands(|cmd_buffer| {
            context.extensions().acceleration_structure.cmd_build_acceleration_structures(
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

            context.device().cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
        })?;
        
        Ok((tlas_buffer, tlas))
    }

}

pub struct Scene<'a> {
    context: &'a DeviceContext,
    meshes: Vec<Mesh<'a>>,
    
    #[allow(unused)]
    tlas_buffer: DeviceBuffer<'a>,
    tlas: vk::AccelerationStructureKHR,
}

impl<'a> Scene<'a> {

    pub fn get_mesh(&self, mesh_id: usize) -> &Mesh<'a> {
        &self.meshes[mesh_id]
    }

    pub fn tlas(&self) -> vk::AccelerationStructureKHR {
        self.tlas
    }

}

impl<'a> Drop for Scene<'a> {
    fn drop(&mut self) {
        unsafe {
            self.context.extensions().acceleration_structure.destroy_acceleration_structure(self.tlas, None);
        }    
    }
}

// pub struct GeometryInstance {
//     mesh: usize,

// }

// pub struct Scene<'a> {
//     context: &'a DeviceContext,
//     meshes: Vec<Mesh<'a>>,
//     instances: Vec<GeometryInstance>,
//     accel_structure: vk::AccelerationStructureKHR,
// }

// impl<'a> Scene<'a> {

//     pub fn new(
//         meshes: Vec<Mesh<'a>>,
//         instances: GeometryInstance,
//         context: &'a DeviceContext
//     ) -> VkResult<Self> {

        

//     }


//     fn create_and_build_accel_structure() {

//     }

// }
