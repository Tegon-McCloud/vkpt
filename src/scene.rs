

// use ash::{vk, prelude::VkResult};

// use crate::context::DeviceContext;

pub struct MeshInstance {
    mesh: usize,
    transform: [f32; 12],
}

pub struct TlasBuilder<'a> {
    context: &'a DeviceContext,
    meshes: Vec<Mesh<'a>>,
    
}

pub struct Tlas {

    
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
