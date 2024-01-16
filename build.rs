
use std::path::{Path, PathBuf};

use anyhow::Result;

extern crate shaderc;

fn get_shader_kind<P: AsRef<Path>>(path: P) -> shaderc::ShaderKind {

    let extension = path.as_ref()
        .extension()
        .and_then(|os_str| os_str.to_str());


    match extension {
        Some("vert") => shaderc::ShaderKind::Vertex,
        Some("frag") => shaderc::ShaderKind::Fragment,
        Some("comp") => shaderc::ShaderKind::Compute,
        Some("rgen") => shaderc::ShaderKind::RayGeneration,
        Some("rint") => shaderc::ShaderKind::Intersection,
        Some("rahit") => shaderc::ShaderKind::AnyHit,
        Some("rchit") => shaderc::ShaderKind::ClosestHit,
        Some("rmiss") => shaderc::ShaderKind::Miss,
        Some("rcall") => shaderc::ShaderKind::Callable,
        _ => shaderc::ShaderKind::InferFromSource,
    }
}

fn main() -> Result<()> {

    println!("cargo:rerun-if-changed=shader_src");
    println!("cargo:rerun-if-changed=shader_bin");
    

    let shader_compiler = shaderc::Compiler::new().unwrap();
    let mut compile_options = shaderc::CompileOptions::new().unwrap();
    
    compile_options.set_target_spirv(shaderc::SpirvVersion::V1_6);
    compile_options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_3 as u32);

    // options.add_macro_definition("EP", Some("main"));

    let out_dir = PathBuf::from("shader_bin");
    std::fs::create_dir_all(&out_dir)?;

    for entry in std::fs::read_dir("shader_src")? {

        let entry = entry?;

        if !entry.file_type()?.is_file() {
            continue;
        }

        let in_path = entry.path();

        let shader_kind = get_shader_kind(&in_path);
        let shader_name = in_path.file_name().and_then(|os_str| os_str.to_str()).unwrap_or_default();
        let shader_source = std::fs::read_to_string(&in_path)?;

        let shader_binary = shader_compiler.compile_into_spirv(
            &shader_source,
            shader_kind,
            shader_name,
            "main",
            Some(&compile_options)
        )?;
        
        std::fs::write(
            out_dir.join(String::from_iter([shader_name, ".spv"])),
            shader_binary.as_binary_u8()
        )?;
    }

    Ok(())
}
