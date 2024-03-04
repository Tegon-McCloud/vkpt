
use std::path::{Path, PathBuf};

use anyhow::Result;

extern crate shaderc;

enum ShaderKind {
    Header,
    Source(shaderc::ShaderKind)
}

fn get_shader_kind<P: AsRef<Path>>(path: P) -> ShaderKind {

    let extension = path.as_ref()
        .extension()
        .and_then(|os_str| os_str.to_str());

    if extension == Some("glsl") {
        ShaderKind::Header
    } else {
        ShaderKind::Source(match extension {
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
        })
    }
}

fn main() -> Result<()> {

    let in_dir = PathBuf::from("shader_src");
    let out_dir = PathBuf::from("shader_bin");
    std::fs::create_dir_all(&out_dir)?;

    println!("cargo:rerun-if-changed=shader_src");
    println!("cargo:rerun-if-changed=shader_bin");

    let shader_compiler = shaderc::Compiler::new().unwrap();
    let mut compile_options = shaderc::CompileOptions::new().unwrap();

    compile_options.set_target_spirv(shaderc::SpirvVersion::V1_6);
    compile_options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_3 as u32);
    compile_options.set_include_callback(|include_name, _include_type, _src_name, _depth| {

        let include_path = std::fs::canonicalize(in_dir.as_path().join(include_name)).map_err(|err| err.to_string())?;
        let content = std::fs::read_to_string(&include_path).map_err(|err| err.to_string())?;

        Ok(shaderc::ResolvedInclude {
            resolved_name: include_path.to_string_lossy().into_owned(),
            content,
        })
    });


    for entry in std::fs::read_dir(&in_dir)? {

        let entry = entry?;

        if !entry.file_type()?.is_file() {
            continue;
        }

        let in_path = entry.path();


        let kind = get_shader_kind(&in_path);

        if let ShaderKind::Source(shaderc_kind) = kind {
            let name = in_path.file_name().and_then(|os_str| os_str.to_str()).unwrap_or_default();
            let source = std::fs::read_to_string(&in_path)?;
            
            let binary = shader_compiler.compile_into_spirv(
                &source,
                shaderc_kind,
                name,
                "main",
                Some(&compile_options)
            )?;

            std::fs::write(
                out_dir.join(String::from_iter([name, ".spv"])),
                binary.as_binary_u8()
            )?;
        }


    }

    Ok(())
}
